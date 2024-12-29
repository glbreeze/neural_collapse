import os
import json
import wandb
import argparse
import numpy as np
import torch
from scipy import stats

from ood_utils.common import setup_seed, get_num_cls, get_test_labels
from ood_utils.detection_util import get_Mahalanobis_score, get_mean_prec, get_and_print_results, get_mean_cov_it, get_feat_labels, get_measures 
from ood_utils.file_ops import save_as_dataframe, setup_log
from ood_utils.plot_util import plot_distribution
from ood_utils.train_eval_util import  set_model_clip, set_train_loader, set_val_loader, set_ood_loader_ImageNet
# sys.path.append(os.path.dirname(__file__))


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', default='bird200', type=str, help='in-distribution dataset',
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100', 'pet37', 'food101', 'car196', 'bird200'])

    parser.add_argument('--root-dir', default="/vast/lg154/datasets", type=str, help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood", type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type = int, help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=512, type=int, help='mini-batch size')
    parser.add_argument('--T', type=int, default=1, help='temperature parameter')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16', choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=['MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'], help='score options')
    
    # for new image to text embedding distance 
    parser.add_argument('--text_embed', type=str, default='default', help='which text embed to compare to')  # default | mean | min

    # for Mahalanobis score
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim； 512 for ViT-B and 768 for ViT-L')
    parser.add_argument('--normalize', type = bool, default = False, help='whether use normalized features for Maha score')
    parser.add_argument('--generate', type = bool, default = True, help='whether to generate class-wise means or read from files for Maha score')
    parser.add_argument('--template_dir', type = str, default = 'img_templates', help='the loc of stored classwise mean and precision matrix')
    parser.add_argument('--subset', default = False, type =bool, help = "whether uses a subset of samples in the training set")
    parser.add_argument('--max_count', default = 100, type =int, help = "how many samples are used to estimate classwise mean and precision matrix")
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)

    return args

def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='lg_ood', name=args.name)
    wandb.config.update(args)

    net, img_preprocess, processor = set_model_clip(args)
    net.eval()
    
    def collate_fn(batch):
        return processor(images=batch, return_tensors="pt").pixel_values

    if args.in_dataset in ['ImageNet10']:
        out_datasets = ['ImageNet20']
    elif args.in_dataset in ['ImageNet20']:
        out_datasets = ['ImageNet10']
    elif args.in_dataset in [ 'ImageNet', 'ImageNet100', 'bird200', 'car196', 'food101', 'pet37']:
         out_datasets = ['iNaturalist'] #,'SUN', 'places365', 'dtd']
    
    # ========== ID features 
    train_loader = set_train_loader(args, img_preprocess, subset = args.subset)
    id_feats, id_labels = get_feat_labels(train_loader, net, device, args)
    
    class_names = get_test_labels(args, train_loader)
    class_names = [name for name in class_names]  # convert it to list
    
    # ============= Get the text embeddings
    if args.text_embed in ['default']: 
        prompts = [f"a photo of {class_name}" for class_name in class_names]
        text_inputs = processor.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)
        with torch.no_grad():
            text_mean_embeddings = net.get_text_features(text_inputs)
            text_mean_embeddings = text_mean_embeddings / text_mean_embeddings.norm(dim=-1, keepdim=True)
    else: 
        # ====== compute embedding of captions
        caption_file = f"{args.in_dataset.lower()}_captions.json"
        with open(caption_file, 'r') as f:
            all_captions = json.load(f)
        
        cls_embeddings = {}
        for category, captions in all_captions.items():
            text_inputs = processor.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(device)
            with torch.no_grad():
                text_features = net.get_text_features(text_inputs).float()  # Shape: [num_captions, embedding_dim]
                text_features /= text_features.norm(dim=-1, keepdim=True)   # Normalize the embeddings

            # Compute mean and variance of embeddings
            mean_embedding = text_features.mean(dim=0)  # Shape: [embedding_dim]
            mean_embedding = mean_embedding / torch.norm(mean_embedding, dim=-1)
            var_embedding = text_features.var(dim=0)   # Shape: [embedding_dim]
            
            # compute var of the cosine distance
            cos_dist = 1- torch.matmul(text_features, mean_embedding)
            var_dist = torch.var(cos_dist, unbiased=False) 
            
            cls_embeddings[category] = {
                "mean": mean_embedding,
                "variance": var_embedding, 
                "var_dist": var_dist.item(), 
                "min_dist": cos_dist.min().item(), 
                "max_dist": cos_dist.max().item(),
                "caption_feats": text_features
            }
        text_mean_embeddings = torch.cat([value['mean'].view(1, -1) for key, value in cls_embeddings.items()], dim=0)
        text_embeddings = torch.cat([value['caption_feats'] for key, value in cls_embeddings.items()], dim=0)
    
    # ============= Get the distance between ID img2text
    if args.text_embed in ['default', 'mean']: 
        id_scores = id_feats @ text_mean_embeddings.T
        id_scores = id_scores.max(dim=-1).values
    elif args.text_embed in ['max']: 
        id_scores = id_feats @ text_embeddings.T
        id_scores = id_scores.max(dim=-1).values
    id_scores = id_scores.cpu().numpy()
    log.debug(f"distribution of ID distance: {stats.describe(id_scores)}")
    
    # ============= get OD features 
    for out_dataset in out_datasets:
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, img_preprocess, 
                                            root=os.path.join(args.root_dir, 'ImageNet_OOD_dataset'), subset=args.subset)
        od_feats, _ = get_feat_labels(ood_loader, net, device, args)
    
        if args.text_embed in ['default', 'mean']: 
            od_scores = od_feats @ text_mean_embeddings.T
            od_scores = od_scores.max(dim=-1).values
        elif args.text_embed in ['min']: 
            od_scores = od_feats @ text_embeddings.T
            od_scores = od_scores.max(dim=-1).values
        od_scores = od_scores.cpu().numpy()
        log.debug(f"distribution of OD distance: {stats.describe(od_scores)}")
        
        auroc, aupr, fpr, threshold = get_measures(id_scores, od_scores, recall_level=0.95)
        log.debug(f"Using {args.text_embed} method, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR@0.95: {fpr:.4f}, threshold: {threshold:.4f}")
        
        plot_path = plot_distribution(args, id_scores, od_scores, out_dataset)
        wandb.log({f"fpr:{fpr:.2f},auroc:{auroc:.2f},aupr:{aupr:.2f}--{args.in_dataset} vs {out_dataset}": wandb.Image(plot_path)})


if __name__ == '__main__':
    main()
