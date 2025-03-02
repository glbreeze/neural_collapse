# -*- coding: utf-8 -*-
import os
import wandb
import torch
import random
import pickle
import argparse
from nc_metric import analysis_feat
from model import ResNet, MLP
from dataset.data import get_dataloader
from temperature_scaling import ModelWithTemperature
from utils import Graph_Vars, set_log_path, log, print_args, get_scheduler, get_logits_labels_feats, AverageMeter
from utils import CrossEntropyLabelSmooth, CrossEntropyHinge, KoLeoLoss, FocalLoss

import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from metrics import ECELoss, AdaptiveECELoss


def classwise_acc(targets, preds):
    eps = np.finfo(np.float64).eps
    cf = confusion_matrix(targets, preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    return cls_acc


def train_one_epoch(model, criterion, train_loader, optimizer, args, state):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_loss = AverageMeter('Loss', ':.4e')
    train_kl_loss = AverageMeter('KoLeo_Loss', ':.4e')
    train_acc = AverageMeter('Train_acc', ':.4e')
    koleo_loss = KoLeoLoss(type=args.koleo_type)
    
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != args.batch_size:
            continue

        data, target = data.to(device), target.to(device)
        out, feat = model(data, ret_feat=True)

        optimizer.zero_grad()
        
        ce_loss = criterion(out, target) 
        
        kl_loss = torch.tensor(0.0).to(device)
        
        if args.koleo_wt != 0:  
            # Compute global mean using the current batch
            if args.kl_cls_wt in ['null', 'n']:
                kl_cls_weight = None
            elif args.kl_cls_wt in ['inv']:
                kl_cls_weight = np.median(state['var_cls']) / state['var_cls']
                kl_cls_weight = torch.tensor(kl_cls_weight).to(device)
            else:
                raise ValueError(f"Unsupported value for args.kl_cls_wt: {args.kl_cls_wt}")
            
            if args.koleo_type == 'c':
                M = torch.zeros(len(feat), args.num_classes).to(device)
                M[torch.arange(len(feat)), target] = 1            # [B, C]
                M = torch.nn.functional.normalize(M, p=1, dim=0)  # [B, C]
                cls_mean = torch.einsum('cb,bd->cd', M.T, feat)   # [C, B] * [B, D]
                cls_in_batch = torch.unique(target)
                cls_mean = cls_mean[cls_in_batch]
                glb_mean = torch.mean(cls_mean, dim=0)
                kl_loss = koleo_loss(feat-glb_mean.detach(), labels=target, cls_weight=kl_cls_weight)
                
            elif args.koleo_type == 'm': 
                state['GLB_mean'] = args.kl_beta * state['GLB_mean'] + (1 - args.kl_beta) * torch.mean(feat.detach(), dim=0)
                kl_loss = koleo_loss(feat-state['GLB_mean'].detach(),  labels=target, cls_weight=kl_cls_weight)
                
            elif args.koleo_type == 'd':
                kl_loss = koleo_loss(feat, labels=target, cls_weight=kl_cls_weight)
                
            else: 
                state['GLB_mean'] = args.kl_beta * state['GLB_mean'] + (1 - args.kl_beta) * torch.mean(feat.detach(), dim=0)
                for class_id in range(args.num_classes):
                    class_mask = (target == class_id)
                    
                    if class_mask.sum() > 0:  # Avoid empty class handling (division by zero)
                        class_mean = torch.mean(feat[class_mask].detach(), dim=0)
                        state['CLS_mean'][class_id] = args.kl_beta * state['CLS_mean'][class_id] + (1 - args.kl_beta) * class_mean

                kl_loss = koleo_loss(feat-state['GLB_mean'].detach(), (state['CLS_mean']-state['GLB_mean']).detach(), 
                                     labels=target, cls_weight=kl_cls_weight)
            
        total_loss = ce_loss + kl_loss * args.koleo_wt  
        train_loss.update(ce_loss.item(), target.size(0))
        train_kl_loss.update(kl_loss.item(), target.size(0))

        total_loss.backward()
        optimizer.step()

        train_acc.update(torch.sum(out.argmax(dim=-1) == target).item() / target.size(0),
                         target.size(0)
                         )
    return train_loss, train_kl_loss, train_acc


def main(args):
    MAX_TEST_ACC, MIN_TEST_LOSS, MIN_TEST_ECE =0.0, 100.0, 100.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==================== data loader ====================
    train_loader, test_loader = get_dataloader(args)

    # ====================  define model ====================
    if args.model.lower() == 'mlp':
        model = MLP(hidden = args.width, depth = args.depth, fc_bias=args.bias, num_classes=args.num_classes)
    else:
        model = ResNet(pretrained=False, num_classes=args.num_classes, backbone=args.model, args=args)
    model = model.to(device)
    
    state = {}
    if args.koleo_type in ['m', 'l1', 'l2', 'cs', 'd2']:
        state['GLB_mean'] = torch.zeros(model.feat_dim).to(device)
    if args.koleo_type in ['l1', 'l2', 'cs', 'd2']:
        state['CLS_mean'] = torch.zeros(args.num_classes, model.feat_dim).to(device)
    state['var_cls'] = np.ones(args.num_classes)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ls':
        criterion = CrossEntropyLabelSmooth(args.num_classes, epsilon=args.eps)
    elif args.loss == 'fl':
        criterion = FocalLoss(gamma=args.eps)
    elif args.loss == 'ceh':
        criterion = CrossEntropyHinge(args.num_classes, epsilon=0.05)
    elif args.loss == 'hinge':
        criterion = nn.MultiMarginLoss(p=1, margin=args.margin, reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss()
    ECE_dt = {'ece': ECELoss, 'adaece': AdaptiveECELoss}
    ece_criterion15 = ECE_dt[args.ece_type](n_bins=15).cuda()
    ece_criterion20 = ECE_dt[args.ece_type](n_bins=20).cuda()
    ece_criterion25 = ECE_dt[args.ece_type](n_bins=25).cuda()

    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = get_scheduler(args, optimizer)

    # ====================  start training ====================
    wandb.watch(model, criterion, log=None)
    for epoch in range(args.max_epochs):
        
        # ==================== train         
        train_loss, train_kl_loss, train_acc = train_one_epoch(model, criterion, train_loader, optimizer, args, state=state)
        lr_scheduler.step()
        
        # ================= check NC
        if (epoch + 1) % args.log_freq == 0 or epoch == 0:
            logits, labels, feats = get_logits_labels_feats(train_loader, model)  # on cuda
            nc_train, centroid = analysis_feat(labels, feats, args, W=model.classifier.weight.detach(),centroid=None)
            train_loss1 = F.cross_entropy(logits, labels, reduction='mean').item() 
            train_acc1 = (logits.argmax(dim=-1) == labels).sum().item()/len(labels)
            train_loss2 = F.cross_entropy(logits/(nc_train['h_norm']*nc_train['w_norm'] + 1e-8), labels, reduction='mean').item()
            
            logits, labels, feats = get_logits_labels_feats(test_loader, model)   # on cuda
            val_loss = F.cross_entropy(logits, labels, reduction='mean').item()   # on cuda 
            val_acc = (logits.argmax(dim=-1) == labels).sum().item()/len(labels)  # on cuda
            val_acc_cls = classwise_acc(labels.cpu().numpy(), logits.argmax(dim=-1).cpu().numpy())
            
            nc_val, _ = analysis_feat(labels, feats, args, W=model.classifier.weight.detach(), centroid=centroid)
            state['var_cls'] = nc_train['var_cls']
            
            wandb.log({
                'overall/lr': optimizer.param_groups[0]['lr'],
                'overall/val_loss': val_loss,
                'overall/val_acc': val_acc,
                'overall/train_loss': train_loss.avg,
                'overall/train_acc': train_acc.avg,
                
                'overall/train_acc1':train_acc1,
                'overall/train_loss1':train_loss1, 
                'overall/train_loss2':train_loss2,
                
                'train_nc/nc1': nc_train['nc1'],  'train_nc/nc2': nc_train['nc2'],
                'train_nc/nc3': nc_train['nc3'],  'train_nc/nc2h': nc_train['nc2h'],
                'train_nc/ncc_acc': nc_train['ncc_acc'],
                'other_nc/w_norm': nc_train['w_norm'], 'other_nc/h_norm': nc_train['h_norm'],
                'other_nc/nc2w': nc_train['nc2w'],
                
                'val_nc/nc1': nc_val['nc1'], 'val_nc/nc2': nc_val['nc2'],
                'val_nc/nc3': nc_val['nc3'], 'val_nc/nc2h': nc_val['nc2h'],
                'val_nc/ncc_acc': nc_val['ncc_acc'],
            }, step=epoch)
            
            # ================= check ECE
            if args.ece_flag:
                val_ece15 = ece_criterion15(logits, labels).item()  # on cuda
                val_ece20 = ece_criterion20(logits, labels).item()  # on cuda
                val_ece25 = ece_criterion25(logits, labels).item()  # on cuda
                # post process ece
                scaled_model = ModelWithTemperature(model)
                val_ece20_post, _ = scaled_model.set_temperature(test_loader, cross_validate='ece', 
                                                                n_bins=20, ece_type=args.ece_type)
                wandb.log({
                    'ece/val_ece15': val_ece15,
                    'ece/val_ece20': val_ece20,
                    'ece/val_ece25': val_ece25,
                    'ece/val_ece20_post': val_ece20_post,
                    'ece/best_temp': scaled_model.temperature,
                    }, step=epoch)
        
        # ==================== log train loss 
            
            # if (epoch + 1) % (args.log_freq*5) == 0 or epoch == 0:
            #     data = [[label, nc1_tr, nc1_va, var_tr, var_va, acc] for (label, nc1_tr, nc1_va, var_tr, var_va, acc) in
            #             zip(np.arange(args.num_classes), nc_train['nc1_cls'], nc_val['nc1_cls'], nc_train['var_cls'], nc_val['var_cls'], val_acc_cls)]
            #     table = wandb.Table(data=data, columns=["label", "nc1_train", 'nc1_val', 'var_train', 'var_val', 'acc'])
            #     wandb.log({"per class nc1 train": wandb.plot.bar(table, "label", "var_train", title="var train")}, step=epoch)
            #     wandb.log({"per class nc1 val": wandb.plot.bar(table, "label", "var_val", title="var val")}, step=epoch)
            #     wandb.log({"per class acc val": wandb.plot.bar(table, "label", "acc", title="val acc")}, step=epoch)

            # try:
            #     nc_train_all.load_dt(nc_train, epoch=epoch)
            #     nc_val_all.load_dt(nc_val, epoch=epoch)
            # except:
            #     nc_train_all = Graph_Vars(nc_train)
            #     nc_val_all   = Graph_Vars(nc_val)
            #     nc_train_all.load_dt(nc_train, epoch=epoch)
            #     nc_val_all.load_dt(nc_val, epoch=epoch)
        
        # ================= store the model
        # if (val_acc > MAX_TEST_ACC and epoch >= 100) and args.save_ckpt > 0:
        #         MAX_TEST_ACC = val_acc
        #         BEST_NET = model.state_dict()
        #         torch.save(BEST_NET, os.path.join(args.output_dir, "best_acc_net.pt"))
        #         log('EP{} Store model (best TEST ACC) to {}'.format(epoch, os.path.join(args.output_dir, "best_acc_net.pt")))
        # if (val_loss < MIN_TEST_LOSS and epoch >= 100) and args.save_ckpt > 0:
        #         MIN_TEST_LOSS = val_loss
        #         BEST_NET = model.state_dict()
        #         torch.save(BEST_NET, os.path.join(args.output_dir, "best_loss_net.pt"))
        #         log('EP{} Store model (best TEST LOSS) to {}'.format(epoch, os.path.join(args.output_dir, "best_loss_net.pt")))
        # if (val_ece20 < MIN_TEST_ECE and epoch >= 100) and args.save_ckpt > 0:
        #         MIN_TEST_ECE = val_ece20
        #         BEST_NET = model.state_dict()
        #         torch.save(BEST_NET, os.path.join(args.output_dir, "best_ece_net.pt"))
        #         log('EP{} Store model (best TEST ECE) to {}'.format(epoch, os.path.join(args.output_dir, "best_ece_net.pt")))
        if (args.save_ckpt > 0) and ((epoch+1) % args.save_ckpt ==0 or epoch == 0 or epoch == args.max_epochs-1):
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'ep{}.pt'.format(epoch)))


def set_seed(SEED=666):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='neural collapse')
    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument('--dset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--ETF_fc', action='store_true', default=False)

    # aug
    parser.add_argument('--aug', type=str, default='null')
    # not needed
    parser.add_argument('--min_scale', type=float, default=0.2)  # scale for MoCo Aug
    
    # ece type ece|adaece
    parser.add_argument('--ece_type', type=str, default='ece')
    parser.add_argument('--ece_flag', action='store_true', default=False)

    # dataset parameters of CIFAR10
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--norm', type=str, default='bn', help='Type of norm layer')  # bn|gn

    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--no-bias', dest='bias', default=True, action='store_false')

    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--scheduler', type=str, default='ms')  # step|ms/multi_step/cosine
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=600)

    parser.add_argument('--wd', type=float, default=5e-4)  # '54'|'01_54' | '01_54_54'
    parser.add_argument('--koleo_wt', type=float, default=0.0)
    parser.add_argument('--kl_cls_wt', type=str, default='n')  # d|c  default|center
    parser.add_argument('--koleo_type', type=str, default='d')  # d|c  default|center
    parser.add_argument('--kl_beta', type=float, default=0.9)  # d|c  default|center
    parser.add_argument('--loss', type=str, default='ce')  # ce|ls|ceh|hinge
    parser.add_argument('--eps', type=float, default=0.05)  # for ls loss
    parser.add_argument('--margin', type=float, default=1.0)  # for hinge loss

    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--save_ckpt', type=int, default=-1)
    parser.add_argument('--log_freq', type=int, default=2)

    args = parser.parse_args()
    args.output_dir = os.path.join('/scratch/lg154/sseg/neural_collapse/result/{}/{}/'.format(args.dset, args.model), args.exp_name)

    if args.dset == 'cifar100':
        args.num_classes=100
    elif args.dset == 'tinyi':
        args.num_classes=200
    elif args.dset == 'cifar10':
        args.num_classes = 10

    set_seed(SEED=args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    set_log_path(args.output_dir)
    log('save log to path {}'.format(args.output_dir))
    log(print_args(args))

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='nc_2025',
               name=args.exp_name
               )
    wandb.config.update(args)

    main(args)