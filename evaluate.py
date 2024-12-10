# -*- coding: utf-8 -*-
import pdb
import os
import torch
import random
import pickle
import argparse
from dataset.data import get_dataloader
from model import ResNet, MLP
from utils import CrossEntropyLabelSmooth, CrossEntropyHinge, KoLeoLoss

import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

# Import dataloaders
from metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss

# Import temperature scaling and NLL utilities
from temperature_scaling import ModelWithTemperature

# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
    'stl10': 10
}


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dset', type=str, default='cifar10')
    parser.add_argument('--aug', type=str, default='null')
    parser.add_argument("--model", type=str, default='resnet18', help='name of the model')
    parser.add_argument('--ETF_fc', action='store_true', default=False)
    parser.add_argument('--norm', type=str, default='bn', help='Type of norm layer')  # bn|gn
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument("--save_path", type=str, default='result3/', help='Path to import the model')
    parser.add_argument("--exp_name", type=str, default='null', help="exp_name of the model")
    parser.add_argument("--ckpt", type=str, default='null', help="file name of the pre-trained model")
    parser.add_argument('--load_fc', action='store_false', default=True)
    parser.add_argument("--num_bins", type=int, default=20, dest="num_bins", help='Number of bins')

    parser.add_argument("-batch_size", type=int, default=128, help="Batch size")
    parser.add_argument('--test_ood', action='store_true', default=False)
    parser.add_argument('--min_scale', type=float, default=0.2)  # scale for MoCo Aug

    parser.add_argument("--cv_error", type=str, default='ece', help='Error function to do temp scaling')  # ece|nll
    parser.add_argument("--ece_type", default='ece', type=str)
    parser.add_argument("--bins", default=20, type=int)
    parser.add_argument("-log", action="store_true", dest="log", help="whether to print log data")

    return parser.parse_args()


def get_logits_labels(data_loader, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
    return logits, labels


if __name__ == "__main__":

    # Setting additional parameters
    out = {}
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parseArgs()
    args.num_classes = dataset_num_classes[args.dset]
    args.save_path = os.path.join(args.save_path, args.dset, args.model, args.exp_name)

    # ==================== data loader ====================
    train_loader, test_loader = get_dataloader(args)

    # ====================  define model ====================
    if args.model.lower() == 'mlp':
        model = MLP(hidden=args.width, depth=args.depth, fc_bias=args.bias, num_classes=args.num_classes)
    else:
        model = ResNet(pretrained=False, num_classes=args.num_classes, backbone=args.model, args=args)
    model = model.to(device)
    cudnn.benchmark = True
    model.load_state_dict(torch.load(os.path.join(args.save_path, '{}.pt'.format(args.ckpt)), map_location=device))

    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss(n_bins=args.bins).cuda()
    adaece_criterion = AdaptiveECELoss(n_bins=args.bins).cuda()
    cece_criterion = ClasswiseECELoss(n_bins=args.bins).cuda()

    logits, labels = get_logits_labels(test_loader, model)
    test_acc = (logits.argmax(dim=-1) == labels).sum().item() / len(labels)  # on cuda
    out['before_tune'] = {'logits': logits.cpu().numpy(), 'labels':labels.cpu().numpy()}

    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()

    print('CKPT:{}, Test_acc:{:.4f}, nll:{:.4f}, ece:{:.4f}, adaece:{:.4f}, cece:{:.4f}'.format(
        args.ckpt, test_acc, p_nll, p_ece, p_adaece, p_cece
    ))

    # ====================  Scaling with Temperature ====================
    # pdb.set_trace()
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(test_loader, cross_validate='ece', ece_type=args.ece_type, n_bins=args.bins)
    T_opt = scaled_model.get_temperature()
    logits, labels = get_logits_labels(test_loader, scaled_model)
    test_acc = (logits.argmax(dim=-1) == labels).sum().item() / len(labels)  # on cuda
    
    out['after_tune'] = {'logits': logits.cpu().numpy(), 'labels':labels.cpu().numpy()}
    file_path = os.path.join(args.save_path, '{}.pickle'.format(args.ckpt))
    with open(file_path, 'wb') as f:
            pickle.dump(out, f)
    print(f'----write output to {file_path}')

    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()

    print('--After Tuning, ckpt:{}, Test_acc:{:.4f}, nll:{:.4f}, ece:{:.4f}, adaece:{:.4f}, cece:{:.4f}'.format(
        args.ckpt, test_acc, nll, ece, adaece, cece
    ))
    print('Optimal temperature: ' + str(T_opt))
    print('Test accuracy: {:.4f}'.format(test_acc))
    print('Test NLL: ' + str(nll))
    print('ECE: ' + str(ece))
    print('AdaECE: ' + str(adaece))
    print('Classwise ECE: ' + str(cece))
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Test_acc", "pre_nll", "pre_ece", "post_nll", "post_ece", "bestT"))
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        round(test_acc, 4), round(p_nll, 4), round(p_ece, 4), round(nll, 4), round(ece, 4), round(T_opt, 4)))

