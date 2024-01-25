# -*- coding: utf-8 -*-
import pdb
import os
import pickle
import torch
import random
import pickle
import argparse
from dataset.data import get_dataloader
from model import ResNet, MLP
from utils import Graph_Vars, set_optimizer, set_optimizer_b, set_optimizer_b1, set_log_path, log, print_args
from utils import compute_ETF, get_logits_labels, get_logits_labels_feats, analysis_nc1, analysis_nc1_new

import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

# Import temperature scaling and NLL utilities
from temperature_scaling import ModelWithTemperature
from metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss

# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200
}

def parseArgs():
    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dset', type=str, default='cifar10')
    parser.add_argument("--model", type=str, default='resnet18', help='name of the model')
    parser.add_argument('--ETF_fc', action='store_true', default=False)
    parser.add_argument('--norm', type=str, default='bn', help='Type of norm layer')  # bn|gn
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument("--save_path", type=str, default='result/', help='Path to import the model')
    parser.add_argument("--exp_name", type=str, default='null', help="exp_name of the model")
    parser.add_argument("--ckpt", type=str, default='null', help="file name of the pre-trained model")
    parser.add_argument('--load_fc', action='store_false', default=True)
    parser.add_argument("--num_bins", type=int, default=20, dest="num_bins", help='Number of bins')

    parser.add_argument("-batch_size", type=int, default=128, help="Batch size")
    parser.add_argument('--test_ood', action='store_true', default=False)
    parser.add_argument('--min_scale', type=float, default=0.2)  # scale for MoCo Aug

    parser.add_argument("--cv_error", type=str, default='ece', help='Error function to do temp scaling')  # ece|nll
    parser.add_argument("-log", action="store_true", dest="log", help="whether to print log data")

    return parser.parse_args()


class Graph_Dt:
    def __init__(self):
        self.epoch = []
        self.acc = []
        self.loss = []
        self.loss_inc = []
        self.loss_cor = []

        self.nc1 = []
        self.nc1_cor = []
        self.nc1_inc = []

        self.ece_pre  = []
        self.ece_post = []
        self.adece_pre  = []
        self.adece_post = []
        self.opt_t = [] 
        
        self.ent_cor = []
        self.ent_inc = []

    def load_dt(self, nc_dt, epoch):
        self.epoch.append(epoch)
        for key in nc_dt:
            try:
                self.__getattribute__(key).append(nc_dt[key])
            except:
                print('{} is not attribute of Graph var'.format(key))


if __name__ == "__main__":

    out = {}
    # Setting additional parameters
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parseArgs()
    args.num_classes = dataset_num_classes[args.dset]
    args.save_path = os.path.join(args.save_path, args.dset, args.model, args.exp_name)
    ckpts = [fname for fname in os.listdir(args.save_path) if fname.startswith('ep') and fname.endswith('pt')]
    ckpts_indices = [int(fname.split('.')[0][2:]) for fname in ckpts]
    ckpts_indices.sort()

    # ==================== data loader ====================
    train_loader, test_loader = get_dataloader(args)
    plot_var = Graph_Dt()

    for ckpt_idx in ckpts_indices:
        out_sv = {}
        args.ckpt = os.path.join(args.save_path, 'ep{}.pt'.format(ckpt_idx))

        # ====================  define model ====================
        if args.model.lower() == 'mlp':
            model = MLP(hidden=args.width, depth=args.depth, fc_bias=args.bias, num_classes=args.num_classes)
        else:
            model = ResNet(pretrained=False, num_classes=args.num_classes, backbone=args.model, args=args)
        model = model.to(device)
        cudnn.benchmark = True
        model.load_state_dict(torch.load(args.ckpt))

        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss(n_bins=args.num_bins).cuda()
        adaece_criterion = AdaptiveECELoss(n_bins=args.num_bins).cuda()
        cece_criterion = ClasswiseECELoss(n_bins=args.num_bins).cuda()

        logits, labels, feats = get_logits_labels_feats(test_loader, model)
        test_acc = (logits.argmax(dim=-1) == labels).sum().item() / len(labels)  # on cuda
        out_sv['before_tune'] = {'logits': logits.cpu().numpy(), 'labels': labels.cpu().numpy()}

        p_ece = ece_criterion(logits, labels).item()
        p_adaece = adaece_criterion(logits, labels).item()
        p_cece = cece_criterion(logits, labels).item()
        p_nll = nll_criterion(logits, labels).item()
        pdb.set_trace()
        nc_dt = analysis_nc1_new(logits, labels, feats, num_classes=args.num_classes)
        
        nc_dt['ece_pre'] = p_ece
        nc_dt['adece_pre'] = p_adaece
        print('CKPT:{}__Pre, Test_acc:{:.4f}, nll:{:.4f}, ece:{:.4f}, adaece:{:.4f}, cece:{:.4f}'.format(ckpt_idx, test_acc, p_nll, p_ece, p_adaece, p_cece))

        # ====================  Scaling with Temperature ====================
        # pdb.set_trace()
        scaled_model = ModelWithTemperature(model)
        scaled_model.set_temperature(test_loader, cross_validate=args.cv_error, n_bins=args.num_bins)
        T_opt = scaled_model.get_temperature()
        logits, labels = get_logits_labels(test_loader, scaled_model)
        test_acc = (logits.argmax(dim=-1) == labels).sum().item() / len(labels)  # on cuda
        out_sv['after_tune'] = {'logits': logits.cpu().numpy(), 'labels': labels.cpu().numpy()}
        # with open(os.path.join(args.save_path, '{}.pickle'.format(ckpt_name)), 'wb') as f:
        #     pickle.dump(out, f)

        ece = ece_criterion(logits, labels).item()
        adaece = adaece_criterion(logits, labels).item()
        cece = cece_criterion(logits, labels).item()
        nll = nll_criterion(logits, labels).item()

        print('CKPT:{}__Post, Test_acc:{:.4f}, nll:{:.4f}, ece:{:.4f}, adaece:{:.4f}, cece:{:.4f}'.format(ckpt_idx, test_acc, nll, ece, adaece, cece))

        nc_dt['ece_post'] = ece
        nc_dt['adece_post'] = adaece
        nc_dt['opt_t'] = T_opt
        plot_var.load_dt(nc_dt, epoch=ckpt_idx)

    with open(os.path.join(args.save_path, 'evaluate_all_new_debug.pickle'), 'wb') as f:
        pickle.dump(plot_var, f)


