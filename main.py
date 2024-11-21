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
from utils import CrossEntropyLabelSmooth, CrossEntropyHinge, KoLeoLoss

import numpy as np
import torch.nn as nn
from metrics import ECELoss
from torch.nn import functional as F


def train_one_epoch(model, criterion, train_loader, optimizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_loss = AverageMeter('Loss', ':.4e')
    train_acc = AverageMeter('Train_acc', ':.4e')
    koleo_loss = KoLeoLoss()
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != args.batch_size:
            continue

        data, target = data.to(device), target.to(device)
        out, feat = model(data, ret_feat=True)

        optimizer.zero_grad()
        loss = criterion(out, target)  # all
        if args.koleo_wt > 0:
            # compute global mean
            if args.koleo_type == 'c':
                M = torch.zeros(len(feat), args.num_classes).to(device)
                M[torch.arange(len(feat)), target] = 1            # [B, C]
                M = torch.nn.functional.normalize(M, p=1, dim=0)  # [B, C]
                cls_mean = torch.einsum('cb,bd->cd', M.T, feat)   # [C, B] * [B, D]
                cls_in_batch = torch.unique(target)
                cls_mean = cls_mean[cls_in_batch]
                glb_mean = torch.mean(cls_mean, dim=0)

                kl_loss = koleo_loss(feat-glb_mean.detach())
            elif args.koleo_type == 'm': 
                global GLB_mean
                GLB_mean = args.kl_beta * GLB_mean + (1 - args.kl_beta) * torch.mean(feat.detach(), dim=0)
                kl_loss = koleo_loss(feat-GLB_mean.detach())
            else:
                kl_loss = koleo_loss(feat)
            loss += kl_loss * args.koleo_wt
        train_loss.update(loss.item(), target.size(0))

        loss.backward()
        optimizer.step()

        train_acc.update(torch.sum(out.argmax(dim=-1) == target).item() / target.size(0),
                         target.size(0)
                         )
    return train_loss, train_acc


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
    global GLB_mean
    GLB_mean = torch.zeros(model.feat_dim).to(device)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ls':
        criterion = CrossEntropyLabelSmooth(args.num_classes, epsilon=args.eps)
    elif args.loss == 'ceh':
        criterion = CrossEntropyHinge(args.num_classes, epsilon=0.05)
    elif args.loss == 'hinge':
        criterion = nn.MultiMarginLoss(p=1, margin=args.margin, reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss()
    ece_criterion15 = ECELoss(n_bins=15).cuda()
    ece_criterion20 = ECELoss(n_bins=20).cuda()
    ece_criterion25 = ECELoss(n_bins=25).cuda()

    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = get_scheduler(args, optimizer)

    # ====================  start training ====================
    wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(args.max_epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, train_loader, optimizer, args)
        lr_scheduler.step()
            
        # ================= check ECE
        if (epoch + 1) % args.log_freq == 0 or epoch == 0:
            logits, labels, feats = get_logits_labels_feats(test_loader, model)   # on cuda
            val_loss = F.cross_entropy(logits, labels, reduction='mean').item()   # on cuda 
            val_acc = (logits.argmax(dim=-1) == labels).sum().item()/len(labels)  # on cuda
            val_ece15 = ece_criterion15(logits, labels).item()  # on cuda
            val_ece20 = ece_criterion20(logits, labels).item()  # on cuda
            val_ece25 = ece_criterion25(logits, labels).item()  # on cuda

            # post process ece
            scaled_model = ModelWithTemperature(model)
            val_ece20_post, _ = scaled_model.set_temperature(test_loader, cross_validate='ece', n_bins=20)

            wandb.log({
                'overall/lr': optimizer.param_groups[0]['lr'],
                'overall/train_loss': train_loss.avg,
                'overall/train_acc': train_acc.avg,
                'overall/val_loss': val_loss,
                'overall/val_acc': val_acc,
                'ece/val_ece15': val_ece15,
                'ece/val_ece20': val_ece20,
                'ece/val_ece25': val_ece25,
                'ece/val_ece20_post': val_ece20_post,
                'ece/best_temp': ModelWithTemperature.temperature
                },
                step=epoch)

            # ================= check NCs
            nc_val = analysis_feat(labels, feats, args, W=model.classifier.weight.detach())

            logits, labels, feats = get_logits_labels_feats(train_loader, model)  # on cuda
            nc_train = analysis_feat(labels, feats, args, W=model.classifier.weight.detach())

            wandb.log({
                'train_nc/nc1': nc_train['nc1'],       'train_nc/nc2': nc_train['nc2'],
                'train_nc/nc3': nc_train['nc3'],       'train_nc/nc2h': nc_train['nc2h'],
                'train_nc/w_norm': nc_train['w_norm'], 'train_nc/h_norm': nc_train['h_norm'],

                'val_nc/nc1': nc_val['nc1'], 'val_nc/nc2': nc_val['nc2'],
                'val_nc/nc3': nc_val['nc3'], 'val_nc/nc2h': nc_val['nc2h'],
                'val_nc/nc2w': nc_val['nc2w'],
            }, step=epoch)

            # try:
            #     nc_train_all.load_dt(nc_train, epoch=epoch)
            #     nc_val_all.load_dt(nc_val, epoch=epoch)
            # except:
            #     nc_train_all = Graph_Vars(nc_train)
            #     nc_val_all   = Graph_Vars(nc_val)
            #     nc_train_all.load_dt(nc_train, epoch=epoch)
            #     nc_val_all.load_dt(nc_val, epoch=epoch)
        
        # ================= store the model
        if (val_acc > MAX_TEST_ACC and epoch >= 100) and args.save_ckpt > 0:
                MAX_TEST_ACC = val_acc
                BEST_NET = model.state_dict()
                torch.save(BEST_NET, os.path.join(args.output_dir, "best_acc_net.pt"))
                log('EP{} Store model (best TEST ACC) to {}'.format(epoch, os.path.join(args.output_dir, "best_acc_net.pt")))
        if (val_loss < MIN_TEST_LOSS and epoch >= 100) and args.save_ckpt > 0:
                MIN_TEST_LOSS = val_loss
                BEST_NET = model.state_dict()
                torch.save(BEST_NET, os.path.join(args.output_dir, "best_loss_net.pt"))
                log('EP{} Store model (best TEST LOSS) to {}'.format(epoch, os.path.join(args.output_dir, "best_loss_net.pt")))
        if (val_ece20 < MIN_TEST_ECE and epoch >= 100) and args.save_ckpt > 0:
                MIN_TEST_ECE = val_ece20
                BEST_NET = model.state_dict()
                torch.save(BEST_NET, os.path.join(args.output_dir, "best_ece_net.pt"))
                log('EP{} Store model (best TEST ECE) to {}'.format(epoch, os.path.join(args.output_dir, "best_ece_net.pt")))
        if (args.save_ckpt > 0) and ((epoch+1) % args.save_ckpt ==0 or epoch == 0):
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
    wandb.init(project='nc_ece',
               name=args.exp_name
               )
    wandb.config.update(args)

    main(args)