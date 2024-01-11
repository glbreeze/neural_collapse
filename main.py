# -*- coding: utf-8 -*-
import os
import torch
import wandb
import random
import pickle
import argparse
from dataset.data import get_dataloader
from model import ResNet, MLP
from utils import Graph_Vars, set_optimizer, set_optimizer_b, set_optimizer_b1, set_log_path, log, print_args, get_scheduler
from utils import compute_ETF, compute_W_H_relation
from utils import CrossEntropyLabelSmooth, CrossEntropyHinge, KoLeoLoss, analysis, AverageMeter, accuracy

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


# analysis parameters
exam_epochs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
              230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430,
              440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640,
              650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850,
              860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000]


def plot_var(x_list, y_train, y_test, z=None, fname='fig.png', type='', title=None, zlabel=None, ylabel=None):
    plt.figure()
    plt.semilogy(x_list, y_train, label='{} of train'.format(type))
    plt.semilogy(x_list, y_test, label='{} of test'.format(type))
    if z is not None:
        plt.semilogy(x_list, z, label=zlabel)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel if ylabel is not None else type)
    plt.title(title if title is not None else type)
    plt.savefig(fname)


def train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, lr_scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    losses = AverageMeter('Loss', ':.4e')
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
                M = torch.zeros(len(feat), args.C).to(device)
                M[torch.arange(len(feat)), target] = 1            # [B, C]
                M = torch.nn.functional.normalize(M, p=1, dim=0)  # [B, C]
                cls_mean = torch.einsum('cb,bd->cd', M.T, feat)   # [C, B] * [B, D]
                cls_in_batch = torch.unique(target)
                cls_mean = cls_mean[cls_in_batch]
                glb_mean = torch.mean(cls_mean, dim=0)

                kl_loss = koleo_loss(feat-glb_mean.detach())

            else:
                kl_loss = koleo_loss(feat)
            loss += kl_loss * args.koleo_wt

        # ==== update loss and acc
        train_acc.update(torch.sum(out.argmax(dim=-1) == target).item() / target.size(0), target.size(0))
        losses.update(loss.item(), target.size(0))

        loss.backward()
        optimizer.step()

        if args.scheduler == 'cosine' and epoch <= args.decay_epochs:
            lr_scheduler.step()

    return train_acc.avg, losses.avg


def validate(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output, feat = model(input, ret_feat=True)

            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    return top1.avg, top5.avg


def main(args):

    MAX_TEST_ACC, BEST_EPOCH=0.0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==================== data loader ====================
    train_loader, test_loader = get_dataloader(args)

    # ====================  define model ====================
    if args.model.lower() == 'mlp':
        model = MLP(hidden = args.width, depth = args.depth, fc_bias=args.bias, num_classes=args.C)
    else:
        model = ResNet(pretrained=False, num_classes=args.C, backbone=args.model, args=args)
    model = model.to(device)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ls':
        criterion = CrossEntropyLabelSmooth(args.C, epsilon=args.eps)
    elif args.loss == 'ceh':
        criterion = CrossEntropyHinge(args.C, epsilon=0.05)
    elif args.loss == 'hinge':
        criterion = nn.MultiMarginLoss(p=1, margin=args.margin, reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss()
    criterion_summed = nn.CrossEntropyLoss(reduction='sum')

    if args.bwd == '1_1' or len(args.bwd) == 1:
        optimizer = set_optimizer(model, args, 0.9, log)
    elif len(args.bwd) == 3:
        optimizer = set_optimizer_b(model, args, 0.9, log)
    elif len(args.bwd) == 5:
        optimizer = set_optimizer_b1(model, args, 0.9, log)

    if args.ckpt not in ['', 'null', 'none']:
        optimizer = torch.optim.SGD(model.classifier.parameters(),
                                    weight_decay=args.cls_wd,
                                    lr=args.lr,
                                    momentum=0.9)
        for param in model.features.parameters():
            param.requires_grad = False

    lr_scheduler = get_scheduler(args, optimizer, n_batches=len(train_loader))

    graphs1 = Graph_Vars()  # for training nc
    # graphs2 = Graph_Vars()  # for testing nc

    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=20)                         # --------------------------wandb
    for epoch in range(1, args.max_epochs + 1):
        train_acc, train_loss = train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, lr_scheduler=lr_scheduler)
        val_acc1, val_acc5 = validate(model, test_loader)
        log('Train\tEpoch: {}, Train Loss: {:.5f}, Train Accuracy: {:.5f}, Val Acc: {:.5f}, LR: {:.6f}'.format(
        epoch, train_loss, train_acc, val_acc1, optimizer.param_groups[0]['lr']))

        wandb.log({'train/train_acc': train_acc,
                   'train/train_loss': train_loss, 
                   'val/acc1':val_acc1,
                   'val/acc5':val_acc5,
                   'lr': optimizer.param_groups[0]['lr']},
                  step=epoch)

        if args.scheduler in ['step', 'ms', 'multi_step', 'poly']:
            lr_scheduler.step()

        if epoch in exam_epochs:

            nc_train = analysis(model, criterion_summed, train_loader, args)
            nc_train['test_acc'] = val_acc1
            # nc_val   = analysis(model, criterion_summed, test_loader, args)
            graphs1.load_dt(nc_train, epoch=epoch, lr=optimizer.param_groups[0]['lr'])
            # graphs2.load_dt(nc_val,   epoch=epoch, lr=optimizer.param_groups[0]['lr'])

            wandb.log({'nc/loss': nc_train['loss'],
                       'nc/acc': nc_train['acc'],
                       'nc/nc1': nc_train['nc1'],
                       'nc/nc2h': nc_train['nc2_h'],
                       'nc/nc2w': nc_train['nc2_w'],
                       'nc/nc3_1': nc_train['nc3_1'],
                       'nc/nc3': nc_train['nc3'],
                       'nc/w_norm': nc_train['w_mnorm'],
                       'nc/h_norm': nc_train['h_mnorm'],
                       }, step=epoch + 1)

            log('>>>>EP{}, train loss:{:.4f}, acc:{:.4f}, NC1:{:.4f}, NC3_1:{:.4f}, NC3:{:.4f}, Test_Acc:{:.4f}'.format(
                epoch, graphs1.loss[-1], graphs1.acc[-1], graphs1.nc1[-1], graphs1.nc3_1[-1], graphs1.nc3[-1], graphs1.test_acc[-1]))

            if val_acc1 > MAX_TEST_ACC and epoch >= 100:
                MAX_TEST_ACC = val_acc1
                BEST_EPOCH = epoch
                BEST_NET = model.state_dict()
                torch.save(BEST_NET, os.path.join(args.output_dir, "best_net.pt"))

    BEST_IDX = exam_epochs.index(BEST_EPOCH)
    log('>>>>EP{}, Best Test Acc:{}, Train NC1:{:.4f}, NC2h:{:.4f}, NC2w:{:.4f}, NC3_1:{:.4f}, NC3:{:.4f}'.format(
        BEST_EPOCH, MAX_TEST_ACC, graphs1.nc1[BEST_IDX], graphs1.nc2_h[BEST_IDX], graphs1.nc2_w[BEST_IDX], graphs1.nc3_1[BEST_IDX], graphs1.nc3[BEST_IDX]
    ))

    fname = os.path.join(args.output_dir, 'graph1.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(graphs1, f)


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
    parser.add_argument('--test_ood', action='store_true', default=False)
    parser.add_argument('--min_scale', type=float, default=0.2)  # scale for MoCo Aug
    parser.add_argument('--ckpt', type=str, default='')   # exp_name: wd54_ls
    parser.add_argument('--load_fc', action='store_true', default=False)

    # dataset parameters of CIFAR10
    parser.add_argument('--im_size', type=int, default=32)
    parser.add_argument('--padded_im_size', type=int, default=36)
    parser.add_argument('--C', type=int, default=10)
    parser.add_argument('--norm', type=str, default='bn', help='Type of norm layer')  # bn|gn

    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--no-bias', dest='bias', default=True, action='store_false')

    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--end_lr', type=float, default=0.00001)  # poly LRD
    parser.add_argument('--power', type=float, default=2.0)       # poly LRD
    parser.add_argument('--decay_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--scheduler', type=str, default='ms')  # step|ms/multi_step/cosine

    parser.add_argument('--wd', type=str, default='54')  # '54'|'01_54' | '01_54_54'
    parser.add_argument('--bwd', type=str, default='1_1')
    parser.add_argument('--koleo_wt', type=float, default=0.0)
    parser.add_argument('--koleo_type', type=str, default='d')  # d|c  default|center
    parser.add_argument('--loss', type=str, default='ce')  # ce|ls|ceh|hinge
    parser.add_argument('--eps', type=float, default=0.05)  # for ls loss
    parser.add_argument('--margin', type=float, default=1.0)  # for ls loss

    parser.add_argument('--exp_name', type=str, default='baseline')

    args = parser.parse_args()
    args.output_dir = os.path.join('/scratch/lg154/sseg/neural_collapse/result/{}/{}/'.format(args.dset, args.model), args.exp_name)
    if args.scheduler == 'ms':
        args.scheduler = 'multi_step'
    wds = args.wd.split('_')
    if len(wds) == 1:
        args.conv_wd, args.bn_wd, args.cls_wd = [float(wd[0]) / 10 ** int(wd[1]) for wd in wds] * 3
    elif len(wds) == 2:
        args.conv_wd, args.cls_wd = [float(wd[0]) / 10 ** int(wd[1]) for wd in wds]
        args.bn_wd = args.conv_wd
    elif len(wds) == 3:
        args.conv_wd, args.bn_wd, args.cls_wd = [float(wd[0]) / 10 ** int(wd[1]) for wd in wds]
    if args.dset == 'cifar100':
        args.C=100
    elif args.dset == 'tinyi':
        args.C=200
    if args.ckpt not in ['', 'none', 'null']:
        args.ckpt = os.path.join('result', args.dset, args.model, args.ckpt, 'best_net.pt')

    set_seed(SEED = args.seed)

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
    wandb.init(project="NC3_" + str(args.dset),
               name=args.exp_name.split('/')[-1]
               )
    wandb.config.update(args)
    main(wandb.config)