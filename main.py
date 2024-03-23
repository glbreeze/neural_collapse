# -*- coding: utf-8 -*-
import os
import wandb
import torch
import random
import pickle
import argparse
from nc_metric import analysis
from model import ResNet, MLP
from dataset.data import get_dataloader
from utils import Graph_Vars, set_optimizer, set_log_path, log, print_args, get_scheduler, get_logits_labels, AverageMeter
from utils import CrossEntropyLabelSmooth, CrossEntropyHinge, KoLeoLoss

import numpy as np
import torch.nn as nn
from metrics import ECELoss
from torch.nn import functional as F
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


def train_one_epoch(model, criterion, train_loader, optimizer, epoch, args):
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
        train_loss.update(loss.item(), target.size(0))

        loss.backward()
        optimizer.step()

        train_acc.update(torch.sum(out.argmax(dim=-1) == target).item() / target.size(0),
                         target.size(0)
                         )
    return train_loss, train_acc



def main(args):
    MAX_TEST_ACC, MIN_TEST_LOSS, MIN_TEST_ECE =0.0, 100.0, 100.0
    EP_ACC, EP_LOSS, EP_ECE = 0, 0, 0
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

    optimizer = set_optimizer(model, args, 0.9, log)
    lr_scheduler = get_scheduler(args, optimizer)

    graphs1 = Graph_Vars()  # for training nc
    graphs2 = Graph_Vars()  # for testing nc

    # ====================  start training ====================
    wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(args.max_epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, train_loader, optimizer, epoch, args)
        lr_scheduler.step()
            
        ece_criterion = ECELoss(n_bins=20).cuda()
        logits, labels = get_logits_labels(test_loader, model)                # on cuda 
        val_loss = F.cross_entropy(logits, labels, reduction='mean').item()   # on cuda 
        val_acc = (logits.argmax(dim=-1) == labels).sum().item()/len(labels)  # on cuda 
        val_ece = ece_criterion(logits, labels).item()                        # on cuda 
        log('---->EP{}, test acc: {:.4f}, test loss: {:.4f}, test ece: {:.4f}'.format(
            epoch, val_acc, val_loss, val_ece
        ))

        wandb.log({
            'overall/lr': optimizer.param_groups[0]['lr'],
            'overall/train_loss': train_loss.avg,
            'overall/train_acc': train_acc.avg,
            'overall/val_loss': val_loss,
            'overall/val_acc': val_acc,
            'overall/val_ece': val_ece
            },
            step=epoch + 1)
        
        # ================= store the model
        if val_acc > MAX_TEST_ACC and epoch >= 100:
                MAX_TEST_ACC = val_acc
                EP_ACC = epoch
                BEST_NET = model.state_dict()
                torch.save(BEST_NET, os.path.join(args.output_dir, "best_acc_net.pt"))
                log('EP{} Store model (best TEST ACC) to {}'.format(epoch, os.path.join(args.output_dir, "best_acc_net.pt")))
        if val_loss < MIN_TEST_LOSS and epoch >= 100:
                MIN_TEST_LOSS = val_loss
                EP_LOSS = epoch
                BEST_NET = model.state_dict()
                torch.save(BEST_NET, os.path.join(args.output_dir, "best_loss_net.pt"))
                log('EP{} Store model (best TEST LOSS) to {}'.format(epoch, os.path.join(args.output_dir, "best_loss_net.pt")))
        if val_ece < MIN_TEST_ECE and epoch >= 100:
                MIN_TEST_ECE = val_ece
                EP_ECE = epoch
                BEST_NET = model.state_dict()
                torch.save(BEST_NET, os.path.join(args.output_dir, "best_ece_net.pt"))
                log('EP{} Store model (best TEST ECE) to {}'.format(epoch, os.path.join(args.output_dir, "best_ece_net.pt")))

        if (epoch % 10 ==0 or epoch == 5) and args.save_pt:
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'ep{}.pt'.format(epoch)))

        # ================= check NCs
        if epoch in exam_epochs:

            nc_train = analysis(model, train_loader, args)
            nc_val   = analysis(model, test_loader, args)
            graphs1.load_dt(nc_train, epoch=epoch, lr=optimizer.param_groups[0]['lr'])
            graphs2.load_dt(nc_val,   epoch=epoch, lr=optimizer.param_groups[0]['lr'])

            log('>>>>EP{}, train loss:{:.4f}, acc:{:.4f}, NC1:{:.4f}, NC2:{:.4f}, NC3:{:.4f}, w-norm:{:.4f}, h-norm:{:.4f}'.format(
                epoch, graphs1.loss[-1], graphs1.acc[-1], graphs1.nc1[-1], graphs1.nc2[-1], graphs1.nc3[-1], graphs1.w_mnorm[-1], graphs1.h_mnorm[-1]
            ))
            log('>>>>EP{}, test loss:{:.4f}, acc:{:.4f}, NC1:{:.4f}, NC2:{:.4f}, NC3:{:.4f}'.format(
                epoch, graphs2.loss[-1], graphs2.acc[-1], graphs2.nc1[-1], graphs2.nc2[-1], graphs2.nc3[-1]
                ))

            wandb.log({
                'train_nc/nc1': nc_train['nc1'],
                'train_nc/nc2': nc_train['nc2'],
                'train_nc/nc3': nc_train['nc3'],
                'train_nc/w-norm': nc_train['w_mnorm'],
                'train_nc/h-norm': nc_train['h_mnorm'],
                'val_nc/nc1': nc_val['nc1'],
                'val_nc/nc2': nc_val['nc2'],
                'val_nc/nc3': nc_val['nc3'],
                'val_nc/w-norm': nc_val['w_mnorm'],
                'val_nc/h-norm': nc_val['h_mnorm'],
            },
                step=epoch + 1)

    fname = os.path.join(args.output_dir, 'graph1.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(graphs1, f)
    fname = os.path.join(args.output_dir, 'graph2.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(graphs2, f)
    
    log('Finished Traning, Best TEST_ACC EP:{}/{:.4f}; Best TEST_LOSS EP:{}/{:.4f}; Best TEST_ECE EP:{}/{:.4f};'.format(
        EP_ACC, MAX_TEST_ACC, EP_LOSS, MIN_TEST_LOSS, EP_ECE, MIN_TEST_ECE))
    
    # plot loss
    plot_var(graphs1.epoch, graphs1.lr, graphs2.lr, type='Learning Rate',
                fname=os.path.join(args.output_dir, 'lr.png'))

    plot_var(graphs1.epoch, graphs1.loss, graphs2.loss, type='Loss',
                fname=os.path.join(args.output_dir, 'loss.png'))

    plot_var(graphs1.epoch,
                [100*(1-acc) for acc in graphs1.acc],
                [100*(1-acc) for acc in graphs2.acc],
                type='Error',
                fname=os.path.join(args.output_dir, 'error.png'))

    plot_var(graphs1.epoch, graphs1.nc1, graphs2.nc1, type='NC1',
                fname=os.path.join(args.output_dir, 'nc1.png'))

    plot_var(graphs1.epoch, graphs1.nc2_norm_h, graphs2.nc2_norm_h, z=graphs1.nc2_norm_w, type='NC2-1',
                fname=os.path.join(args.output_dir, 'nc2_1.png'), zlabel='NC2-1 of Classifier')

    plot_var(graphs1.epoch, graphs1.nc2_cos_h, graphs2.nc2_cos_h, z=graphs1.nc2_cos_w, type='NC2-2',
                fname=os.path.join(args.output_dir, 'nc2_2.png'), zlabel='NC2-2 of Classifier')

    plot_var(graphs1.epoch, graphs1.nc2_h, graphs2.nc2_h, z=graphs1.nc2_w, type='NC2',
                fname=os.path.join(args.output_dir, 'nc2.png'), zlabel='NC2 of Classifier')

    plot_var(graphs1.epoch, graphs1.nc3, graphs2.nc3, type='NC3', ylabel='||W - H||^2',
                fname=os.path.join(args.output_dir, 'nc3.png'))


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
    parser.add_argument('--lr_decay', type=float, default=0.5)  # for step
    parser.add_argument('--scheduler', type=str, default='ms')  # step|ms/multi_step/cosine
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=1000)

    parser.add_argument('--wd', type=str, default='54')  # '54'|'01_54' | '01_54_54'
    parser.add_argument('--koleo_wt', type=float, default=0.0)
    parser.add_argument('--koleo_type', type=str, default='d')  # d|c  default|center
    parser.add_argument('--loss', type=str, default='ce')  # ce|ls|ceh|hinge
    parser.add_argument('--eps', type=float, default=0.05)  # for ls loss
    parser.add_argument('--margin', type=float, default=1.0)  # for hinge loss

    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--save_pt', default=False, action='store_true')

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
    wandb.init(project='nc3' + args.dataset,
               name=args.exp_name
               )
    wandb.config.update(args)

    main(args)