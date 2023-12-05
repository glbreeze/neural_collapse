# -*- coding: utf-8 -*-
import os
import torch
import random
import pickle
import argparse
from dataset.data import get_dataloader
from model import Detached_ResNet
from utils import Graph_Vars, set_optimizer, set_optimizer_b, set_optimizer_b1, set_log_path, log, print_args, get_scheduler
from utils import compute_ETF, compute_W_H_relation
from utils import CrossEntropyLabelSmooth, CrossEntropyHinge, KoLeoLoss

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds


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
        loss.backward()
        optimizer.step()

        accuracy = torch.mean((torch.argmax(out, dim=1) == target).float()).item()

        if args.scheduler == 'cosine' and epoch <= args.decay_epochs:
            lr_scheduler.step()

    log('Train\tEpoch: {} [{}/{}] Batch Loss: {:.6f} Batch Accuracy: {:.6f} LR: {:.6f}'.format(
        epoch,
        batch_idx,
        len(train_loader),
        loss.item(),
        accuracy,
        optimizer.param_groups[0]['lr']
    ))



def analysis(model, criterion_summed, loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    N    = [0 for _ in range(args.C)]
    mean = [0 for _ in range(args.C)]
    Sw   = 0

    loss = 0
    n_correct = 0
    n_match = 0

    for computation in ['Mean', 'Cov']:
        for batch_idx, (data, target) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output, h = model(data, ret_feat=True)  # [B, C], [B, 512]


            for c in range(args.C):
                idxs = (target == c).nonzero(as_tuple=True)[0]
                if len(idxs) == 0:  # If no class-c in this batch
                    continue

                h_c = h[idxs, :]  # [B, 512]

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0)  #  CHW
                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov
                    z = h_c - mean[c].unsqueeze(0)  # [B, 512]
                    cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))   # [B 512 1] [B 1 512] -> [B, 512, 512]
                    Sw += torch.sum(cov, dim=0)  # [512, 512]

            # during calculation of class cov, calculate loss
            if computation == 'Cov':
                loss += criterion_summed(output, target).item()

                # 1) network's accuracy
                net_pred = torch.argmax(output, dim=1)
                n_correct += sum(net_pred == target).item()

                # 2) agreement between prediction and nearest class center
                hm_dist = torch.cdist(h, M.T, p=2)  # [B, 512], [K, 512] -> [B, K]
                NCC_pred = torch.argmin(hm_dist, dim=-1)
                n_match += torch.sum(NCC_pred == net_pred).item()

        if computation == 'Mean':
            for c in range(args.C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
        elif computation == 'Cov':
            loss /= sum(N)
            Sw /= sum(N)
            acc = n_correct/sum(N)
            ncc_mismatch = 1 - n_match / sum(N)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True)  # [512, C]

    # between-class covariance
    M_ = M - muG  # [512, C]
    Sb = torch.matmul(M_, M_.T) / args.C

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.C - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    Sw_invSb = np.trace(Sw @ inv_Sb)

    # ========== NC2.1 and NC2.2
    W = model.classifier.weight.T  # [512, C]
    M_norms = torch.norm(M_, dim=0)  # [C]
    W_norms = torch.norm(W , dim=0)  # [C]

    norm_M_CoV = (torch.std(M_norms) / torch.mean(M_norms)).item()
    norm_W_CoV = (torch.std(W_norms) / torch.mean(W_norms)).item()

    # mutual coherence
    def coherence(V):
        G = V.T @ V  # [C, D] [D, C]
        G += torch.ones((args.C, args.C), device=device) / (args.C - 1)
        G -= torch.diag(torch.diag(G))  # [C, C]
        return torch.norm(G, 1).item() / (args.C * (args.C - 1))

    cos_M = coherence(M_ / M_norms)  # [D, C]
    cos_W = coherence(W  / W_norms)

    # =========== NC3  ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, 'fro')
    normalized_W = W  / torch.norm(W, 'fro')
    W_M_dist = (torch.norm(normalized_W - normalized_M) ** 2).item()

    # =========== NC2
    nc2_h = compute_ETF(M_.T, device)
    nc2_w = compute_ETF(W.T, device)

    # =========== NC3 (all losses are equal paper)
    nc3_1 = compute_W_H_relation(W.T, M_, device)

    normalized_M = M_  / torch.norm(M_,  dim=0)    # [512, C]/[C]
    normalized_W = W.T / torch.norm(W.T, dim=0)    # [512, C]/[C]
    l2_dist = torch.norm(normalized_M - normalized_W, dim=0)  # [C]
    nc3_2 = torch.mean(l2_dist)

    return {
        'loss': loss,
        'acc': acc,
        'ncc_mismatch': ncc_mismatch,
        'nc1': Sw_invSb,
        'nc2_norm_h': norm_M_CoV,
        'nc2_norm_w': norm_W_CoV,
        'nc2_cos_h': cos_M,
        'nc2_cos_w': cos_W,
        'nc2_h': nc2_h,
        'nc2_w': nc2_w,
        'nc3': W_M_dist,
        'nc3_1': nc3_1,
        'nc3_2': nc3_2,
    }


def main(args):
    MAX_TEST_ACC, BEST_EPOCH=0.0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==================== data loader ====================
    train_loader, test_loader = get_dataloader(args)

    # ====================  define model ====================
    model = Detached_ResNet(pretrained=False, num_classes=args.C, backbone=args.model, args=args)
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
    graphs2 = Graph_Vars()  # for testing nc

    for epoch in range(1, args.max_epochs + 1):
        train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, lr_scheduler=lr_scheduler)

        if args.scheduler in ['step', 'ms', 'multi_step', 'poly']:
            lr_scheduler.step()

        if epoch in exam_epochs:

            nc_train = analysis(model, criterion_summed, train_loader, args)
            nc_val   = analysis(model, criterion_summed, test_loader, args)
            graphs1.load_dt(nc_train, epoch=epoch, lr=optimizer.param_groups[0]['lr'])
            graphs2.load_dt(nc_val,   epoch=epoch, lr=optimizer.param_groups[0]['lr'])

            log('>>>>EP{}, train loss:{:.4f}, acc:{:.4f}, NC1:{:.4f}, NC2h:{:.4f}, NC2w:{:.4f}, NC3:{:.4f}-- '
                'NC2-1:{:.4f}, NC2-2:{:.4f}, NC2-1W:{:.4f}, NC2-2W:{:.4f}, NC3:{:.4f}, NC3_2:{:.4f}'.format(
                epoch, graphs1.loss[-1], graphs1.acc[-1], graphs1.nc1[-1], graphs1.nc2_h[-1], graphs1.nc2_w[-1], graphs1.nc3_1[-1],
                graphs1.nc2_norm_h[-1], graphs1.nc2_cos_h[-1], graphs1.nc2_norm_w[-1], graphs1.nc2_cos_w[-1], graphs1.nc3[-1], graphs1.nc3_2[-1]
            ))

            log('>>>>EP{}, test loss:{:.4f}, acc:{:.4f}, NC1:{:.4f}, NC2h:{:.4f}, NC2w:{:.4f}, NC3:{:.4f}-- '
                'NC2-1:{:.4f}, NC2-2:{:.4f}, NC3:{:.4f}'.format(
                epoch, graphs2.loss[-1], graphs2.acc[-1], graphs2.nc1[-1], graphs2.nc2_h[-1], graphs2.nc2_w[-1], graphs2.nc3_1[-1],
                graphs2.nc2_norm_h[-1], graphs2.nc2_cos_h[-1], graphs2.nc3[-1]
                ))

            if graphs2.acc[-1] > MAX_TEST_ACC and epoch >= 100:
                MAX_TEST_ACC = graphs2.acc[-1]
                BEST_EPOCH = epoch
                BEST_NET = model.state_dict()
                torch.save(BEST_NET, os.path.join(args.output_dir, "best_net.pt"))

            # plot loss
            if epoch in [args.max_epochs//2, args.max_epochs] == 0:
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

    BEST_IDX = exam_epochs.index(BEST_EPOCH)
    log('>>>>EP{}, Best Test Acc:{}, Train NC1:{:.4f}, NC2h:{:.4f}, NC2w:{:.4f}, NC3:{:.4f} -- '
        'NC2-1:{:.4f}, NC2-2:{:.4f}, NC2-1W:{:.4f}, NC2-2W:{:.4f}, NC3:{:.4f}'.format(
        BEST_EPOCH, MAX_TEST_ACC, graphs1.nc1[BEST_IDX], graphs1.nc2_h[BEST_IDX], graphs1.nc2_w[BEST_IDX], graphs1.nc3_1[BEST_IDX],
        graphs1.nc2_norm_h[BEST_IDX], graphs1.nc2_cos_h[BEST_IDX], graphs1.nc2_norm_w[BEST_IDX], graphs1.nc2_cos_w[BEST_IDX], graphs1.nc3[BEST_IDX]
    ))

    fname = os.path.join(args.output_dir, 'graph1.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(graphs1, f)
    fname = os.path.join(args.output_dir, 'graph2.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(graphs2, f)


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

    main(args)