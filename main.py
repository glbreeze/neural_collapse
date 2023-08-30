# -*- coding: utf-8 -*-
import os
import pdb
import torch
import random
import pickle
import argparse
from data import get_dataloader
from model import Detached_ResNet
from utils import Graph_Vars, set_optimizer, set_optimizer_b, set_optimizer_b1, set_log_path, log, print_args, KoLeoLoss
from utils import CrossEntropyLabelSmooth, CrossEntropyHinge

import numpy as np
import torch.nn as nn
import torch.optim as optim
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


def train_one_epoch(model, criterion, train_loader, optimizer, epoch, args):
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

        if batch_idx % (len(train_loader)//2) == 0:
            log('Train\tEpoch: {} [{}/{}] Batch Loss: {:.6f} Batch Accuracy: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                loss.item(),
                accuracy))


def analysis(graphs, model, criterion_summed, loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    N    = [0 for _ in range(args.C)]
    mean = [0 for _ in range(args.C)]
    Sw   = 0

    loss = 0
    net_correct = 0
    NCC_match_net = 0

    for computation in ['Mean', 'Cov']:
        # pbar = tqdm(total=len(loader), position=0, leave=True)
        for batch_idx, (data, target) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output, h = model(data, ret_feat=True)  # [B, C], [B, 512]

            # during calculation of class means, calculate loss
            if computation == 'Mean':
                loss += criterion_summed(output, target).item()

            for c in range(args.C):
                # features belonging to class c
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

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs, :], dim=1)
                    net_correct += sum(net_pred == target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i, :] - M.T, dim=1) for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                    NCC_match_net += sum(NCC_pred == net_pred).item()

        if computation == 'Mean':
            for c in range(args.C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)

    graphs.loss.append(loss)
    graphs.accuracy.append(net_correct / sum(N))
    graphs.NCC_mismatch.append(1 - NCC_match_net / sum(N))

    # loss with weight decay
    weight_decay_loss = 0.0
    for name, param in model.classifier.named_parameters():
        if 'fc' in name:
            weight_decay_loss += torch.sum(param ** 2).item()
    reg_loss = loss + 0.5 * args.cls_wd * weight_decay_loss
    graphs.reg_loss.append(reg_loss)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True)  # [512, C]

    # between-class covariance
    M_ = M - muG  # [512, C]
    Sb = torch.matmul(M_, M_.T) / args.C

    # avg norm
    W = model.classifier.weight  #[C, 512]
    M_norms = torch.norm(M_, dim=0)  # [C]
    W_norms = torch.norm(W.T, dim=0) # [C]

    graphs.norm_M_CoV.append((torch.std(M_norms) / torch.mean(M_norms)).item())
    graphs.norm_W_CoV.append((torch.std(W_norms) / torch.mean(W_norms)).item())

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.C - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, 'fro')
    normalized_W = W.T / torch.norm(W.T, 'fro')
    graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M) ** 2).item())

    # mutual coherence
    def coherence(V):
        G = V.T @ V  # [C, D] [D, C]
        G += torch.ones((args.C, args.C), device=device) / (args.C - 1)
        G -= torch.diag(torch.diag(G))  # [C, C]
        return torch.norm(G, 1).item() / (args.C * (args.C - 1))

    graphs.cos_M.append(coherence(M_ / M_norms)) # [D, C]
    graphs.cos_W.append(coherence(W.T / W_norms))


def main(args):
    MAX_TEST_ACC, BEST_EPOCH=0.0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==================== data loader ====================
    train_loader, test_loader = get_dataloader(args)

    # ====================  define model ====================
    model = Detached_ResNet(pretrained=False, num_classes=args.C, backbone=args.model)
    model = model.to(device)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ls':
        criterion = CrossEntropyLabelSmooth(args.C, epsilon=0.05)
    elif args.loss == 'ceh':
        criterion = CrossEntropyHinge(args.C, epsilon=0.05)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion_summed = nn.CrossEntropyLoss(reduction='sum')

    if args.bwd == '1_1' or len(args.bwd) == 1:
        optimizer = set_optimizer(model, args, 0.9, log)
    elif len(args.bwd) == 3:
        optimizer = set_optimizer_b(model, args, 0.9, log)
    elif len(args.bwd) == 5:
        optimizer = set_optimizer_b1(model, args, 0.9, log)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.max_epochs//10, gamma=args.lr_decay)

    graphs1 = Graph_Vars()
    graphs2 = Graph_Vars()

    epoch_list = []
    for epoch in range(1, args.max_epochs + 1):

        train_one_epoch(model, criterion, train_loader, optimizer, epoch, args)
        lr_scheduler.step()

        if epoch in exam_epochs:

            epoch_list.append(epoch)
            analysis(graphs1, model, criterion_summed, train_loader, args)
            analysis(graphs2, model, criterion_summed, test_loader, args)

            log('>>>> epoch {}, train loss:{:.4f}, acc:{:.4f}, NC1:{:.4f}, NC2-1:{:.4f}, NC2-2:{:.4f}, NC2-1W:{:.4f}, NC2-2W:{:.4f}, NC3:{:.4f}'.format(
                epoch, graphs1.loss[-1], graphs1.accuracy[-1], graphs1.Sw_invSb[-1],
                graphs1.norm_M_CoV[-1], graphs1.cos_M[-1], graphs1.norm_W_CoV[-1], graphs1.cos_W[-1], graphs1.W_M_dist[-1]
            ))

            log('>>>> epoch {}, test loss:{:.4f}, acc:{:.4f}, NC1:{:.4f}, NC2-1:{:.4f}, NC2-2:{:.4f}, NC3:{:.4f}'.format(
                epoch, graphs2.loss[-1], graphs2.accuracy[-1], graphs2.Sw_invSb[-1],
                graphs2.norm_M_CoV[-1], graphs2.cos_M[-1], graphs2.W_M_dist[-1]
                ))

            if graphs2.accuracy[-1] > MAX_TEST_ACC:
                MAX_TEST_ACC = graphs2.accuracy[-1]
                BEST_EPOCH = epoch

            # plot loss
            if epoch % 50 == 0:
                plot_var(epoch_list, graphs1.loss, graphs2.loss, type='Loss',
                         fname=os.path.join(args.output_dir, 'loss.png'))

                # plot Error
                plot_var(epoch_list,
                         [100*(1-acc) for acc in graphs1.accuracy],
                         [100*(1-acc) for acc in graphs2.accuracy],
                         type='Error',
                         fname=os.path.join(args.output_dir, 'error.png'))

                plot_var(epoch_list, graphs1.Sw_invSb, graphs2.Sw_invSb, type='NC1',
                         fname=os.path.join(args.output_dir, 'nc1.png'))

                plot_var(epoch_list, graphs1.norm_M_CoV, graphs2.norm_M_CoV, z=graphs1.norm_W_CoV, type='NC2-1',
                         fname=os.path.join(args.output_dir, 'nc2_1.png'), zlabel='NC2-1 of Classifier')

                plot_var(epoch_list, graphs1.cos_M, graphs2.cos_M, z=graphs1.cos_W, type='NC2-2',
                         fname=os.path.join(args.output_dir, 'nc2_2.png'), zlabel='NC2-2 of Classifier')

                plot_var(epoch_list, graphs1.W_M_dist, graphs2.W_M_dist, type='NC3', ylabel='||W^T - H||^2',
                         fname=os.path.join(args.output_dir, 'nc3.png'))

    BEST_IDX = exam_epochs.index(BEST_EPOCH)
    log('>>>> Epoch:{}, Best Test Acc:{}, Train NC1:{}, NC2-1:{}, NC2-2:{}, NC2-1W:{}, NC2-2W:{}, NC3:{}'.format(
        BEST_EPOCH, MAX_TEST_ACC, graphs1.Sw_invSb, graphs1.norm_M_CoV, graphs1.cos_M,
        graphs1.norm_W_CoV, graphs1.cos_W, graphs1.W_M_dist
    ))

    fname = os.path.join(args.output_dir, 'graph1.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(graphs1, f)
    fname = os.path.join(args.output_dir, 'graph2.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(graphs2, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='neural collapse')
    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument('--dset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet18')

    # dataset parameters of CIFAR10
    parser.add_argument('--im_size', type=int, default=32)
    parser.add_argument('--padded_im_size', type=int, default=36)
    parser.add_argument('--C', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--wd', type=str, default='54')  # '54'|'01_54' | '01_54_54'
    parser.add_argument('--bwd', type=str, default='1_1')
    parser.add_argument('--koleo_wt', type=float, default=0.0)
    parser.add_argument('--koleo_type', type=str, default='d')  # d|c  default|center
    parser.add_argument('--loss', type=str, default='ce')  # ce|ls|ceh

    parser.add_argument('--exp_name', type=str, default='baseline')

    args = parser.parse_args()
    args.output_dir = os.path.join('/scratch/lg154/sseg/neural_collapse/result/{}/{}/'.format(args.dset, args.model), args.exp_name)
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
    
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    set_log_path(args.output_dir)
    log('save log to path {}'.format(args.output_dir))
    log(print_args(args))

    main(args)