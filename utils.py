import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse.linalg import svds


def _get_polynomial_decay(lr, end_lr, decay_epochs, from_epoch=0, power=1.0):
  # Note: epochs are zero indexed by pytorch
  end_epoch = float(from_epoch + decay_epochs)

  def lr_lambda(epoch):
    if epoch < from_epoch:
      return 1.0
    epoch = min(epoch, end_epoch)
    new_lr = ((lr - end_lr) * (1. - epoch / end_epoch) ** power + end_lr)
    return new_lr / lr  # LambdaLR expects returning a factor

  return lr_lambda


def get_scheduler(args, optimizer, n_batches):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """

    lr_lambda = _get_polynomial_decay(args.lr, args.end_lr,
                                      decay_epochs=args.decay_epochs,
                                      from_epoch=0, power=args.power)
    SCHEDULERS = {
        'step': optim.lr_scheduler.StepLR(optimizer, step_size=args.max_epochs//10, gamma=args.lr_decay),
        'multi_step': optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,350], gamma=0.1),
        'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches * args.decay_epochs, eta_min=args.end_lr),
        'poly': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    }
    return SCHEDULERS[args.scheduler]


def compute_ETF(W, device):  # W [K, 512]
    K = W.shape[0]
    # W = W - torch.mean(W, dim=0, keepdim=True)
    WWT = torch.mm(W, W.T)            # [K, 512] [512, K] -> [K, K]
    WWT /= torch.norm(WWT, p='fro')   # [K, K]

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, H, device):  # W:[K, 512] H:[512, K]
    """ H is already normalized"""
    K = W.shape[0]

    # W = W - torch.mean(W, dim=0, keepdim=True)
    WH = torch.mm(W, H.to(device))   # [K, 512] [512, K]
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:  Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.05, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)

        if torch.cuda.is_available(): targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class CrossEntropyHinge(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:  Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.05, reduction=True):
        super(CrossEntropyHinge, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)
        targets = targets.to(device)

        loss = (- targets * log_probs).sum(dim=1)

        mask = loss.detach() >= -torch.log(torch.tensor(1-self.epsilon))
        loss = loss * mask

        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, output, eps=1e-8):
        """
        Args:
            output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            output = F.normalize(output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(output)  # noqa: E741
            distances = self.pdist(output, output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss


def set_optimizer(model, args, momentum, log, conv_wd=None, bn_wd=None, cls_wd=None):
    conv_params, bn_params, cls_params = [], [], []

    for name, param in model.named_parameters():
        if 'conv' in name or "downsample.0" in name or "features.0" in name:
            conv_params.append(param)
        elif 'bn' in name or 'downsample.1' in name or "features.1" in name:
            bn_params.append(param)
        elif 'classifier' in name or 'fc' in name:
            cls_params.append(param)

    params_to_optimize = [
        {"params": conv_params, "weight_decay": conv_wd if conv_wd is not None else args.conv_wd},
        {"params": bn_params, "weight_decay": bn_wd if bn_wd is not None else args.bn_wd},
        {"params": cls_params, "weight_decay": cls_wd if cls_wd is not None else args.cls_wd},
    ]

    optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=momentum)
    log('>>>>>Set Optimizer conv_wd:{}, bn_wd:{}, cls_wd:{}'.format(
        conv_wd if conv_wd is not None else args.conv_wd,
        bn_wd if bn_wd is not None else args.bn_wd,
        cls_wd if cls_wd is not None else args.cls_wd))
    return optimizer


def set_optimizer_b(model, args, momentum, log,):
    conv_params, bn_params, bnb_params, cls_params, clsb_params = [], [], [], [], []

    for name, param in model.named_parameters():
        if 'conv' in name or "downsample.0" in name or "features.0" in name:
            conv_params.append(param)
        elif 'bn' in name or 'downsample.1' in name or "features.1" in name:
            if 'weight' in name:
                bn_params.append(param)
            else:
                bnb_params.append(param)
        elif 'classifier' in name or 'fc' in name:
            if 'weight' in name:
                cls_params.append(param)
            else:
                clsb_params.append(param)

    params_to_optimize = [
        {"params": conv_params, "weight_decay": args.conv_wd},
        {"params": bn_params, "weight_decay": args.bn_wd},
        {"params": bnb_params, "weight_decay": args.bn_wd * int(args.bwd.split('_')[0]) },
        {"params": cls_params, "weight_decay": args.cls_wd},
        {"params": clsb_params, "weight_decay": args.cls_wd * int(args.bwd.split('_')[1]) },
    ]

    optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=momentum)
    log('>>>>>Set Optimizer conv_wd:{}, bn_wd:{}, bnb_wd:{}, cls_wd:{}, clsb_wd:{}'.format(
        args.conv_wd,
        args.bn_wd, args.bn_wd * int(args.bwd.split('_')[0]),
        args.cls_wd, args.cls_wd * int(args.bwd.split('_')[1])
    ))
    return optimizer


def set_optimizer_b1(model, args, momentum, log,):
    conv_params, bn_params, bnb_params, bnb1_params, cls_params, clsb_params = [], [], [], [], [], []

    for name, param in model.named_parameters():
        if 'conv' in name or "downsample.0" in name or "features.0" in name:
            conv_params.append(param)
        elif 'bn' in name or 'downsample.1' in name or "features.1" in name:
            if 'weight' in name:
                bn_params.append(param)
            else:
                if ('7.2.bn3.bias' in name and args.model == 'resnet50') or \
                   ('7.1.bn2.bias' in name and args.model == 'resnet18'):
                    bnb1_params.append(param)
                    log('----{} assigned to BNB_last (bnb1)'.format(name))
                else:
                    bnb_params.append(param)
        elif 'classifier' in name or 'fc' in name:
            if 'weight' in name:
                cls_params.append(param)
            else:
                clsb_params.append(param)

    params_to_optimize = [
        {"params": conv_params, "weight_decay": args.conv_wd},
        {"params": bn_params,   "weight_decay": args.bn_wd},                                # W in BN layers
        {"params": bnb_params,  "weight_decay": args.bn_wd * int(args.bwd.split('_')[0])},  # Bias in BN layers except last BN layer
        {"params": bnb1_params, "weight_decay": args.bn_wd * int(args.bwd.split('_')[1])},  # Bias in last BN layer
        {"params": cls_params,  "weight_decay": args.cls_wd},
        {"params": clsb_params, "weight_decay": args.cls_wd * int(args.bwd.split('_')[2])},
    ]

    optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=momentum)
    log('>>>>>Set Optimizer conv_wd:{}, bn_wd:{}, bnb_wd:{}, bnb1_wd:{}, cls_wd:{}, clsb_wd:{}'.format(
        args.conv_wd,
        args.bn_wd, args.bn_wd * int(args.bwd.split('_')[0]), args.bn_wd * int(args.bwd.split('_')[1]),
        args.cls_wd, args.cls_wd * int(args.bwd.split('_')[2])
    ))
    return optimizer


class Graph_Vars:
    def __init__(self):
        self.epoch = []
        self.acc = []
        self.loss = []
        self.ncc_mismatch = []

        self.nc1 = []

        self.nc2_norm_h = []
        self.nc2_norm_w = []
        self.nc2_cos_h = []
        self.nc2_cos_w = []
        self.nc2_h = []
        self.nc2_w =[]

        self.norm_h = []
        self.norm_w = []

        self.nc3 = []
        self.nc3_1 = []
        self.nc3_2 = []

        self.lr = []

    def load_dt(self, nc_dt, epoch, lr=None):
        self.epoch.append(epoch)
        if lr:
            self.lr.append(lr)
        for key in nc_dt:
            try:
                self.__getattribute__(key).append(nc_dt[key])
            except:
                print('{} is not attribute of Graph var'.format(key))


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "==========================================\n"
    return s


def get_logits_labels_feats(data_loader, net):
    logits_list = []
    labels_list = []
    feats_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits, feats = net(data, ret_feat=True)
            logits_list.append(logits)
            labels_list.append(label)
            feats_list.append(feats)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        feats = torch.cat(feats_list, dim=0).cuda()  # [N, 512]
    return logits, labels, feats



def analysis_nc1(logits, targets, feats, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, N_inc, N_cor = [0] * num_classes, [0] * num_classes, [0] * num_classes
    mean = [0] * num_classes
    Sw_all, Sw_cor, Sw_inc = 0, 0, 0

    loss, n_correct, n_match = 0, 0, 0
    logits, targets, feats = logits.to(device), targets.to(device), feats.to(device)
    preds = logits.argmax(dim=-1)

    criterion = nn.CrossEntropyLoss().cuda()
    idxs_cor = (targets == preds).nonzero(as_tuple=True)[0]
    idxs_inc = (targets != preds).nonzero(as_tuple=True)[0]
    loss = criterion(logits, targets).item()
    loss_cor = criterion(logits[idxs_cor], targets[idxs_cor]).item()
    loss_inc = criterion(logits[idxs_inc], targets[idxs_inc]).item()
    acc = (logits.argmax(dim=-1) == targets).sum().item() / len(targets)

    for computation in ['Mean', 'Cov']:

        for c in range(num_classes):
            idxs = (targets == c).nonzero(as_tuple=True)[0]
            idxs_cor = (targets == c and targets == preds).nonzero(as_tuple=True)[0]
            idxs_inc = (targets == c and targets != preds).nonzero(as_tuple=True)[0]

            if computation == 'Mean':
                # update class means
                h_c = feats[idxs, :]  # [B, 512]
                mean[c] += torch.sum(h_c, dim=0)  #  CHW
                N[c] += h_c.shape[0]

            elif computation == 'Cov':
                # update within-class cov
                z = feats[idxs, :] - mean[c].unsqueeze(0)  # [B, 512]
                cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))   # [B 512 1] [B 1 512] -> [B, 512, 512]
                Sw_all += torch.sum(cov, dim=0)  # [512, 512]

                z = feats[idxs_cor, :] - mean[c].unsqueeze(0)  # [B, 512]
                cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))  # [B 512 1] [B 1 512] -> [B, 512, 512]
                Sw_cor += torch.sum(cov, dim=0)  # [512, 512]
                N_cor += z.shape[0]

                z = feats[idxs_inc, :] - mean[c].unsqueeze(0)  # [B, 512]
                cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))  # [B 512 1] [B 1 512] -> [B, 512, 512]
                Sw_inc += torch.sum(cov, dim=0)  # [512, 512]
                N_inc += z.shape[0]

        if computation == 'Mean':
            for c in range(num_classes):
                mean[c] /= N[c]
                M = torch.stack(mean).T
                muG = torch.mean(M, dim=1, keepdim=True)  # [512, C]
                # between-class covariance
                M_ = M - muG  # [512, C]
                Sb = torch.matmul(M_, M_.T) / num_classes
        elif computation == 'Cov':
            Sw_all /= sum(N)
            Sw_inc /= sum(N_inc)
            Sw_cor /= sum(N_cor)
            Sw_all = Sw_all.cpu().numpy()
            Sw_inc = Sw_inc.cpu().numpy()
            Sw_cor = Sw_cor.cpu().numpy()

            Sb = Sb.cpu().numpy()
            eigvec, eigval, _ = svds(Sb, k=num_classes - 1)
            inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
            nc1_all = np.trace(Sw_all @ inv_Sb)
            nc1_cor = np.trace(Sw_cor @ inv_Sb)
            nc1_inc = np.trace(Sw_inc @ inv_Sb)

    return {
        'loss': loss,
        'loss_inc': loss_inc,
        'loss_cor': loss_cor,
        'acc': acc,
        'nc1': nc1_all,
        'nc1_inc': nc1_inc,
        'nc1_cor': nc1_cor
    }
