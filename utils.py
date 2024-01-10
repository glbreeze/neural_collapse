import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
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
        self.test_acc = []
        self.w_mnorm = []
        self.h_mnorm = []

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
    Sw  = np.float32(Sw)
    Sb  = np.float32(Sb)
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

    # =========== NC2
    nc2_h = compute_ETF(M_.T, device)
    nc2_w = compute_ETF(W.T, device)

    # =========== NC3  ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, 'fro')
    normalized_W = W  / torch.norm(W, 'fro')
    W_M_dist = (torch.norm(normalized_W - normalized_M) ** 2).item()

    # =========== NC3 (all losses are equal paper)
    nc3_1 = compute_W_H_relation(W.T, M_, device)

    normalized_M = M_ / torch.norm(M_, dim=0)  # [512, C]/[C]
    normalized_W = W  / torch.norm(W,  dim=0)  # [512, C]/[C]
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
        "w_mnorm": np.mean(W_norms.cpu().numpy()),
        "h_mnorm": np.mean(M_norms.cpu().numpy()),
    }


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
