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


def get_scheduler(args, optimizer):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    if args.scheduler in ['ms', 'multi_step']:
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)
    elif args.scheduler in ['cos', 'cosine']:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)


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
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, reduction=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction: return loss.mean()
        else: return loss.sum()


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

    def __init__(self, type='d'):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        self.type=type

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

    def _forward_base(self, output, eps=1e-8, labels=None, cls_weight=None):
        """
        Args:
            output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            output = F.normalize(output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(output)  # noqa: E741
            distances = self.pdist(output, output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps)
            if cls_weight is not None: 
                loss = loss * cls_weight[labels]
        return loss.mean()
    
    def _forward_dist(self, output, cls_mean, labels, eps=1e-8, cls_weight=None):
        normalized_feat = F.normalize(output, p=2, eps=1e-8, dim=-1)  # output is already recentered by global mean
        normalized_cls_mean = F.normalize(cls_mean, p=2, eps=1e-8, dim=-1)

        if self.type == 'cs':
            cosine_similarities = torch.matmul(normalized_feat, normalized_cls_mean.T)
            class_cosine_similarities = cosine_similarities.gather(1, labels.unsqueeze(1))
            distances = 1 - class_cosine_similarities.squeeze()
            loss = -torch.log(distances + eps)
            if cls_weight is not None: 
                loss = loss * cls_weight[labels]
            return loss.mean()
        
        elif self.type == 'l1':
            distances = torch.abs(normalized_feat - normalized_cls_mean[labels]).sum(dim=1)
            loss = - distances
            if cls_weight is not None: 
                loss = loss * cls_weight[labels]
            return loss.mean()
        
        elif self.type == 'l2':
            distances = torch.norm(normalized_feat - normalized_cls_mean[labels], p=2, dim=1)
            loss = 1 - distances
            if cls_weight is not None: 
                loss = loss * cls_weight[labels]
            return loss.mean()
        
        elif self.type == 'd2':
            distances = torch.norm(normalized_feat - normalized_cls_mean[labels], p=2, dim=1)
            loss = 1 - (distances**2)
            if cls_weight is not None: 
                loss = loss * cls_weight[labels]
            return loss.mean()

    def forward(self, output, cls_mean=None, labels=None, eps=1e-8, cls_weight=None): 
        if self.type in ['d', 'm', 'c']: 
            loss = self._forward_base(output, eps=eps, labels=labels, cls_weight=cls_weight)
        elif self.type in ['l1', 'l2', 'cs', 'd2']: 
            loss = self._forward_dist(output, cls_mean=cls_mean, labels=labels, eps=eps, cls_weight=cls_weight)
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


class Graph_Vars:
    def __init__(self, dt):
        self.epoch = []
        for key in dt.keys():
            if key not in self.__dict__:
                self.__setattr__(key, [])

    def load_dt(self, nc_dt, epoch):
        self.epoch.append(epoch)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits_list = []
    labels_list = []
    feats_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            logits, feats = net(data, ret_feat=True)
            logits_list.append(logits)
            labels_list.append(label)
            feats_list.append(feats)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        feats = torch.cat(feats_list, dim=0) # [N, 512]
    return logits, labels, feats


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels


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

