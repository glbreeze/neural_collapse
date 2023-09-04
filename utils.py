import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        'multi_step': optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400], gamma=0.1),
        'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches * args.decay_epochs, eta_min=args.end_lr),
        'poly': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    }
    return SCHEDULERS[args.scheduler]


def compute_ETF(W, device):
    K = W.shape[0]
    W = W - torch.mean(W, dim=0, keepdim=True)
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, H, device):
    """ H is already normalized"""
    K = W.shape[0]

    W = W - torch.mean(W, dim=0, keepdim=True)
    WH = torch.mm(W, H.to(device))
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
        self.accuracy = []
        self.loss = []
        self.reg_loss = []

        # NC1
        self.Sw_invSb = []

        # NC2
        self.norm_M_CoV = []
        self.norm_W_CoV = []
        self.cos_M = []
        self.cos_W = []

        # NC3
        self.W_M_dist = []

        self.nc2 = []
        self.nc3 = []

        # NC4
        self.NCC_mismatch = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []
        self.lr = []


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

