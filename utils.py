import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



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
                if (name == 'layer4.2.bn3.bias' and args.model == 'resnet50') or \
                   (name == 'layer4.1.bn2.bias' and args.model == 'resnet18'):
                    bnb1_params.append(param)
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

        # NC4
        self.NCC_mismatch = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []


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

