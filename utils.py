import os
import torch.optim as optim


def set_optimizer(model, feature_decay, classifier_decay, lr, momentum):
    # Group parameters of feature extractor and classifier
    params_to_optimize = [
        {"params": model.features.parameters(), "weight_decay": feature_decay},
        {"params": model.classifier.parameters(), "weight_decay": classifier_decay},
    ]
    optimizer = optim.SGD(params_to_optimize, lr=lr, momentum=momentum)
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

