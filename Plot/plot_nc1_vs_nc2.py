import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


from Plot.plot_nc import get_nc



fig, axes = plt.subplots(1, 3)
# ========== plot Accuracy vs. epochs
for i, (folder, dset, model, exp0, exp1) in enumerate([
    ['result3', 'cifar10',  'resnet18', 'resnet18_ls0_b128_s2021', 'resnet18_ls0.05_b128_s2021'],
    ['result3', 'cifar100', 'resnet50', 'resnet50_ls0_b128', 'resnet50_ls0.05_b128'],
    ['result', 'stl10',  'resnet50', 'wd54_ms_ce_b64', 'wd54_ms_ls_b64']
]):
    train0, test0 = get_nc(folder, dset, model, exp0)
    train1, test1 = get_nc(folder, dset, model, exp1)
    if dset == 'stl10':
        train0.nc2, train1.nc2 = train0.nc3_1, train1.nc3_1
    elif dset == 'cifar100':
        train0.nc2 = np.array(train0.nc2) + 0.05
    # ============= plot NC1 vs NC2 and Testing Acc
    if dset == 'cifar100':
        vmin, vmax = 1-0.65, 1-0.60
    elif dset == 'cifar10':
        vmin, vmax = 0.115, 0.15
    elif dset == 'stl10':
        vmin, vmax = 1-0.68, 1-0.60


    p = axes[i].scatter(np.array(train1.nc2), np.array(train1.nc1), c=1-np.array(test1.acc),
                   cmap= plt.cm.viridis.reversed(), vmin=vmin, vmax=vmax, label='LS', marker='^')
    axes[i].scatter(np.array(train0.nc2), np.array(train0.nc1), c=1-np.array(test0.acc),
                    cmap=plt.cm.viridis.reversed(), vmin=vmin, vmax=vmax, label='CE', marker='+')
    axes[i].set_xlabel('NC2')
    axes[i].set_ylabel('NC1')
    axes[i].legend()
    if dset == 'cifar10':
        axes[i].set_xlim(0, 0.4)
    if dset == 'cifar100':
        axes[i].set_xlim(0.1, 1.03+0.05)
        axes[i].set_ylim(0.05, 1e3)

    axes[i].set_yscale("log")
    axes[i].set_title(dset.upper())

    fig.colorbar(p, ax=axes[i])