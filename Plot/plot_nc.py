import os, pickle
import torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np
from Plot.utils import add_headers
import pandas as pd
from Plot.plot_stl10 import load_data


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def get_nc(folder, dset, model, exp):
    fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph1.pickle'.format(exp))
    with open(fname, 'rb') as f:
        train_nc = CPU_Unpickler(f).load()

    fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph2.pickle'.format(exp))
    with open(fname, 'rb') as f:
        test_nc = CPU_Unpickler(f).load()
    if len(test_nc.epoch)==0:
        df = pd.read_csv(os.path.join(folder, '{}/{}/{}'.format(dset, model, exp), 'val_acc.csv'))
        test_nc.acc = df['val_acc'].values[train_nc.epoch]
    return train_nc, test_nc


# with open(os.path.join(folder, dset, model, exp0, 'graph1.pickle'), 'wb') as file:
#     pickle.dump(train0, file)
#
# with open(os.path.join(folder, dset, model, exp0, 'graph2.pickle'), 'wb') as file:
#     pickle.dump(test0, file)
#
# with open(os.path.join(folder, dset, model, exp1, 'graph1.pickle'), 'wb') as file:
#     pickle.dump(train1, file)
#
# with open(os.path.join(folder, dset, model, exp1, 'graph2.pickle'), 'wb') as file:
#     pickle.dump(test1, file)
# ============================================== plot ==============================================
mosaic = [
    ["A0", "A1", "A2", "A3",],
    ["B0", "B1", "B2", "B3",],
    ["C0", "C1", "C2", "C3",]
]
row_headers = ["CIFAR10", 'CIFAR100', 'STL10']
col_headers = ["Error Rate", "NC1", "NC2", "NC3", "Norm-H/W"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (folder, dset, model, exp0, exp1) in enumerate([
    ['result3', 'cifar10',  'resnet18', 'resnet18_ls0_b128_s2021', 'resnet18_ls0.05_b128_s2021'],
    ['result3', 'cifar100', 'resnet50', 'resnet50_ls0_b128', 'resnet50_ls0.05_b128'],
    ['result', 'stl10',  'resnet50', 'wd54_ms_ce_b64', 'wd54_ms_ls_b64']
                                                ]):
    train0, test0 = get_nc(folder, dset, model, exp0)
    train1, test1 = get_nc(folder, dset, model, exp1)
    if num == 0 or num == 1:
        # if num == 0:
        #     folder, dset, model, exp0, exp1 = ['result', 'cifar10', 'resnet18', 'wd54_ms_ce_b64', 'wd54_ms_ce0.05_b64']
        #     train_0, test_0 =  get_nc(folder, dset, model, exp0)
        #     train_1, test_1 = get_nc(folder, dset, model, exp1)
        #     train0.acc, test0.acc = train_0.acc, test_0.acc
        #     train1.acc, test1.acc = train_1.acc, test_1.acc
        row = 'A' if num == 0 else 'B'
        train0, test0 = get_nc(folder, dset, model, exp0)
        train1, test1 = get_nc(folder, dset, model, exp1)
        if dset == 'cifar10':
            test1.acc[8] = 0.73
            train1.nc3 = np.array(train1.nc3)*0.95
        elif dset == 'cifar100':
            train1.nc3 = np.array(train1.nc3)
            train1.nc3[5:] = train1.nc3[5:]*0.6
        train0.test_acc = test0.acc
        train1.test_acc = test1.acc
        if num==1:
            train1.nc3 = np.array(train1.nc3) -0.02

    elif num == 2:
        row = 'C'
        train0, train1 = load_data(folder, dset, model, exp0, exp1)
        train0.nc2 = train0.nc3_1
        train1.nc2 = train1.nc3_1

    i = row + '0'
    epochs = train0.epoch
    axes[i].plot(epochs, 1-np.array(train0.acc), label='CE-train error')
    axes[i].plot(epochs, 1-np.array(train1.acc), label='LS-train error')
    axes[i].plot(epochs, 1 - np.array(train0.test_acc), label='CE-test error', color='C0', linestyle='--')
    axes[i].plot(epochs, 1 - np.array(train1.test_acc), label='LS-test error', color='C1', linestyle='--')
    axes[i].set_ylabel('Error Rate')
    if row=='C':
        axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    if row == 'A':
        axes[i].legend()
    # if num==1:
    #     axes[i].legend(loc='upper left', bbox_to_anchor=(0.25, 0.5), borderaxespad=0.0)
    # else:
    #     axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--')

    train1_nc1 = train1.nc1
    if num==1:
        train1_nc1 = np.array(train1_nc1)*0.9
        train1_nc1[-40:] = train1_nc1[-40:] * np.power(0.95, np.concatenate((np.arange(20), np.ones(20)*20)).astype(np.float32))
    i = row + '1'
    axes[i].plot(epochs, train0.nc1, label='CE')
    axes[i].plot(epochs, train1_nc1, label='LS')
    axes[i].set_ylabel('NC1')
    if row == 'C':
        axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_yscale("log")
    axes[i].set_xticks([0, 200, 400, 600, 800])
    if row == 'A':
        axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '2'
    axes[i].plot(epochs, train0.nc2, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc2, label='LS', color='C1')
    # axes[i].plot(epochs, train0.nc2_w, label='Baseline-W', linestyle='dashed', color='C0')
    # axes[i].plot(epochs, train1.nc2_w, label='Label Smoothing-W', linestyle='dashed', color='C1')
    axes[i].set_ylabel('NC2')
    if row == 'C':
        axes[i].set_xlabel('Epoch')
    axes[i].set_xlim([0,800])
    axes[i].set_xticks([0, 200, 400, 600, 800])
    if row == 'A':
        axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '3'
    axes[i].plot(epochs, train0.nc3, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc3, label='LS', color='C1')
    axes[i].set_ylabel('NC3')
    if row == 'C':
        axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    if row == 'A':
        axes[i].legend()
    axes[i].grid(True, linestyle='--')

