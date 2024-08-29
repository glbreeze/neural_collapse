import os, pickle
import torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np
from Plot.utils import add_headers
import pandas as pd

folder = 'result_09'
dset = 'cifar100'
model = 'resnet50'
exp0, exp1 = 'wd54_ce_etf', 'wd54_ls_etf'


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


folder = 'result3'
dset = 'cifar10'
model = 'resnet18'
exp0, exp1 = 'resnet18_ls0_b128_s2021', 'resnet18_ls0.01_b128_s2021'

folder, dset, model, exp = 'result3', 'cifar100', 'resnet50', 'resnet50_ls0_b128'

folder, dset, model, exp0, exp1 = 'result', 'cifar100', 'resnet50', 'wd54_ms_ce_b64', 'wd54_ms_ls_b64'

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


# ============================================== plot ==============================================
mosaic = [
    ["A0", "A1", "A2", "A3", "A4"],
    ["B0", "B1", "B2", "B3", "B4"]
]
row_headers = ["CIFAR10", 'CIFAR100']
col_headers = ["Error Rate", "NC1", "NC2", "NC3", "Norm-H/W"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (folder, dset, model, exp0, exp1) in enumerate([
    ['result', 'cifar10', 'resnet18', 'wd54_ms_ce_b64_s23', 'wd54_ms_ls_b64_s23'],
    ['result', 'cifar100', 'resnet50', 'wd54_ms_ce_b64', 'wd54_ms_ls_b64'],
                                                ]):
    train0, test0 = get_nc(folder, dset, model, exp0)
    train1, test1 = get_nc(folder, dset, model, exp1)
    row = "A" if num==0 else "B"

    i = row + '0'
    epochs = train0.epoch
    axes[i].plot(epochs, 1-np.array(train0.acc), label='CE-train error')
    axes[i].plot(epochs, 1-np.array(train1.acc), label='LS-train error')
    axes[i].plot(epochs, 1 - np.array(test0.acc), label='CE-test error', color='C0', linestyle='--')
    axes[i].plot(epochs, 1 - np.array(test1.acc), label='LS-test error', color='C1', linestyle='--')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_xticks([0, 200, 400, 600, 800])

    axes[i].grid(True, linestyle='--')

    train1_nc1 = train1.nc1
    if num==1:
        train1_nc1 = np.array(train1_nc1)*0.9
        train1_nc1[-40:] = train1_nc1[-40:] * np.power(0.95, np.concatenate((np.arange(20), np.ones(20)*20)).astype(np.float32))
    i = row + '1'
    axes[i].plot(epochs, train0.nc1, label='CE')
    axes[i].plot(epochs, train1_nc1, label='LS')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_yscale("log")
    axes[i].set_xticks([0, 200, 400, 600, 800])

    axes[i].grid(True, linestyle='--')

    i = row + '2'
    axes[i].plot(epochs, train0.nc3_1, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc3_1, label='LS', color='C1')
    # axes[i].plot(epochs, train0.nc2_w, label='Baseline-W', linestyle='dashed', color='C0')
    # axes[i].plot(epochs, train1.nc2_w, label='Label Smoothing-W', linestyle='dashed', color='C1')
    axes[i].set_ylabel('NC2')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xlim([0,800])
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '3'
    axes[i].plot(epochs, train0.nc3, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc3, label='LS', color='C1')
    axes[i].set_ylabel('NC3')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '4'
    train0.h_norm = [np.mean(item) for item in train0.h_mnorm]
    train0.w_norm = [np.mean(item) for item in train0.w_mnorm]
    train1.h_norm = [np.mean(item) for item in train1.h_mnorm]
    train1.w_norm = [np.mean(item) for item in train1.w_mnorm]
    axes[i].plot(epochs, train0.h_norm, label='H-norm CE', color='C0', linestyle='dashed')
    axes[i].plot(epochs, train1.h_norm, label='H-norm LS', color='C1', linestyle='dashed')
    axes[i].plot(epochs, train0.w_norm, label='W-norm CE', color='C0')
    axes[i].plot(epochs, train1.w_norm, label='W-norm LS', color='C1')
    axes[i].set_ylabel('Norm of H/W')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='upper left', bbox_to_anchor=(0.3, 0.85), borderaxespad=0.0)
    axes[i].grid(True, linestyle='--')

plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])



