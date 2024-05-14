

import os, io
import torch
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

folder = 'result'
dset = 'cifar10'
model = 'resnet18'
exp0, exp1 = 'wd54_ms_ce_b64', 'wd54_ms_ls_b64'


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


# statistics on training set
fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph1.pickle'.format(exp0))
with open(fname, 'rb') as f:
    train_base = CPU_Unpickler(f).load()

fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph1.pickle'.format(exp1))
with open(fname, 'rb') as f:
    train_new1 = CPU_Unpickler(f).load()

# statistics on test set
fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph2.pickle'.format(exp0))
with open(fname, 'rb') as f:
    test_base = CPU_Unpickler(f).load()

fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph2.pickle'.format(exp1))
with open(fname, 'rb') as f:
    test_new1 = CPU_Unpickler(f).load()

fig, axes = plt.subplots(1, 3)
epochs = train_base.epoch


# ========== plot NC1 vs. epochs
i = 0
axes[i].plot(epochs, train_base.nc1, label='CE-Train NC1', color='C0')
axes[i].plot(epochs, train_new1.nc1, label='LS-Train NC1', color='C1')
axes[i].plot(epochs, test_base.nc1, label='CE-Test NC1', color='C0', linestyle='--')
axes[i].plot(epochs, test_new1.nc1, label='LS-Test NC1', color='C1', linestyle='--')
axes[i].set_ylabel('NC1')
axes[i].set_xlabel('Epoch')
axes[i].set_yscale("log");
axes[i].legend()
axes[i].set_title('NC1 vs. Epochs')

# ========== plot NC2 vs. epochs
i = 1
axes[i].plot(epochs, train_base.nc3_1, label='CE-Train NC2', color='C0')
axes[i].plot(epochs, train_new1.nc3_1, label='LS-Train NC2', color='C1')
axes[i].plot(epochs, test_base.nc3_1, label='CE-Test NC2', color='C0', linestyle='--')
axes[i].plot(epochs, test_new1.nc3_1, label='LS-Test NC2', color='C1', linestyle='--')
axes[i].set_ylabel('NC2')
axes[i].set_xlabel('Epoch')
axes[i].legend()
axes[i].set_title('NC2 vs. Epochs')

# ========== plot NC3 vs. epochs
i = 2
axes[i].plot(epochs, train_base.nc3, label='CE-Train NC3', color='C0')
axes[i].plot(epochs, train_new1.nc3, label='LS-Train NC3', color='C1')
axes[i].plot(epochs, test_base.nc3, label='CE-Test NC3', color='C0', linestyle='--')
axes[i].plot(epochs, test_new1.nc3, label='LS-Test NC3', color='C1', linestyle='--')
axes[i].set_ylabel('NC3')
axes[i].set_xlabel('Epoch')
axes[i].legend()
axes[i].set_title('NC3 vs. Epochs')
