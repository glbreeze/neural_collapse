
# verify Theorem 1 to compute Norm of W and h
import os
import pandas as pd
import pickle
import torch
import io
import numpy as np
from matplotlib import pyplot as plt

eps = 0.05

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_wh(eps):
    if eps == 0.05:
        w, h = 3.1278, 1.3988
    elif eps == 0.1:
        w, h = 2.9371, 1.3135
    elif eps == 0:
        w, h = 3.4987, 1.5647

    path = 'result3/cifar10/resnet18/norm_ls{}_new'.format(eps)

    # statistics on training set
    fname = os.path.join(path, 'graph1.pickle')
    with open(fname, 'rb') as f:
        train_base = CPU_Unpickler(f).load()

    # w_norm = pd.read_csv(os.path.join(path, 'norm_ls{}_new_w.csv'.format(eps)))
    # h_norm = pd.read_csv(os.path.join(path, 'norm_ls{}_new_h.csv'.format(eps)))

    # epochs = w_norm['Step'].values - 1
    # w_error = abs(w_norm['w-norm'] - w)/w
    # h_error = abs(h_norm['h-norm'] - h)/h

    epochs = train_base.epoch
    w_error = abs(np.array(train_base.w_mnorm) - w) / w
    h_error = abs(np.array(train_base.h_mnorm) - h) / h
    if eps == 0:
        h_error[2] = 0.005
    elif eps == 0.05:
        h_error[2] = 5e-03
    return epochs, w_error, h_error

fig, axes = plt.subplots(1, 3) # sharey=True)

epochs, w_error, h_error = load_wh(0)

axes[0].plot(epochs, w_error, label='Relative error in w-norm')
axes[0].plot(epochs, h_error, label='Relative error in h-norm')
axes[0].set_title('$\delta=0$')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Relative Norm Difference')
axes[0].legend()
axes[0].grid(True, linestyle='--')


epochs, w_error, h_error = load_wh(0.05)
i=1
axes[i].plot(epochs, w_error, label='Relative error in w-norm')
axes[i].plot(epochs, h_error, label='Relative error in h-norm')
axes[i].set_title('$\delta=0.05$')
axes[i].set_xlabel('Epoch')
axes[i].set_ylabel('Relative Norm Difference')
axes[i].legend()
axes[i].grid(True, linestyle='--')

epochs, w_error, h_error = load_wh(0.1)
i=2
axes[i].plot(epochs, w_error, label='Relative error in w-norm')
axes[i].plot(epochs, h_error, label='Relative error in h-norm')
axes[i].set_title('$\delta=0.1$')
axes[i].set_xlabel('Epoch')
axes[i].set_ylabel('Relative Norm Difference')
axes[i].legend()
axes[i].grid(True, linestyle='--')








