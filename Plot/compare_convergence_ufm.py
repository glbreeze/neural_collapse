

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Plot.utils import add_headers



# ========= colar mapping =========
# Normalize x values to [0, 1] for colormap mapping
cmap1 = plt.cm.viridis
cmap2 = plt.cm.plasma
norm = plt.Normalize(1, 6)

# ============================ Plot ============================
mosaic = [
    ["A0", "A1", "A2", "A3", ],
    ["B0", "B1", "B2", "B3", ]
]
row_headers = ["CIFAR10", "CIFAR100"]
col_headers = ["Train Error", "NC1", "NC2", "NC3"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for row in ['A','B']:
    if row == 'A':
        path = 'result3/cifar10/ufm'
    elif row == 'B':
        path = 'result3/cifar100/ufm'

    nc1 = pd.read_csv(os.path.join(path, 'train_nc1.csv'))
    nc2 = pd.read_csv(os.path.join(path, 'train_nc2.csv'))
    nc3 = pd.read_csv(os.path.join(path, 'train_nc3.csv'))
    acc = pd.read_csv(os.path.join(path, 'train_acc.csv'))
    test_acc = pd.read_csv(os.path.join(path, 'test_acc.csv'))

    i = row + '0'
    # ==== Training Acc
    axes[i].plot(acc['Step'].values, 1 - acc['ls0'].values, label='$\delta=0$', color=cmap1(norm(1)))
    axes[i].plot(acc['Step'].values, 1 - acc['ls0.05'].values, label='$\delta=0.05$', color=cmap1(norm(2)))
    axes[i].plot(acc['Step'].values, 1 - acc['ls0.1'].values, label='$\delta=0.1$', color=cmap1(norm(3)))
    axes[i].plot(acc['Step'].values, 1 - acc['ls0.2'].values, label='$\delta=0.2$', color=cmap1(norm(4)))
    axes[i].plot(acc['Step'].values, 1 - acc['ls0.3'].values, label='$\delta=0.3$', color=cmap1(norm(5)))
    axes[i].plot(acc['Step'].values, 1 - acc['ls0.4'].values, label='$\delta=0.4$', color=cmap1(norm(6)))
    # axes[i].plot(acc['Step'].values, 1 - acc['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    # axes[i].plot(np.arange(800), 1 - acc['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))
    axes[i].set_ylabel('Train Error Rate')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    axes[i].set_yscale("log")
    axes[i].set_xlim(0,800)

    i = row + '1'
    # ===== NC1
    axes[i].plot(nc1['Step'].values - 1, nc1['ls0'].values, label='$\delta=0$', color=cmap1(norm(1)))
    axes[i].plot(nc1['Step'].values - 1, nc1['ls0.05'].values, label='$\delta=0.05$', color=cmap1(norm(2)))
    axes[i].plot(nc1['Step'].values - 1, nc1['ls0.1'].values, label='$\delta=0.1$', color=cmap1(norm(3)))
    axes[i].plot(nc1['Step'].values - 1, nc1['ls0.2'].values, label='$\delta=0.2$', color=cmap1(norm(4)))
    axes[i].plot(nc1['Step'].values - 1, nc1['ls0.3'].values, label='$\delta=0.3$', color=cmap1(norm(5)))
    axes[i].plot(nc1['Step'].values - 1, nc1['ls0.4'].values, label='$\delta=0.4$', color=cmap1(norm(6)))
    # axes[i].plot(nc1['Step'].values - 1, nc1['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    #axes[i].plot(nc1['Step'].values - 1, nc1['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))
    axes[i].set_ylabel('NC1')
    axes[i].set_yscale("log")
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    axes[i].set_xlim(0,800)

    i = row + '2'
    # ===== NC2
    axes[i].plot(nc1['Step'].values - 1, nc2['ls0'].values, label='$\delta=0$', color=cmap1(norm(1)))
    axes[i].plot(nc1['Step'].values - 1, nc2['ls0.05'].values, label='$\delta=0.05$', color=cmap1(norm(2)))
    axes[i].plot(nc1['Step'].values - 1, nc2['ls0.1'].values, label='$\delta=0.1$', color=cmap1(norm(3)))
    axes[i].plot(nc1['Step'].values - 1, nc2['ls0.2'].values, label='$\delta=0.2$', color=cmap1(norm(4)))
    axes[i].plot(nc1['Step'].values - 1, nc2['ls0.3'].values, label='$\delta=0.3$', color=cmap1(norm(5)))
    axes[i].plot(nc1['Step'].values - 1, nc2['ls0.4'].values, label='$\delta=0.4$', color=cmap1(norm(6)))
    #axes[i].plot(nc1['Step'].values - 1, nc2['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    #axes[i].plot(nc1['Step'].values - 1, nc2['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))

    axes[i].set_ylabel('NC2')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    axes[i].set_xlim(0, 800)
    axes[i].set_xticks([0, 200, 400, 600, 800])

    # ===== NC3
    i = row+'3'
    axes[i].plot(nc1['Step'].values - 1, nc3['ls0'].values, label='$\delta=0$', color=cmap1(norm(1)))
    axes[i].plot(nc1['Step'].values - 1, nc3['ls0.05'].values, label='$\delta=0.05$', color=cmap1(norm(2)))
    axes[i].plot(nc1['Step'].values - 1, nc3['ls0.1'].values, label='$\delta=0.1$', color=cmap1(norm(3)))
    axes[i].plot(nc1['Step'].values - 1, nc3['ls0.2'].values, label='$\delta=0.2$', color=cmap1(norm(4)))
    axes[i].plot(nc1['Step'].values - 1, nc3['ls0.3'].values, label='$\delta=0.3$', color=cmap1(norm(5)))
    axes[i].plot(nc1['Step'].values - 1, nc3['ls0.4'].values, label='$\delta=0.4$', color=cmap1(norm(6)))
    #axes[i].plot(nc1['Step'].values - 1, nc3['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    #axes[i].plot(nc1['Step'].values - 1, nc3['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))
    axes[i].set_ylabel('NC3')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    axes[i].set_yscale("log")
    axes[i].set_xlim(0, 800)
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))



    # i = row + '4'
    # # ==== Testing Acc
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0'].values, label='$\delta=0$', color=cmap1(norm(1)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.05'].values, label='$\delta=0.05$', color=cmap1(norm(2)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.1'].values, label='$\delta=0.1$', color=cmap1(norm(3)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.2'].values, label='$\delta=0.2$', color=cmap1(norm(4)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.3'].values, label='$\delta=0.3$', color=cmap1(norm(5)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.4'].values, label='$\delta=0.4$', color=cmap1(norm(6)))
    # #axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    # #axes[i].plot(np.arange(800), 1 - test_acc['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))
    # axes[i].set_ylabel('Test Error Rate')
    # axes[i].set_xlabel('Epoch')
    # axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")

    # axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #fig.suptitle("Model Convergence with Different Smoothing Hyperparameter")


