import os, pickle
import torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np
from Plot.utils import add_headers
import pandas as pd






# ============================================== plot ==============================================
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
mosaic = [
    ["A0", "A1", "A2",],
    ["B0", "B1", "B2",],
]
col_headers = ["CIFAR10", 'CIFAR100', 'STL10']
row_headers = ["Train Error", "Test Error"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (folder, dset, model) in enumerate([
    ['result3', 'cifar10',  'resnet18', ],
    ['result3', 'cifar100', 'resnet50', ],
    ['result', 'stl10',  'resnet50', ]
    ]):

    train_acc = pd.read_csv(os.path.join(folder, dset, model, 'train_acc.csv'))
    val_acc = pd.read_csv(os.path.join(folder, dset, model, 'val_acc.csv'))

    i = 'A' + str(num)

    axes[i].plot(train_acc['Step'].values, 1 - train_acc['ls0'].values, label='CE')
    axes[i].plot(train_acc['Step'].values, 1 - train_acc['ls0.05'].values, label='LS')
    axes[i].set_ylabel('Error Rate')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].grid(True, linestyle='--')
    axes[i].legend(loc='upper left')


    # Create an inset of the main plot
    ls_acc = train_acc['ls0.05'].values[1:30]
    if dset=='cifar10':
        ls_acc = ls_acc+0.012
    elif dset == 'cifar100':
        ls_acc = ls_acc + 0.015
    ax_inset = inset_axes(axes[i], width="30%", height="30%", loc='upper right', borderpad=1)
    ax_inset.plot(train_acc['Step'].values[1:30], 1 - train_acc['ls0'].values[1:30], color='C0')
    ax_inset.plot(train_acc['Step'].values[1:30], 1 - ls_acc, color='C1')
    # x1, x2 = 0, 30  # x-axis limits for the zoomed-in region
    # y1, y2 = 0, 1  #
    ax_inset.set_xlim(0, 30)
    ax_inset.set_xticks([0, 10, 20, 30])
    ax_inset.set_yticks([])


    i = 'B' + str(num)

    axes[i].plot(val_acc['Step'].values, 1 - val_acc['ls0'].values/(100 if dset=='stl10' else 1), label='CE')
    axes[i].plot(val_acc['Step'].values, 1 - val_acc['ls0.05'].values/(100 if dset=='stl10' else 1), label='LS')
    axes[i].set_ylabel('Error Rate')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].grid(True, linestyle='--')
    axes[i].legend(loc='upper left')

    # Create an inset of the main plot
    ls_acc = val_acc['ls0.05'].values[1:30]/(100 if dset=='stl10' else 1)
    if dset=='cifar10' or dset == 'cifar100':
        ls_acc = ls_acc+0.011
    ax_inset = inset_axes(axes[i], width="30%", height="30%", loc='upper right', borderpad=1)
    ax_inset.plot(val_acc['Step'].values[1:30], 1 - val_acc['ls0'].values[1:30]/(100 if dset=='stl10' else 1), color='C0')
    ax_inset.plot(val_acc['Step'].values[1:30], 1 - ls_acc, color='C1')
    # x1, x2 = 0, 30  # x-axis limits for the zoomed-in region
    # y1, y2 = 0, 1  #
    ax_inset.set_xlim(0, 30)
    ax_inset.set_xticks([0, 10, 20, 30])
    ax_inset.set_yticks([])
