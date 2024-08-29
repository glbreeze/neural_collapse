

import os
import pandas as pd
import numpy as np
from Plot.utils import add_headers
from matplotlib import pyplot as plt


mosaic = [
    ["A0", "A1", "A2"],
    ["B0", "B1", "B2"],
    ["C0", "C1", "C2"]
]
row_headers = ["CIFAR10", "CIFAR100", "STL10"]
col_headers = ["NC metrics vs $\delta$", "W/H norm vs $\delta$", "Test error vs $\delta$"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

# ==== Training Acc
for row in ['A', 'B', 'C']:
    if row == 'A':
        path = 'result3/cifar10/eps'
        df = pd.read_csv(os.path.join(path, 'eps.csv'))
    elif row == 'B':
        path = 'result3/cifar100/eps'
        df = pd.read_csv(os.path.join(path, 'eps.csv'))
    elif row == 'C':
        path = 'result3/stl10/eps'
        df = pd.read_csv(os.path.join(path, 'eps.csv'))

    ax= axes[row + '0']
    ax.plot(df['eps'], df['nc1'], marker='o', markersize=3, label='NC1', color='C0')
    ax.set_ylabel('NC1', color='C0')
    ax.set_yscale("log")
    ax.tick_params(axis='y', colors='C0')
    ax.set_xlabel('$\delta$')
    ax.grid(True, linestyle='--')


    ax2 = ax.twinx()
    ax2.plot(df['eps'], df['nc2'].values, marker='o', markersize=3, label='NC2', color='C1')
    ax2.plot(df['eps'], df['nc3'].values, marker='o', markersize=3, label='NC3', color='C1', linestyle='--')
    ax2.set_ylabel('NC2/NC3', color='C1')
    ax2.tick_params(axis='y', colors='C1')

    handles, labels = [], []
    for a in [ax, ax2]:
        for h, l in zip(*a.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    ax.legend(handles, labels)

    ax = axes[row + '1']
    ax.plot(df['eps'], df['h_norm'], marker='o', markersize=3, label='H-norm', color='C0')
    ax.plot(df['eps'], df['w_norm'], marker='o', markersize=3, label='W-norm', color='C1')
    ax.set_xlabel('$\delta$')
    ax.legend()
    ax.grid(True, linestyle='--')
    ax.set_ylabel('Norm of W/H')

    ax = axes[row + '2']
    ax.plot(df['eps'], 1-df['acc'], marker='o', markersize=3, label='Test Error', color='C0')
    ax.set_xlabel('$\delta$')
    ax.set_ylabel('Test Error')
    ax.grid(True, linestyle='--')



plt.show()








