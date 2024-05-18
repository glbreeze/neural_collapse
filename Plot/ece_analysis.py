

import os
import pandas as pd
import numpy as np
from Plot.utils import add_headers
from matplotlib import pyplot as plt


mosaic = [
    ["A0", "A1", ],
    ["B0", "B1", ]
]
row_headers = ["CIFAR10", "CIFAR100"]
col_headers = None # ["Test ECE", "Optimal T", "Test ECE After Temperature Scaling"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

# ==== Training Acc
for row in ['A', 'B']:
    if row == 'A':
        path = 'result3/cifar10'
        df = pd.read_excel(os.path.join(path, 'ece_analysis_cifar10.xlsx'))
    elif row == 'B':
        path = 'result3/cifar100'
        df = pd.read_excel(os.path.join(path, 'ece_analysis_cifar100.xlsx'))

    i = row + '0'
    axes[i].plot(df['eps'], df['p_ece'], marker='o', markersize=3, label='Pre ECE' )
    axes[i].plot(df['eps'], df['ece'], marker='o', markersize=3, label='Post ECE', color='C1')
    axes[i].set_xlabel('$\delta$')
    if row == 'A':
        axes[i].legend()
    axes[i].grid(True, linestyle='--')
    axes[i].set_ylabel('ECE')

    i = row + '1'
    line1, = axes[i].plot(df['eps'], df['h_norm'], marker='o', markersize=3, label='h-norm', color='C0')
    line2, = axes[i].plot(df['eps'], df['w_norm'], marker='o', markersize=3, label='w-norm', color='C0', linestyle='--' )
    axes[i].set_xlabel('$\delta$')

    if row == 'B':
        axes[i].set_ylim(0, 9)
    axes[i].grid(True, linestyle='--')
    axes[i].set_ylabel('w-norm/h-norm', color='C0')
    axes[i].tick_params(axis='y', colors='C0')

    ax2 = axes[i].twinx()
    line3, = ax2.plot(df['eps'], df['best_t'], marker='s', markersize=3, label='Best T', color='C1')
    ax2.set_ylabel('Optimal T', color='C1')
    ax2.tick_params(axis='y', colors='C1')
    ax2.axhline(y=1, color='C1', linestyle=':')

    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    if row=='A':
        axes[i].legend(lines, labels)


plt.show()











mosaic = [
    ["A0", "A1", "A2"],
    ["B0", "B1", "B2"]
]
row_headers = ["CIFAR10", "CIFAR100"]
col_headers = None # ["Test ECE", "Optimal T", "Test ECE After Temperature Scaling"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

# ==== Training Acc
for row in ['A', 'B']:
    if row == 'A':
        path = 'result3/cifar10'
        df = pd.read_excel(os.path.join(path, 'ece_analysis_cifar10.xlsx'))
    elif row == 'B':
        path = 'result3/cifar100'
        df = pd.read_excel(os.path.join(path, 'ece_analysis_cifar100.xlsx'))

    i = row + '0'
    axes[i].plot(df['eps'], df['p_ece'], marker='o', markersize=3, label='Pre ECE' )
    axes[i].plot(df['eps'], df['ece'], marker='o', markersize=3, label='Post ECE', color='C1')
    axes[i].set_xlabel('$\delta$')
    axes[i].legend()

    i = row + '1'
    axes[i].plot(df['eps'], df['w_norm'], marker='o', markersize=3, label='w-norm', )
    axes[i].plot(df['eps'], df['h_norm'], marker='o',  markersize=3, label='h-norm', )
    axes[i].plot(df['eps'], df['best_t'], marker='s', markersize=3, label='Best T',  linestyle='--')
    axes[i].set_xlabel('$\delta$')
    axes[i].legend()
    if row == 'B':
        axes[i].set_ylim(0, 9)

    i = row + '2'
    axes[i].plot(df['eps'], df['ece'], marker='o', markersize=3, label='Post ECE', color='C0')
    axes[i].plot(df['eps'], 1-df['test_acc'], marker='+', markersize=3, label='Test Error', color='C1')
    axes[i].set_ylim(0, 0.2)
    axes[i].set_xlabel('$\delta$')

    ax2 = axes[i].twinx()
    ax2.plot(df['eps'], df['train_nc1'],  label='train_nc1', marker='s', markersize=3, color='C2')
    ax2.set_yscale('log')

    lines1, labels1 = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[i].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.show()







