

import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict


def reliability_plot(confs, preds, labels, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, bns, align='edge', width=0.05, color='pink', label='Expected')
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    plt.show()


def bin_strength_plot(confs, preds, labels, num_bins=15):
    '''
    Method to draw a plot for the number of samples in each confidence bin.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][COUNT] / float(num_samples)) * 100
        y.append(n)
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Percentage samples')
    plt.ylabel('Percentage of samples')
    plt.xlabel('Confidence')
    plt.show()

import os
import copy
import torch
import pickle
import numpy as np
import torch.nn.functional as F
num_bins=20
dset, model ='cifar10', 'resnet18'
exp0, exp1 = 'wd54_ms_ce0.05_b64', 'wd54_ms_ls0.05_b64'
ckpt0, ckpt1 = 'best_acc_net', 'best_acc_net'

with open(os.path.join('result/{}/{}/{}'.format(dset, model, exp0), '{}.pickle'.format(ckpt0)), 'rb') as f:
    out0 = pickle.load(f)

with open(os.path.join('result/{}/{}/{}'.format(dset, model, exp1), '{}.pickle'.format(ckpt1)), 'rb') as f:
    out1 = pickle.load(f)

def get_bn_dict(out):
    softmaxes = F.softmax(torch.tensor(out['before_tune']['logits']), dim=1)
    con0, pred0 = torch.max(softmaxes, 1)
    softmaxes = F.softmax(torch.tensor(out['after_tune']['logits']), dim=1)
    con0a, pred0a = torch.max(softmaxes, 1)
    labels0 = out['before_tune']['labels']
    labels1 = out['before_tune']['labels']
    bin_dict0  = _populate_bins(con0,  pred0,  labels0, num_bins=20)
    bin_dict0a = _populate_bins(con0a, pred0a, labels1, num_bins=20)
    return bin_dict0, bin_dict0a

bin_dict0, bin_dict0a = get_bn_dict(out0)
bin_dict1, bin_dict1a = get_bn_dict(out1)
bns = [(i / float(num_bins)) for i in range(num_bins)]


# ====== plot acc
def get_values(bin_dict, name):
    num_samples = sum([bin['count'] for k, bin in bin_dict0.items()])
    y = []
    if name in ['acc']:
        for i in range(num_bins):
            if bin_dict[i]['count'] >= 2:
                n = (bin_dict[i][name] / bin_dict[i]['count']) * 100
            else:
                n=0
            y.append(n)
    elif name =='count':
        for i in range(num_bins):
            n = (bin_dict[i]['count'] / float(num_samples)) * 100
            y.append(n)
    return y

y0 = get_values(bin_dict0, 'acc')
y1 = get_values(bin_dict1, 'acc')
y0a = get_values(bin_dict0a, 'acc')
y1a = get_values(bin_dict1a, 'acc')

# =============== reliablity plot Before and After ===============
plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(1, 2)  # width:20, height:3

bar_width = 0.45
# Create positions for the bars
positions1 = np.arange(len(bns))
positions2 = positions1 + bar_width
positions0 = (positions1+positions2)/2
custom_ticks = list(positions0)+[20]

# Create side-by-side bar plots
axes[0].bar(positions0, np.array(bns), width=bar_width*2, label='Expected', color='pink', alpha=0.5)
axes[0].bar(positions1, np.array(y0)/100, width=bar_width, label='CE', color='blue', alpha=0.5)
axes[0].bar(positions2, np.array(y1)/100, width=bar_width, label='LS', color='green', alpha=0.5)
axes[0].legend()
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Confidence')
axes[0].set_xticks(custom_ticks[::4], (bns+[1.0])[::4])
axes[0].set_title('Pre Temperature Scaling')

axes[1].bar(positions0, np.array(bns), width=bar_width*2, label='Expected', color='pink', alpha=0.5)
axes[1].bar(positions1, np.array(y0a)/100, width=bar_width, label='CE', color='blue', alpha=0.5)
axes[1].bar(positions2, np.array(y1a)/100, width=bar_width, label='LS', color='green', alpha=0.5)
axes[1].legend()
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Confidence')
axes[0].set_xticks(custom_ticks[::4], (bns+[1.0])[::4])
axes[1].set_title('Post Temperature Scaling')



# =============== reliablity plot Before and After ===============


mosaic = [
    ["A0", "A1",],
    ["B0", "B1",],
]
subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)


bar_width = 0.45
# Create positions for the bars
positions1 = np.arange(len(bns))
positions2 = positions1 + bar_width
positions0 = (positions1+positions2)/2
custom_ticks = list(positions0)+[20]

y0 = get_values(bin_dict0, 'acc')
y1 = get_values(bin_dict1, 'acc')
y0a = get_values(bin_dict0a, 'acc')
y1a = get_values(bin_dict1a, 'acc')

i='A0'
# Create side-by-side bar plots
axes[i].bar(positions0, np.array(bns), width=bar_width*2, label='Expected', color='pink', alpha=0.5)
axes[i].bar(positions1, np.array(y0)/100, width=bar_width, label='CE', color='blue', alpha=0.5)
axes[i].bar(positions2, np.array(y1)/100, width=bar_width, label='LS', color='green', alpha=0.5)
axes[i].legend()
axes[i].set_ylabel('Accuracy')
axes[i].set_xlabel('Confidence')
axes[i].set_xticks(custom_ticks[::4], (bns+[1.0])[::4])
axes[i].set_title('Pre Temperature Scaling')
axes[i].grid(linestyle='--', alpha=0.5, zorder=0)

i='A1'
axes[i].bar(positions0, np.array(bns), width=bar_width*2, label='Expected', color='pink', alpha=0.5)
axes[i].bar(positions1, np.array(y0a)/100, width=bar_width, label='CE', color='blue', alpha=0.5)
axes[i].bar(positions2, np.array(y1a)/100, width=bar_width, label='LS', color='green', alpha=0.5)
axes[i].legend()
axes[i].set_ylabel('Accuracy')
axes[i].set_xlabel('Confidence')
axes[i].set_xticks(custom_ticks[::4], (bns+[1.0])[::4])
axes[i].set_title('Post Temperature Scaling')
axes[i].grid(linestyle='--', alpha=0.5, zorder=0)

y0 = get_values(bin_dict0, 'count')
y1 = get_values(bin_dict1, 'count')
y0a = get_values(bin_dict0a, 'count')
y1a = get_values(bin_dict1a, 'count')

i='B0' # Pre T-Scaling
axes[i].bar(positions1, np.array(y0), width=bar_width, label='CE', color='blue', alpha=0.5)
axes[i].bar(positions2, np.array(y1), width=bar_width, label='LS', color='green', alpha=0.5)
axes[i].legend()
axes[i].set_ylabel('% of samples')
axes[i].set_ylim(0,90)
axes[i].set_xlabel('Confidence')
axes[i].set_xticks(custom_ticks[::4], (bns+[1.0])[::4])
axes[i].grid(linestyle='--', alpha=0.5, zorder=0)

i='B1' # Post T-Scaling
axes[i].bar(positions1, np.array(y0a), width=bar_width, label='CE', color='blue', alpha=0.5)
axes[i].bar(positions2, np.array(y1a), width=bar_width, label='LS', color='green', alpha=0.5)
axes[i].legend()
axes[i].set_ylabel('% of samples')
axes[i].set_xlabel('Confidence')
axes[i].set_ylim(0,90)
axes[i].set_xticks(custom_ticks[::4], (bns+[1.0])[::4])
axes[i].grid(linestyle='--', alpha=0.5, zorder=0)



# ====== plot confidence
bns = [(i / float(num_bins)) for i in range(num_bins)]
num_samples = len(labels0)
y = []
for i in range(num_bins):
    n = (bin_dict0[i][COUNT] / float(num_samples)) * 100
    y.append(n)
plt.figure(figsize=(10, 8))  # width:20, height:3
plt.bar(bns, y, align='edge', width=0.05,
        color='blue', alpha=0.5, label='Percentage samples')
plt.ylabel('Percentage of samples')
plt.xlabel('Confidence')

