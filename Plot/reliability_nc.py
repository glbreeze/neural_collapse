
from matplotlib import pyplot as plt

import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np

folder = 'result'
dset = 'cifar10'
model = 'resnet18'
exp0, exp1 = 'wd54_ms_ce0.05_b64_sv', 'wd54_ms_ls0.05_b64_sv'


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
epochs = train_base.epoch

# ========== plot NC1 vs. epochs
fig, ax = plt.subplots(1)
ax.plot(train_base.epoch, train_base.nc1, label='Baseline')
ax.plot(train_base.epoch, train_new1.nc1, label='Label Smoothing' )
plt.ylabel('NC1')
plt.xlabel('Epoch')
ax.set_yscale("log");
plt.legend()
plt.title('NC1 vs. Epochs')

# ========== plot NC1 vs. epochs
fig, ax = plt.subplots(1)
ax.plot(train_base.epoch, train_base.nc1, label='Baseline')
ax.plot(train_base.epoch, test_base.nc1, label='Test NC1' )
plt.ylabel('NC1')
plt.xlabel('Epoch')
ax.set_yscale("log");
plt.legend()
plt.title('NC1 vs. Epochs')

# ========== plot Accuracy vs. epochs
fig, ax = plt.subplots(1)
ax.plot(epochs, 1- np.array(train_base.acc), label='Baseline Train Error', color = 'C0', linestyle='dashed' )
ax.plot(epochs, 1- np.array(train_new1.acc), label='Label Smoothing Train Error', color = 'C1', linestyle='dashed' )
ax.plot(epochs, 1- np.array(test_base.acc), label='Baseline Test Error',  color = 'C0')
ax.plot(epochs, 1- np.array(test_new1.acc), label='Label Smoothing Test Error',  color = 'C1' )
plt.ylabel('Error Rate')
plt.xlabel('Epoch')
plt.legend()
plt.title('Error Rate vs. Epochs')


# ========== plot Accuracy vs. epochs
fig, ax = plt.subplots(1)
ax.plot(epochs, np.array(train_base.loss), label='Baseline Train Error', color = 'C0', linestyle='dashed' )
ax.plot(epochs, np.array(train_new1.loss), label='Label Smoothing Train Error', color = 'C1', linestyle='dashed' )
ax.plot(epochs, np.array(test_base.loss), label='Baseline Test Error',  color = 'C0')
ax.plot(epochs, np.array(test_new1.loss), label='Label Smoothing Test Error',  color = 'C1' )
plt.ylabel('Error Rate')
plt.xlabel('Epoch')
plt.legend()
plt.title('Error Rate vs. Epochs')