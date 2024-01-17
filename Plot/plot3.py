import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np


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
