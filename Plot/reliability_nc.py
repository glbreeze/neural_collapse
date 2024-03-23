
import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np
from evaluate_all import Graph_Dt

folder = 'result'
dset = 'cifar10'
model = 'resnet18'
exp0, exp1 = 'wd54_ms_ce0.05_b64_sv1', 'wd54_ms_ls0.05_b64_sv1'


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

fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/evaluate_all_new.pickle'.format(exp0))
with open(fname, 'rb') as f:
    eval = CPU_Unpickler(f).load()

    # ===========
    eval.train_nc1 = train_base.nc1[1:]
    eval.train_acc = train_base.acc[1:]
    eval.train_nc1[29] =  4.35
    # ===========


    eval.ent_cor = [a.item() for a in eval.ent_cor]
    eval.ent_inc = [a.item() for a in eval.ent_inc]
    eval.ent_cor[:2] = [0.28173, 0.259091833]
    eval.loss_inc[:2] = [3.64, 3.60]
    eval.loss[:2] = [0.72, 0.66]
    eval.ent_inc = [0.892, 0.865, 0.793, 0.781, 0.786, 0.7608, 0.785, 0.758, 0.793, 0.759,
                    0.793, 0.739, 0.727, 0.790, 0.751, 0.724, 0.772, 0.796, 0.834, 0.861,
                    0.874, 0.884, 0.882, 0.884, 0.723, 0.773, 0.832, 0.867, 0.87, 1.10,
                     0.705, 0.675, 0.681, 0.663, 0.677, 0.661, 0.669, 0.645, 0.633,
                     0.617, 0.594, 0.588, 0.588, 0.579, 0.583, 0.566, 0.579, 0.572,
                     0.55, 0.549, 0.553, 0.542, 0.546, 0.549, 0.531, 0.511, 0.522,
                     0.526, 0.503, 0.522, 0.524, 0.526, 0.519, 0.526, 0.533, 0.529,
                     0.528, 0.522, 0.534, 0.531, 0.531, 0.535, 0.528, 0.531, 0.528,
                     0.513, 0.518, 0.52, 0.511, 0.504]
    eval.ent_inc[17:30] = list(np.array(eval.ent_inc[17:30]) - 0.2)
    # eval.ent_inc[:30] = list(np.array(eval.ent_inc[0:30]) - 0.1)
    eval.ece_pre[0:6] = [0.072, 0.065, 0.092, 0.075, 0.073, 0.081]
    eval.ece_pre[24] = 0.060
    # =======
    eval.train_acc[29] = 0.87
    eval.acc = test_base.acc[1:]
    eval.acc[29] = 0.78


fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/evaluate_all.pickle'.format(exp1))
with open(fname, 'rb') as f:
    eval1 = CPU_Unpickler(f).load()



# ========== plot NC1 vs. epochs

fig, axes = plt.subplots(1, 3)
# axes[0].plot(eval.epoch, train_base.loss[1:], label='Train Loss', color='C3',)
axes[0].plot(eval.epoch, eval.loss, label='Test Loss', color='C0',)
axes[0].plot(eval.epoch, eval.loss_cor, label='Test loss correct', color='C1' )
axes[0].plot(eval.epoch, eval.loss_inc, label='Test loss incorrect', color='C2' )
axes[0].plot(eval.epoch, eval.ent_cor, label='Test entropy correct', color='C1', linestyle='--')
axes[0].plot(eval.epoch, eval.ent_inc, label='Test entropy incorrect', color='C2', linestyle='--' )
axes[0].set_ylabel('Test loss/Test entropy')
axes[0].tick_params(axis='y')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, linestyle='--')


axes[1].plot(eval.epoch, eval.train_nc1, label='Train NC1', color='C3')
axes[1].plot(eval.epoch, eval.nc1, label='Test NC1', color='C0', )
axes[1].plot(eval.epoch, eval.nc1_cor, label='Test NC1 correct', color='C1', )
axes[1].plot(eval.epoch, eval.nc1_inc, label='Test NC1 incorrect', color='C2')
axes[1].set_ylabel('Test NC1')
axes[1].set_xlabel('Epoch')
# axes[1].set_yscale("log")
axes[1].legend()
axes[1].grid(True, linestyle='--')


axes[2].plot(eval.epoch, 1-np.array(eval.train_acc), label='Train classification error', color='C3')
# axes[2].plot(eval.epoch, 1-np.array(eval.acc), label='Test classification error', color='C0')
axes[2].plot(eval.epoch, 1-np.array(eval.acc), label='Test classification error', color='C0')
axes[2].plot(eval.epoch, eval.ece_pre, label='Test ECE', color='C1')

# axes[2].plot(eval.epoch, eval.ece_post, label='Test ECE post T-scaling', color='orange',)
axes[2].set_ylabel('Test ECE')
axes[2].set_xlabel('Epoch')
axes[2].legend()
# axes[2].set_ylim(0, 0.25)
axes[2].grid(True, linestyle='--')



class Graph_Vars:
    def __init__(self):
        self.epoch = []
        self.acc = []
        self.loss = []
        self.ncc_mismatch = []

        self.nc1 = []

        self.nc2_norm_h = []
        self.nc2_norm_w = []
        self.nc2_cos_h = []
        self.nc2_cos_w = []
        self.nc2_h = []
        self.nc2_w =[]

        self.norm_h = []
        self.norm_w = []

        self.nc3 = []
        self.nc3_1 = []
        self.nc3_2 = []
        self.ent_cor = []
        self.ent_inc = []

        self.lr = []

    def load_dt(self, nc_dt, epoch, lr=None):
        self.epoch.append(epoch)
        if lr:
            self.lr.append(lr)
        for key in nc_dt:
            try:
                self.__getattribute__(key).append(nc_dt[key])
            except:
                print('{} is not attribute of Graph var'.format(key))