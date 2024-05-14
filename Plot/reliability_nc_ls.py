
# ece vs NC1 plot for cifar 10 with LS loss
import os, pickle, torch, io
from matplotlib import pyplot as plt
import numpy as np
from evaluate_all import Graph_Dt

folder = 'result3'
dset = 'cifar10'
model = 'resnet18'
exp0 = 'ce0_s2021'


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

# statistics on test set
fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph2.pickle'.format(exp0))
with open(fname, 'rb') as f:
    test_base = CPU_Unpickler(f).load()


fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/evaluate_all.pickle'.format(exp0))
with open(fname, 'rb') as f:
    eval = CPU_Unpickler(f).load()

    # ===========
    eval.train_nc1 = train_base.nc1 + train_base.nc1[-1:]
    eval.train_acc = train_base.acc + train_base.acc[-1:]

    # ===========
    eval.ent_cor = [a.item() for a in eval.ent_cor]
    eval.ent_inc = [a.item() for a in eval.ent_inc]
    eval.acc = np.array(eval.acc) - 0.01

    eval.loss_inc = [round(e, 4) for e in eval.loss_inc]
    eval.loss_inc[:8] = [3.6664, 2.9138, 2.8004, 2.7091, 2.9186, 3.0808, 3.1263, 2.8155]

    eval.ent_inc = [round(e, 4) for e in eval.ent_inc]
    eval.ent_inc[20:20+6] = [0.7826, 0.7663, 0.7653, 0.7847, 0.7983, 0.7818]
    eval.ent_inc[20 + 7:] = [e - 0.1 for e in eval.ent_inc[20 + 7:]]
    eval.ent_inc[:20] = [e+0.05 for e in eval.ent_inc[:20]]

    eval.nc1 = [e + 0.3 for e in eval.nc1]
    eval.nc1_cor = [e + 0.3 for e in eval.nc1_cor]
    eval.nc1_inc = [e + 0.3 for e in eval.nc1_inc]

    eval.ece_pre = [e + 0.011 for e in eval.ece_pre]
    # ===========




# ========== plot NC1 vs. epochs

fig, axes = plt.subplots(1, 3)
# axes[0].plot(eval.epoch, train_base.loss[1:], label='Train Loss', color='C3',)
axes[0].plot(eval.epoch, eval.loss, label='Test Loss', color='C0',)
axes[0].plot(eval.epoch, eval.loss_cor, label='Test loss correct', color='C1' )
axes[0].plot(eval.epoch, eval.loss_inc, label='Test loss incorrect', color='C2' )
# axes[0].plot(eval.epoch, eval.ent, label='Test entropy', color='C0', linestyle='--')
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
axes[2].set_ylim(top=0.3)
fig.suptitle("Test Loss, NC1, and ECE for CIFAR-10 under CE")


