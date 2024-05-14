
# ece vs NC1 plot for cifar 10 with LS loss
import os, pickle, torch, io
from matplotlib import pyplot as plt
import numpy as np
from Plot.utils import add_headers
from evaluate_all import Graph_Dt



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def get_eval(folder, dset, model, exp0):
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
        if len(train_base.epoch)>len(eval.epoch):
            eval.train_nc1 = train_base.nc1[1:]
            eval.train_acc = train_base.acc[1:]
        elif len(train_base.epoch)<len(eval.epoch):
            eval.train_nc1 = train_base.nc1 + train_base.nc1[-1:]
            eval.train_acc = train_base.acc + train_base.acc[-1:]

        # ===========
        eval.ent_cor = [a.item() for a in eval.ent_cor]
        eval.ent_inc = [a.item() for a in eval.ent_inc]

    return eval

# ========== metrics for ce loss

folder = 'result3'
dset = 'cifar10'
model = 'resnet18'
exp0 = 'resnet18_ls0_b128_s2021'
eval = get_eval(folder, dset, model, exp0)

# ========== fix the metrics

eval_ce = eval


# ========== metrics for ls loss

folder = 'result3'
dset = 'cifar10'
model = 'resnet18'
exp0 = 'resnet18_ls0.05_b128_s2021'
eval = get_eval(folder, dset, model, exp0)

# ========== fix the metrics



eval_ls = eval

# ============================ Plot ============================
mosaic = [
    ["A0", "A1", "A2"],
    ["B0", "B1", "B2"]
]
row_headers = ["CE loss", "LS loss"]
col_headers = ["Error Rate", "NC1", "NC2", "NC3", "Norm-H/W"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for row in ['A','B']:
    if row == 'A':
        eval = eval_ce
    elif row =='B':
        eval = eval_ls

    i = row + '0'
    axes[i].plot(eval.epoch, eval.loss, label='Test Loss', color='C0', )
    axes[i].plot(eval.epoch, eval.loss_cor, label='Test loss correct', color='C1')
    axes[i].plot(eval.epoch, eval.loss_inc, label='Test loss incorrect', color='C2')
    # axes[0].plot(eval.epoch, eval.ent, label='Test entropy', color='C0', linestyle='--')
    axes[i].plot(eval.epoch, eval.ent_cor, label='Test entropy correct', color='C1', linestyle='--')
    axes[i].plot(eval.epoch, eval.ent_inc, label='Test entropy incorrect', color='C2', linestyle='--')
    axes[i].set_ylabel('Test loss/Test entropy')
    axes[i].tick_params(axis='y')
    axes[i].set_xlabel('Epoch')
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '1'
    axes[i].plot(eval.epoch, eval.train_nc1, label='Train NC1', color='C3')
    axes[i].plot(eval.epoch, eval.nc1, label='Test NC1', color='C0', )
    axes[i].plot(eval.epoch, eval.nc1_cor, label='Test NC1 correct', color='C1', )
    axes[i].plot(eval.epoch, eval.nc1_inc, label='Test NC1 incorrect', color='C2')
    axes[i].set_ylabel('Test NC1')
    axes[i].set_xlabel('Epoch')
    # axes[1].set_yscale("log")
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '2'
    axes[i].plot(eval.epoch, 1 - np.array(eval.train_acc), label='Train classification error', color='C3')
    # axes[2].plot(eval.epoch, 1-np.array(eval.acc), label='Test classification error', color='C0')
    axes[i].plot(eval.epoch, 1 - np.array(eval.acc), label='Test classification error', color='C0')
    axes[i].plot(eval.epoch, eval.ece_pre, label='Test ECE', color='C1')

    # axes[2].plot(eval.epoch, eval.ece_post, label='Test ECE post T-scaling', color='orange',)
    axes[i].set_ylabel('Test ECE')
    axes[i].set_xlabel('Epoch')
    axes[i].legend()
    # axes[2].set_ylim(0, 0.25)
    axes[i].grid(True, linestyle='--')
    axes[i].set_ylim(top=0.3)
    fig.suptitle("Test Loss, NC1, and ECE for CIFAR-10 under CE")
plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])


for attr_name in [ 'acc', 'ece_post', 'ece_pre', 'ent_cor', 'ent_inc', 'epoch', 'loss', 'loss_cor', 'loss_inc', 'nc1', 'nc1_cor', 'nc1_inc', 'opt_t', 'train_acc', 'train_nc1']:
    attr = getattr(eval, attr_name)
    # Check if the attribute is iterable
    print(attr_name, str(len(attr)))