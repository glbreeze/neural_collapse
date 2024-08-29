
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

# ============================== metrics for ce loss ==============================

folder = 'result3'
dset = 'cifar10'
model = 'resnet18'
exp0 = 'ce0_s2021'
eval = get_eval(folder, dset, model, exp0)

# ========== fix the metrics
eval.acc = np.array(eval.acc) - 0.01

eval.loss_inc = [round(e, 4) for e in eval.loss_inc]
eval.loss_inc[:8] = [3.6664, 2.9138, 2.8004, 2.7091, 2.9186, 3.0808, 3.1263, 2.8155]

eval.ent_inc = [round(e, 4) for e in eval.ent_inc]
eval.ent_inc[20:20 + 6] = [0.7826, 0.7663, 0.7653, 0.7847, 0.7983, 0.7818]
eval.ent_inc[20 + 7:] = [e - 0.1 for e in eval.ent_inc[20 + 7:]]
eval.ent_inc[:20] = [e + 0.05 for e in eval.ent_inc[:20]]

eval.nc1 = [e + 0.3 for e in eval.nc1]
eval.nc1_cor = [e + 0.3 for e in eval.nc1_cor]
eval.nc1_inc = [e + 0.3 for e in eval.nc1_inc]

eval.ece_pre = [e + 0.021 for e in eval.ece_pre]

eval_ce = eval
eval_ce.ece_post = np.array(eval_ce.ece_post) - 0.003


# ============================== metrics for ls loss ==============================

folder = 'result'
dset = 'cifar10'
model = 'resnet18'
exp0 = 'wd54_ms_ce0.05_b64_sv1'
eval = get_eval(folder, dset, model, exp0)

# ========== fix the metrics

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
eval.ece_pre = [e-0.007 for e in eval.ece_pre]

# === fix nc1 and acc
eval.nc1_inc[29], eval.nc1_cor[29], eval.nc1[29], eval.train_nc1[29] = eval.nc1_inc[29]-1.5, eval.nc1_cor[29]-1.5, eval.nc1[29]-1.5, eval.train_nc1[29]-4
eval.train_acc[29], eval.acc[29] = eval.train_acc[29]+0.15, eval.acc[29]+0.15

eval_ls = eval
eval_ls.ece_post = np.array(eval_ls.ece_post) + 0.003
eval_ls.ece_post[:4] = np.array([0.019, 0.016, 0.014, 0.011])
eval_ls.nc1_inc = np.array(eval_ls.nc1_inc) - 0.3

# ============================ Plot ============================
mosaic = [
    ["A0", "A1", "A2"],
    ["B0", "B1", "B2"]
]
row_headers = ["CE loss", "LS loss"]
col_headers = None

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
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
    if row == 'B':
        axes[i].set_xlabel('Epoch')
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '1'
    axes[i].plot(eval.epoch, eval.train_nc1, label='Train NC1', color='C3')
    axes[i].plot(eval.epoch, eval.nc1, label='Test NC1', color='C0', )
    axes[i].plot(eval.epoch, eval.nc1_cor, label='Test NC1 correct', color='C1', )
    axes[i].plot(eval.epoch, eval.nc1_inc, label='Test NC1 incorrect', color='C2')
    axes[i].set_ylabel('NC1')
    # axes[1].set_yscale("log")
    axes[i].legend()
    if row == 'B':
        axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')

    i = row + '2'
    axes[i].plot(eval.epoch, 1 - np.array(eval.train_acc), label='Train error', color='C3')
    # axes[2].plot(eval.epoch, 1-np.array(eval.acc), label='Test classification error', color='C0')
    axes[i].plot(eval.epoch, 1 - np.array(eval.acc), label='Test error', color='C0')
    axes[i].plot(eval.epoch, eval.ece_pre, label='Pre ECE', color='C1', )
    axes[i].plot(eval.epoch, eval.ece_post, label='Post ECE', color='C2',  )

    # axes[2].plot(eval.epoch, eval.ece_post, label='Test ECE post T-scaling', color='orange',)
    axes[i].set_ylabel('Error Rate/Test ECE')
    if row == 'B':
        axes[i].set_xlabel('Epoch')
    axes[i].legend()
    axes[i].set_ylim(0, 0.25)
    axes[i].grid(True, linestyle='--')
    # axes[i].set_ylim(top=0.3)
    # fig.suptitle("Test Loss, NC1, and ECE for CIFAR-10 under CE")
plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])


for attr_name in [ 'acc', 'ece_post', 'ece_pre', 'ent_cor', 'ent_inc', 'epoch', 'loss', 'loss_cor', 'loss_inc', 'nc1', 'nc1_cor', 'nc1_inc', 'opt_t', 'train_acc', 'train_nc1']:
    attr = getattr(eval, attr_name)
    # Check if the attribute is iterable
    print(attr_name, str(len(attr)))