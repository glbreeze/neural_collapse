import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np


folder = 'result'
dset = 'stl10'
model = 'resnet50'
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
# ========== plot Accuracy vs. epochs
i = 0
axes[i].plot(epochs, 1- np.array(train_base.acc), label='Baseline Train Error', color = 'C0', linestyle='dashed' )
axes[i].plot(epochs, 1- np.array(train_new1.acc), label='Label Smoothing Train Error', color = 'C1', linestyle='dashed' )
axes[i].plot(epochs, 1- np.array(test_base.acc), label='Baseline Test Error',  color = 'C0')
axes[i].plot(epochs, 1- np.array(test_new1.acc), label='Label Smoothing Test Error',  color = 'C1' )
axes[i].set_ylabel('Error Rate')
axes[i].set_xlabel('Epoch')
axes[i].legend()
axes[i].set_title('Error Rate vs. Epochs')


# ========== plot NC1 vs. epochs
i=1
axes[i].plot(epochs, train_base.nc1, label='Baseline')
axes[i].plot(epochs, train_new1.nc1, label='Label Smoothing' )
axes[i].set_ylabel('NC1')
axes[i].set_xlabel('Epoch')
# plt.ylim(7e-2, 1e4)
axes[i].set_yscale("log");
axes[i].legend()
axes[i].set_title('NC1 vs. Epochs')

# ========== plot NC2 vs. epochs
i=2
axes[i].plot(epochs, train_base.nc2_h, label='Baseline-H', color='C0')
axes[i].plot(epochs, train_new1.nc2_h, label='Label Smoothing-H', color='C1' )
axes[i].plot(epochs, train_base.nc2_w, label='Baseline-W',  linestyle='dashed', color='C0')
axes[i].plot(epochs, train_new1.nc2_w, label='Label Smoothing-W',  linestyle='dashed', color='C1')
axes[i].set_ylabel('NC2: Simplex ETF')
axes[i].set_xlabel('Epoch')
axes[i].legend()
axes[i].set_title('NC2: Simplex ETF vs. Epochs')


# ========== plot NC2-1 vs. epochs (Norm)
fig, ax = plt.subplots(1)
ax.plot(epochs, train_base.nc2_cos_h, label='Baseline-H', color='C0')
ax.plot(epochs, train_new1.nc2_cos_h, label='Label Smoothing-H', color='C1' )
ax.plot(epochs, train_base.nc2_norm_w, label='Baseline-W',  linestyle='dashed', color='C0')
ax.plot(epochs, train_new1.nc2_norm_w, label='Label Smoothing-W',  linestyle='dashed', color='C1')
plt.ylabel('NC2-1: EquiNorm')
plt.xlabel('Epoch')
plt.legend()
plt.title('NC2-1: EquiNorm vs. Epochs')


# ========== plot NC2-2 vs. epochs (Cos)
fig, ax = plt.subplots(1)
ax.plot(epochs, train_base.nc2_cos_h, label='Baseline-H', color='C0')
ax.plot(epochs, train_new1.nc2_cos_h, label='Label Smoothing-H', color='C1')
ax.plot(epochs, train_base.nc2_cos_w, label='Baseline-W', linestyle='dashed', color='C0')
ax.plot(epochs, train_new1.nc2_cos_w, label='Label Smoothing-W', linestyle='dashed', color='C1')
plt.ylabel('NC2-2: Equiangular')
plt.xlabel('Epoch')
plt.legend()
plt.title('NC2-2: Equiangular vs. Epochs')


# ========== plot new NC3 vs. epochs
fig, ax = plt.subplots(1)
ax.plot(epochs, train_base.nc3, label='Baseline', color='C0')
ax.plot(epochs, train_new1.nc3, label='Label Smoothing', color='C1' )
plt.ylabel('NC3: Self-Duality')
plt.xlabel('Epoch')
plt.legend()
plt.title('NC3: Self-Duality vs. Epochs')


# ============= plot NC1 vs NC2

fig, ax = plt.subplots(1)
ax.scatter(train_base.nc2_cos_h, train_base.nc1, label='Baseline', s=15,)
ax.scatter(train_new1.nc2_cos_h, train_new1.nc1, label='Label Smoothing', marker='^', s=15, )
plt.ylabel('NC1')
plt.xlabel('NC2-2: Equiangular')
ax.set_yscale("log")
ax.set_xlim(0.02, 0.3)
plt.legend()
plt.title('NC1 vs. NC2(Equiangular)')

fig, ax = plt.subplots(1)
ax.scatter(train_base.nc2_h, train_base.nc1, label='Baseline', s=15,)
ax.scatter(train_new1.nc2_h, train_new1.nc1, label='Label Smoothing', marker='^', s=15, )
plt.ylabel('NC1')
plt.xlabel('NC2: Simplex ETF')
ax.set_yscale("log")
ax.set_xlim(0.1, 0.9)
plt.legend()
plt.title('NC1 vs. NC2(Simplex ETF)')

# ============= plot NC1 vs NC2 and Testing Acc
if dset == 'cifar100':
    vmin, vmax = 0.60, 0.65
elif dset=='cifar10':
    vmin, vmax = 0.86, 0.90
elif dset=='stl10':
    vmin, vmax = 0.59, 0.67


fig, ax = plt.subplots(1)
p = ax.scatter(np.array(train_new1.nc2_h), np.array(train_new1.nc1), c=np.array(test_new1.acc),
               cmap= 'viridis', vmin=vmin, vmax=vmax, label='Label Smoothing', marker='^')
plt.xlabel('NC2: ETF')
plt.ylabel('NC1')
ax.set_yscale("log")
plt.xlim(0.1, 0.9)
plt.title('NC1 vs. NC2(ETF) and Test Acc for LS Loss')
fig.colorbar(p)
# for i, txt in enumerate(exam_epochs[start_idx:]):
#     plt.annotate(txt, (np.array(train_new1.cos_M[start_idx:])[i], np.array(train_new1.Sw_invSb[start_idx:])[i]))


fig, ax = plt.subplots(1)
p = ax.scatter(np.array(train_base.nc2_h), np.array(train_base.nc1), c=np.array(test_base.acc),
                cmap= 'viridis', vmin=vmin, vmax=vmax, label='Baseline')
plt.xlabel('NC2: ETF')
plt.ylabel('NC1')
ax.set_yscale("log")
plt.xlim(0.1, 0.9)
plt.title('NC1 vs. NC2(ETF) and Test Acc for Baseline')
fig.colorbar(p)

# ===== overlay the scatter plot for baseline and LS

fig, ax = plt.subplots(1)
p = ax.scatter(np.array(train_base.nc2_h), np.array(train_base.nc1), label='Baseline', marker='o', s=15,
               c=np.array(test_base.acc), cmap= 'viridis', vmin=vmin, vmax=vmax)
p = ax.scatter(np.array(train_new1.nc2_h), np.array(train_new1.nc1), label='Label Smoothing', marker='^', s=15,
               c=np.array(test_new1.acc), cmap= 'viridis', vmin=vmin, vmax=vmax)
plt.xlabel('NC2: Simplex ETF')
plt.ylabel('NC1')
ax.set_yscale("log")
plt.xlim(0.0, 0.8)
plt.title('NC1 vs. NC2(Simplex ETF) and Test Acc')
fig.colorbar(p)
# for i, txt in enumerate(exam_epochs[start_idx:]):
#     plt.annotate(txt, (np.array(train_new1.cos_M[start_idx:])[i], np.array(train_new1.Sw_invSb[start_idx:])[i]))







plt.scatter(train_base.Sw_invSb, train_base.norm_M_CoV, label='Baseline')
plt.scatter(train_new1.Sw_invSb, train_new1.norm_M_CoV, label='Label Smoothing')
plt.xlabel('NC1')
plt.ylabel('NC2-2: Equinorm')
plt.legend()
plt.title('NC1 vs. NC2(Equinorm)')


plt.scatter(train_base.Sw_invSb, train_base.cos_W, label='Baseline')
plt.scatter(train_new1.Sw_invSb, train_new1.cos_W, label='Label Smoothing')
plt.xlabel('NC1')
plt.ylabel('NC2-2: Equiangular-W')
plt.legend()
plt.title('NC1 vs. NC2(Equiangular-W)')


plt.scatter(train_base.Sw_invSb, train_base.norm_W_CoV, label='Baseline')
plt.scatter(train_new1.Sw_invSb, train_new1.norm_W_CoV, label='Label Smoothing')
plt.xlabel('NC1')
plt.ylabel('NC2-2: Equinorm-W')
plt.legend()
plt.title('NC1 vs. NC2(Equinorm-W)')




plt.plot(1-np.array(test_base.accuracy),  label='Baseline')
plt.plot(1-np.array(test_new1.accuracy), label='Label Smoothing')
plt.xlabel('NC1')
plt.ylabel('NC2-2: Equinorm-W')
plt.legend()
plt.title('NC1 vs. NC2(Equinorm-W)')




plt.plot(exam_epochs, train_base.Sw_invSb, label='Base WD')
plt.plot(exam_epochs, train_new1.Sw_invSb, label='WD on CLS')
plt.legend()
plt.title('Training NC1 when training {} on {}'.format(model, dset))


plt.plot(exam_epochs, train_base.norm_M_CoV, label='Base WD')
plt.plot(exam_epochs, train_new1.norm_M_CoV, label='WD on CLS')
plt.plot(exam_epochs, train_new2.norm_M_CoV, label='WD on BN & CLS')
plt.legend()
plt.title('Training NC2-1 when training {} on {}'.format(model, dset))

plt.plot(exam_epochs, train_base.cos_M, label='Base WD')
plt.plot(exam_epochs, train_new1.cos_M, label='WD on CLS')
plt.plot(exam_epochs, train_new2.cos_M, label='WD on BN & CLS')
plt.legend()
plt.title('Training NC2-2 when training {} on {}'.format(model, dset))


plt.plot(exam_epochs, train_base.cos_W, label='Base WD')
plt.plot(exam_epochs, train_new1.cos_W, label='WD on CLS')
plt.plot(exam_epochs, train_new2.cos_W, label='WD on BN & CLS')
plt.legend()
plt.title('Training NC2-2(W) when training {} on {}'.format(model, dset))


plt.semilogy(exam_epochs, train_base.loss, label='Base WD')
plt.semilogy(exam_epochs, train_new1.loss, label='WD on CLS')
plt.semilogy(exam_epochs, train_new2.loss, label='WD on BN & CLS')
plt.legend()
plt.title('Training Loss when training {} on {}'.format(model, dset))


plt.plot(exam_epochs, 1-np.array(test_base.accuracy), label='Base WD')
plt.plot(exam_epochs, 1-np.array(test_new1.accuracy), label='WD on CLS')
plt.plot(exam_epochs, 1-np.array(test_new2.accuracy), label='WD on BN & CLS')
plt.legend()
plt.title('Test Error when training {} on {}'.format(model, dset))




