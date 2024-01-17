import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np


folder = 'result'
dset = 'cifar10'
model = 'resnet18'
exps = ['ms_ce_gn_b4', 'ms_ce_gn_b8_s2', 'ms_ce_gn_b16', 'ms_ce_gn_b32', 'ms_ce_gn_b64']
# exps = ['wd54_ms_ce_b64', 'wd54_ms_ls_b64_e02', 'wd54_ms_ls_b64', 'wd54_ms_ls_b64_e08', 'wd54_ms_ls_b64_e1', 'wd54_ms_ls_b64_e2', 'wd54_ms_ls_b64_e5', 'wd54_ms_ls_b64_e8']



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

train_nc = {}
test_nc = {}
# statistics on training set / test set
for exp in exps:
    fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph1.pickle'.format(exp))
    with open(fname, 'rb') as f:
        train_nc[exp] = CPU_Unpickler(f).load()

    fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph2.pickle'.format(exp))
    with open(fname, 'rb') as f:
        test_nc[exp] = CPU_Unpickler(f).load()


fig, axes = plt.subplots(1, 1)
epochs = train_nc[exps[0]].epoch
# ========== plot Accuracy vs. epochs
i = 0
for exp in exps:
    axes.plot(epochs, 1- np.array(train_nc[exp].acc), label='train:'+exp, linestyle='dashed' )
axes.set_ylabel('Error Rate')
axes.set_xlabel('Epoch')
axes.legend()
axes.set_title('Train Error Rate vs. Epochs')

# ========== plot Accuracy vs. epochs
fig, axes = plt.subplots(1, 1)
epochs = train_nc[exps[0]].epoch
for exp in exps:
    axes.plot(epochs, 1- np.array(test_nc[exp].acc), label='test:'+exp, )
axes.set_ylabel('Error Rate')
axes.set_xlabel('Epoch')
axes.legend()
axes.set_title('Error Rate vs. Epochs')


# ========== plot NC1 vs. epochs
fig, axes = plt.subplots(1, 2)
epochs = train_nc[exps[0]].epoch
for exp in exps:
    axes.plot(epochs, train_nc[exp].nc1, label='nc1:'+exp)
axes.set_ylabel('NC1')
axes.set_xlabel('Epoch')
# plt.ylim(7e-2, 1e4)
axes.set_yscale("log");
axes.legend()
axes.set_title('NC1 vs. Epochs')

# ========== plot NC2 vs. epochs
fig, axes = plt.subplots(1, 1)
for exp in exps:
    axes.plot(epochs, train_nc[exp].nc2_h, label='nc2-H:'+exp,)
axes.set_ylabel('NC2-H: Simplex ETF')
axes.set_xlabel('Epoch')
axes.legend()
axes.set_title('NC2-H: Simplex ETF vs. Epochs')

# ========== plot NC2 vs. epochs
fig, axes = plt.subplots(1, 1)
for exp in exps:
    axes.plot(epochs, train_nc[exp].nc2_w, label='nc2-W:'+exp,)
axes.set_ylabel('NC2-W: Simplex ETF')
axes.set_xlabel('Epoch')
axes.legend()
axes.set_title('NC2-W: Simplex ETF vs. Epochs')



# ============= plot NC1 vs NC2

fig, ax = plt.subplots(1)
for exp in exps:
    ax.scatter(train_nc[exp].nc2_cos_h, train_nc[exp].nc1, label=exp, s=15,)  # marker='^'
plt.ylabel('NC1')
plt.xlabel('NC2-2: Equiangular')
ax.set_yscale("log")
# ax.set_xlim(0, 0.15)
plt.legend()
plt.title('NC1 vs. NC2(Equiangular)')

fig, ax = plt.subplots(1)
ax.scatter(train_base.nc2_h, train_base.nc1, label='Baseline', s=15,)
ax.scatter(train_new1.nc2_h, train_new1.nc1, label='Label Smoothing', marker='^', s=15, )
plt.ylabel('NC1')
plt.xlabel('NC2: Simplex ETF')
ax.set_yscale("log")
ax.set_xlim(0.2, 0.9)
plt.legend()
plt.title('NC1 vs. NC2(Simplex ETF)')

# ============= plot NC1 vs NC2 and Testing Acc
if dset == 'cifar100':
    vmin, vmax = 0.60, 0.65
elif dset=='cifar10':
    vmin, vmax = 0.86, 0.90
elif dset=='stl10':
    vmin, vmax = 0.56, 0.61


fig, ax = plt.subplots(1)
p = ax.scatter(np.array(train_new1.nc2_h), np.array(train_new1.nc1), c=np.array(test_new1.acc),
               cmap= 'viridis', vmin=vmin, vmax=vmax, label='Label Smoothing',)
plt.xlabel('NC2: ETF')
plt.ylabel('NC1')
ax.set_yscale("log")
plt.xlim(0.2, 0.9)
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
plt.xlim(0.2, 0.9)
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




