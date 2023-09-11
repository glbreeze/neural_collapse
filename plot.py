import os, pickle
from matplotlib import pyplot as plt
from utils import Graph_Vars
from main import exam_epochs
import numpy as np

folder = 'result_09'
dset = 'cifar100'
model = 'resnet50'
exp0, exp1 = 'wd54_ce_etf', 'wd54_ls_etf'

# statistics on training set
fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph1.pickle'.format(exp0))
with open(fname, 'rb') as f:
    train_base = pickle.load(f)

fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph1.pickle'.format(exp1))
with open(fname, 'rb') as f:
    train_new1 = pickle.load(f)

# statistics on test set
fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph2.pickle'.format(exp0))
with open(fname, 'rb') as f:
    test_base = pickle.load(f)

fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph2.pickle'.format(exp1))
with open(fname, 'rb') as f:
    test_new1 = pickle.load(f)

# ========== plot NC1 vs. epochs
fig, ax = plt.subplots(1)
ax.plot(exam_epochs, train_base.Sw_invSb, label='Baseline')
ax.plot(exam_epochs, train_new1.Sw_invSb, label='Label Smoothing' )
plt.ylabel('NC1')
plt.xlabel('Epoch')
ax.set_yscale("log");
plt.legend()
plt.title('NC1 vs. Epochs')

# ========== plot NC2-2 vs. epochs
fig, ax = plt.subplots(1)
ax.plot(exam_epochs, train_base.cos_M, label='Baseline-H', color='C0')
ax.plot(exam_epochs, train_new1.cos_M, label='Label Smoothing-H', color='C1')
ax.plot(exam_epochs, train_base.cos_W, label='Baseline-W', linestyle='dashed', color='C0')
ax.plot(exam_epochs, train_new1.cos_W, label='Label Smoothing-W', linestyle='dashed', color='C1')
plt.ylabel('NC2-2: Equiangular')
plt.xlabel('Epoch')
plt.legend()
plt.title('NC2-2: Equiangular vs. Epochs')


# ========== plot NC2-1 vs. epochs
fig, ax = plt.subplots(1)
ax.plot(exam_epochs, train_base.norm_M_CoV, label='Baseline-H', color='C0')
ax.plot(exam_epochs, train_new1.norm_M_CoV, label='Label Smoothing-H', color='C1' )
ax.plot(exam_epochs, train_base.norm_W_CoV, label='Baseline-W',  linestyle='dashed', color='C0')
ax.plot(exam_epochs, train_new1.norm_W_CoV, label='Label Smoothing-W',  linestyle='dashed', color='C1')
plt.ylabel('NC2-1: EquiNorm')
plt.xlabel('Epoch')
plt.legend()
plt.title('NC2-1: EquiNorm vs. Epochs')

# ========== plot NC3 vs. epochs
fig, ax = plt.subplots(1)
ax.plot(exam_epochs, train_base.W_M_dist, label='Baseline', color='C0')
ax.plot(exam_epochs, train_new1.W_M_dist, label='Label Smoothing', color='C1' )
plt.ylabel('NC3: Self-Duality')
plt.xlabel('Epoch')
plt.legend()
plt.title('NC3: Self-Duality vs. Epochs')

# ========== plot Accuracy vs. epochs
fig, ax = plt.subplots(1)
ax.plot(exam_epochs, 1- np.array(train_base.accuracy), label='Baseline Train Error', color = 'C0', linestyle='dashed' )
ax.plot(exam_epochs, 1- np.array(train_new1.accuracy), label='Label Smoothing Train Error', color = 'C1', linestyle='dashed' )
ax.plot(exam_epochs, 1- np.array(test_base.accuracy), label='Baseline Test Error',  color = 'C0')
ax.plot(exam_epochs, 1- np.array(test_new1.accuracy), label='Label Smoothing Test Error',  color = 'C1' )
plt.ylabel('Error Rate')
plt.xlabel('Epoch')
plt.legend()
plt.title('Error Rate vs. Epochs')


# ============= plot NC1 vs NC2
start_idx = exam_epochs.index(100)
end_idx = exam_epochs.index(1000)

fig, ax = plt.subplots(1)
ax.scatter(train_base.cos_M[start_idx:], train_base.Sw_invSb[start_idx:], label='Baseline')
ax.scatter(train_new1.cos_M[start_idx:], train_new1.Sw_invSb[start_idx:], label='Label Smoothing' )
plt.ylabel('NC1')
plt.xlabel('NC2-2: Equiangular')
ax.set_yscale("log");
plt.legend()
plt.title('NC1 vs. NC2(Equiangular)')

# ============= plot NC1 vs NC2 and Testing Acc
if dset == 'cifar100':
    vmin, vmax = 0.59, 0.64
elif dset=='cifar10':
    vmin, vmax = 0.85, 0.90
elif dset=='stl10':
    vmin, vmax = 0.56, 0.61


fig, ax = plt.subplots(1)
p = ax.scatter(np.array(train_new1.cos_M[start_idx:end_idx]), np.array(train_new1.Sw_invSb[start_idx:end_idx]), label='Label Smoothing',
               c=np.array(test_new1.accuracy[start_idx:end_idx]), cmap= 'viridis', vmin=vmin, vmax=vmax)
plt.xlabel('NC2-2: Equiangular')
plt.ylabel('NC1')
ax.set_yscale("log")
plt.xlim(0.0, 0.10)
plt.title('NC1 vs. NC2(Equiangular) and Test Acc for Label Smooth')
fig.colorbar(p)
# for i, txt in enumerate(exam_epochs[start_idx:]):
#     plt.annotate(txt, (np.array(train_new1.cos_M[start_idx:])[i], np.array(train_new1.Sw_invSb[start_idx:])[i]))


fig, ax = plt.subplots(1)
p = ax.scatter(np.array(train_base.cos_M[start_idx:end_idx]), np.array(train_base.Sw_invSb[start_idx:end_idx]), label='Baseline',
               c=np.array(test_base.accuracy[start_idx:end_idx]), cmap= 'viridis', vmin=vmin, vmax=vmax)
plt.xlabel('NC2-2: Equiangular')
plt.ylabel('NC1')
ax.set_yscale("log")
plt.xlim(0.0, 0.1)
plt.title('NC1 vs. NC2(Equiangular) and Test Acc for Baseline')
fig.colorbar(p)


plt.scatter(train_base.Sw_invSb, train_base.cos_M, label='Baseline',
            s=np.arange(101) )
plt.scatter(train_new1.Sw_invSb, train_new1.cos_M, label='Label Smoothing',
            s=( np.array([max(0.6, e) for e in test_new1.accuracy]) - 0.6) *20 )
plt.xlabel('NC1')
plt.ylabel('NC2-2: Equiangular')
plt.legend()
plt.title('NC1 vs. NC2(Equiangular)')



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




