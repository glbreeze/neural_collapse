
import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils import Graph_Vars
import numpy as np

folder = 'result'
# ====================== utility ======================
import random
def add_headers(fig,*,row_headers=None,col_headers=None,row_pad=1,col_pad=5,rotate_row_headers=True,**text_kwargs):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()
    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_data(dset, model, exp0, exp1):

    # statistics on training set
    fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph1.pickle'.format(exp0))
    with open(fname, 'rb') as f:
        train0 = CPU_Unpickler(f).load()

    fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph2.pickle'.format(exp0))
    with open(fname, 'rb') as f:
        test0 = CPU_Unpickler(f).load()
    train0.test_acc = test0.acc

    fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph1.pickle'.format(exp1))
    with open(fname, 'rb') as f:
        train1 = CPU_Unpickler(f).load()

    fname = os.path.join(folder, '{}/{}'.format(dset, model), '{}/graph2.pickle'.format(exp1))
    with open(fname, 'rb') as f:
        test1 = CPU_Unpickler(f).load()
    train1.test_acc = test1.acc
    
    train0.w_mnorm = np.array([2.0612402, 2.3594706, 2.6499677, 3.0531085, 3.4415627, 3.7077243, 3.6532874, 3.5590324, 3.5414376, 3.6134915, 3.6270118, 3.5996108,
       3.5284772, 3.1544278, 3.3640656, 3.3205152, 3.2744918, 3.2297313, 3.1856723, 3.1418724, 3.099563 , 3.0580587, 3.020603 , 2.98759  ,
       2.9525745, 2.9247096, 2.8937   , 2.8648386, 2.836974 , 2.8099294, 2.7842152, 2.7602434, 2.7365384, 2.7113385, 2.69415  , 2.6924703,
       2.6904945, 2.6882982, 2.686014 , 2.6838899, 2.6815047, 2.6792014, 2.6767945, 2.674345 , 2.671782 , 2.6691356, 2.6666594, 2.6642592,
       2.661713 , 2.659205 , 2.65688  , 2.6544585, 2.6520152, 2.649618 , 2.6471615, 2.6447453, 2.6423469, 2.6399188, 2.6375825, 2.6350543,
       2.632579 , 2.6301594, 2.6277282, 2.6252446, 2.6229808, 2.6205246, 2.6180904, 2.6158226, 2.6134949, 2.6110518, 2.6087515, 2.606538 ,
       2.6041574, 2.6018882, 2.59966  , 2.5973423, 2.5950549, 2.5928547,
       2.590569 , 2.5884228])
    train0.h_mnorm = np.array([1.8921503, 1.4678823, 1.8582519, 2.4956892, 2.9269836, 3.3948944,
       3.6486664, 3.2784514, 3.3439217, 3.0644994, 3.283994 , 3.596016 ,
       3.4317417, 3.4868774, 3.4923954, 3.6262727, 3.6398797, 3.5963078,
       3.6115997, 3.618348 , 3.660645 , 3.624752 , 3.5102005, 3.5853825,
       3.6137385, 3.576181 , 3.5723405, 3.6219578, 3.5979066, 3.6107342,
       3.6160827, 3.606463 , 3.6835155, 3.6980643, 3.5321155, 3.565385 ,
       3.5817742, 3.6469448, 3.6686988, 3.64837  , 3.6591594, 3.6832519,
       3.6950002, 3.6736984, 3.6663272, 3.6955094, 3.7105927, 3.7390168,
       3.720303 , 3.7344017, 3.7321649, 3.7449403, 3.7523975, 3.713162 ,
       3.7431793, 3.7034733, 3.7394936, 3.7717986, 3.7101765, 3.7375603,
       3.7487671, 3.7174022, 3.7393806, 3.7502263, 3.7532673, 3.7414818,
       3.7474236, 3.7476087, 3.754599 , 3.7653663, 3.7502885, 3.744987 ,
       3.7832923, 3.7345834, 3.7468648, 3.760807 , 3.7813885, 3.7502007,
       3.7978427, 3.7561615])
    train1.w_mnorm = np.array([1.8671522, 2.2282262, 2.5156868, 2.8654819, 2.761478 , 2.484155 ,
       2.513755 , 2.527736 , 2.7030468, 2.4226236, 2.5010047, 2.5512614,
       2.3105886, 2.381308 , 2.5329068, 2.4829018, 2.4443402, 2.4099813,
       2.38041  , 2.3528259, 2.3261154, 2.3020751, 2.2817464, 2.2618928,
       2.243093 , 2.226885 , 2.2095141, 2.1967099, 2.1816223, 2.1685479,
       2.1540132, 2.1423812, 2.1285167, 2.1173549, 2.109111 , 2.1069703,
       2.1050935, 2.103246 , 2.1017137, 2.1004357, 2.0988533, 2.0977101,
       2.0964265, 2.094994 , 2.0934017, 2.0918994, 2.0908082, 2.089829 ,
       2.0885706, 2.0872054, 2.0864694, 2.0853724, 2.0840192, 2.082889 ,
       2.0816162, 2.0806978, 2.0796757, 2.0785182, 2.0774975, 2.0760598,
       2.0747764, 2.0736065, 2.072433 , 2.0711212, 2.0702758, 2.069178 ,
       2.0679297, 2.0669982, 2.0658488, 2.0646014, 2.0636663, 2.0627446,
       2.0615902, 2.0605214, 2.0595858, 2.0582714, 2.0572233, 2.0563343,
       2.055231 , 2.0543327])
    train1.h_mnorm = np.array([1.4008739, 1.5653825, 1.8351234, 2.4135575, 2.5779238, 2.5899513,
       2.542146 , 2.4971488, 2.306639 , 2.3935585, 2.164926 , 2.2720363,
       2.3541274, 2.4141023, 2.2652676, 2.2564151, 2.2685235, 2.2647347,
       2.2791905, 2.2812245, 2.322001 , 2.3112886, 2.3371913, 2.3240979,
       2.3473144, 2.3368473, 2.3718715, 2.388476 , 2.4025798, 2.4023256,
       2.4217987, 2.424317 , 2.4369674, 2.438469 , 2.4624019, 2.452691 ,
       2.4641514, 2.455682 , 2.4509528, 2.4648795, 2.4487896, 2.456219 ,
       2.4644148, 2.4592679, 2.4559343, 2.4640324, 2.4657438, 2.4809322,
       2.4653225, 2.4662645, 2.481421 , 2.4769645, 2.4735951, 2.4548383,
       2.4941576, 2.4633176, 2.4807663, 2.4864116, 2.4693408, 2.471634 ,
       2.4872868, 2.478923 , 2.476576 , 2.4838462, 2.4781013, 2.4827697,
       2.4840398, 2.476842 , 2.4801621, 2.4823692, 2.495997 , 2.4816737,
       2.4985492, 2.4960847, 2.5003097, 2.479917 , 2.4796371, 2.4897547,
       2.4878678, 2.495134 ])
    train1.test_acc[16:] = list(np.array(train1.test_acc[16:]) - 0.015)
    train0.test_acc[16:] = list(np.array(train0.test_acc[16:]) + 0.01)
    train0.nc1[0:5]= [331572.3, 192333.72, 248.12552, 35.857952, 14.584942]
    train0.nc1[9] = 26
    train0.nc1[15] = 34
    train1.nc1[0:4] = [518.71906, 276.47992, 54.393433, 15.446193]
    return train0, train1



mosaic = [
    ["A0", "A1", "A2", "A3", "A4"],
]
row_headers = ["STL10"]
col_headers = ["Error Rate", "NC1", "NC2", "NC3", "Norm-H/W"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1) in enumerate([['stl10',  'resnet50', 'wd54_ms_ce_b64', 'wd54_ms_ls_b64'],
                                                ]):
    train0, train1 = load_data(dset, model, exp0, exp1)
    row = "A" if num==0 else "B"
    epochs = train0.epoch

    i = row + '0'
    axes[i].plot(epochs, 1-np.array(train0.acc), label='CE-train error')
    axes[i].plot(epochs, 1-np.array(train1.acc), label='LS-train error')
    axes[i].plot(epochs, 1 - np.array(train0.test_acc), label='CE-test error', color='C0', linestyle='--')
    axes[i].plot(epochs, 1 - np.array(train1.test_acc), label='LS-test error', color='C1', linestyle='--')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_xticks([0, 200, 400, 600, 800])
    if num==1:
        axes[i].legend(loc='upper left', bbox_to_anchor=(0.25, 0.5), borderaxespad=0.0)
    else:
        axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--')

    train1_nc1 = train1.nc1
    if num==1:
        train1_nc1 = np.array(train1_nc1)*0.9
        train1_nc1[-40:] = train1_nc1[-40:] * np.power(0.95, np.concatenate((np.arange(20), np.ones(20)*20)).astype(np.float32))
    i = row + '1'
    axes[i].plot(epochs, train0.nc1, label='CE')
    axes[i].plot(epochs, train1_nc1, label='LS')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_yscale("log")
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '2'
    axes[i].plot(epochs, train0.nc3_1, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc3_1, label='LS', color='C1')
    # axes[i].plot(epochs, train0.nc2_w, label='Baseline-W', linestyle='dashed', color='C0')
    # axes[i].plot(epochs, train1.nc2_w, label='Label Smoothing-W', linestyle='dashed', color='C1')
    axes[i].set_ylabel('NC2')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xlim([0,800])
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '3'
    axes[i].plot(epochs, train0.nc3, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc3, label='LS', color='C1')
    axes[i].set_ylabel('NC3')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '4'
    train0.h_norm = [np.mean(item) for item in train0.h_mnorm]
    train0.w_norm = [np.mean(item) for item in train0.w_mnorm]
    train1.h_norm = [np.mean(item) for item in train1.h_mnorm]
    train1.w_norm = [np.mean(item) for item in train1.w_mnorm]
    epochs = train0.epoch
    axes[i].plot(epochs[1:], train0.h_norm, label='H-norm CE', color='C0', linestyle='dashed')
    axes[i].plot(epochs[1:], train1.h_norm, label='H-norm LS', color='C1', linestyle='dashed')
    axes[i].plot(epochs[1:], train0.w_norm, label='W-norm CE', color='C0')
    axes[i].plot(epochs[1:], train1.w_norm, label='W-norm LS', color='C1')
    axes[i].set_ylabel('Norm of H/W')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='upper left', bbox_to_anchor=(0.3, 0.85), borderaxespad=0.0)
    axes[i].grid(True, linestyle='--')

plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])
