import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from data.mix_gaussian import MixedGaussian
from model.minee import MINEE

random_seeds = np.arange(10)
MI_estimate = []
MI_list = []
MIs = []
diff = []
rep = 1
# use GPU if available
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

for seed in tqdm(random_seeds):
    np.random.seed(seed)
    # initialize random seed
    torch.manual_seed(seed)

    sample_size = 400  # sample size
    rho = 0.9  # model parameter

    rep = 1  # number of repeated runs
    d = 1  # number of dimensions for X (and Y)

    X = np.zeros((rep, sample_size, d))
    Y = np.zeros((rep, sample_size, d))
    mg = MixedGaussian(sample_size=sample_size, rho1=rho, rho2=-rho)
    mi = mg.ground_truth * d

    data = mg.data
    MI = np.average([mg.I(X[0], X[1]) for X in data])
    diff.append((MI - mi) ** 2)
    MIs.append(MI)

    for i in range(rep):
        for j in range(d):
            data = mg.data
            X[i, :, j] = data[:, 0]
            Y[i, :, j] = data[:, 1]

    batch_size = 100  # batch size of data sample
    ref_batch_factor = 10
    lr = 1e-4  # learning rate
    ma_rate = 0.1  # rate of moving average in the gradient estimate

    minee_list = []
    for i in range(rep):
        minee_list.append(MINEE(torch.Tensor(X[i]), torch.Tensor(Y[i]),
                                batch_size=batch_size, ref_batch_factor=ref_batch_factor, lr=lr))

    dXY_list = np.zeros((rep, 0))
    dX_list = np.zeros((rep, 0))
    dY_list = np.zeros((rep, 0))

    for k in tqdm(range(100000)):
        dXY_list = np.append(dXY_list, np.zeros((rep, 1)), axis=1)
        dX_list = np.append(dX_list, np.zeros((rep, 1)), axis=1)
        dY_list = np.append(dY_list, np.zeros((rep, 1)), axis=1)
        for i in range(rep):
            minee_list[i].step()
            dXY_list[i, -1], dX_list[i, -1], dY_list[i, -1] = minee_list[i].forward()
    mi_ma_rate = 0.01  # rate of moving average
    mi_list = (dXY_list - dX_list - dY_list).copy()
    for i in range(1, dXY_list.shape[1]):
        mi_list[:, i] = (1 - mi_ma_rate) * mi_list[:, i - 1] + mi_ma_rate * mi_list[:, i]
    MI_list.append(mi_list)

# var = np.average(diff)
var = np.std(MIs)
plt.axhline(mi, label='ground truth', linestyle='--', color='red')
plt.axhline(mi + 2 * var, label='gt+2delta', linestyle='--', color='blue')
plt.axhline(mi - 2 * var, label='gt-2delta', linestyle='--', color='blue')
for k in range(len(MI_list)):
    mi_list = MI_list[k]
    for i in range(rep):
        plt.plot(mi_list[i, :], label=k)

plt.xlim((0, 100000))
plt.ylim((0, mi * 1.3))
plt.title('MINEE')
plt.xlabel("number of iterations")
plt.ylabel("MI estimate")
plt.legend()
plt.show()