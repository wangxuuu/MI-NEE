import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from data.mix_gaussian import MixedGaussian
from model.mine import MINE 
from scipy.stats import bernoulli

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

    sample_size = 1000   # sample size
    p = 0.6             # parameter of bernoulli distribution
    mu = 1              # mean of Gaussian distribution
    rep = 1             # number of repeated runs
    d = 1               # number of dimensions for X (and Y)
    X = np.zeros((rep,sample_size,d+1))
    Y = np.zeros((rep,sample_size,d))
    Z = np.zeros((rep,sample_size,d+1))

    for i in range(rep):
        X_ = bernoulli.rvs(p, size=sample_size)
    num_0 = len(X_[X_==0])
    num_1 = len(X_[X_==1])

    rho_1 = 0.9
    rho_2 = 0.9
    d = 1
    mg1 = MixedGaussian(sample_size=num_0,rho1=rho_1,rho2=-rho_1)
    mg2 = MixedGaussian(sample_size=num_1,rho1=rho_2,rho2=-rho_2)

    for i in range(rep):
        for j in range(d):
            data1 = mg1.data
            data2 = mg2.data
            X[i,:num_0,j] = data1[:,0]
            X[i,num_0:,j] = data2[:,0]
            Y[i,:num_0,j] = data1[:,1]
            Y[i,num_0:,j] = data2[:,1]
        X[i,:num_0,d] = X_[X_==0]
        X[i,num_0:,d] = X_[X_==1]

    Hxy1 = mg1.ground_truth * d
    Hxy2 = mg2.ground_truth * d

    mi = (1-p)*Hxy1 + p*Hxy2

    data1 = mg1.data
    data2 = mg2.data
    MI1 = np.average([mg1.I(X[0],X[1]) for X in data1])
    MI2 = np.average([mg2.I(X[0],X[1]) for X in data2])

    MI = (1-p)*MI1 + p*MI2

    diff.append((MI-mi)**2)
    MIs.append(MI)
    
    batch_size = 100       # batch size of data sample
    lr = 1e-4              # learning rate
    ma_rate = 0.1          # rate of moving average in the gradient estimate 

    mine_list = []
    for i in range(rep):
        mine_list.append(MINE(torch.Tensor(X[i]),torch.Tensor(Y[i]),
                            batch_size=batch_size,lr=lr,ma_rate=ma_rate))
    dXY_list = np.zeros((rep,0))

    for k in tqdm(range(80000)):
        dXY_list = np.append(dXY_list, np.zeros((rep, 1)), axis=1)
        for i in range(rep):
            mine_list[i].step()
            dXY_list[i,-1] = mine_list[i].forward()
    mi_ma_rate = 0.01            # rate of moving average
    mi_list = dXY_list.copy()    # see also the estimate() member function of MINE
    for i in range(1,dXY_list.shape[1]):
        mi_list[:,i] = (1-mi_ma_rate) * mi_list[:,i-1] + mi_ma_rate * mi_list[:,i]
    MI_list.append(mi_list)

# var = np.average(diff)
var = np.std(MIs)
plt.axhline(mi,label='ground truth',linestyle='--',color='red')
plt.axhline(mi+2*var,label='gt+2$\delta$',linestyle='--',color='blue')
plt.axhline(mi-2*var,label='gt-2$\delta$',linestyle='--',color='blue')
for k in range(len(MI_list)):
    mi_list = MI_list[k]
    for i in range(rep):
        plt.plot(mi_list[i,:], label=k)
        
plt.xlim((0,80000))
plt.ylim((0,mi*1.3))
plt.title('MINE')
plt.xlabel("number of iterations")
plt.ylabel("MI estimate")
plt.legend()
plt.show()
plt.savefig('./results/disc_mine_test.png')