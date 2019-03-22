import numpy as np
#import improved_adagrad
import matplotlib.pyplot as plt
from ada_grad1 import adagrad1
from Ada_Grad import ad_grad
L=2###number of input nodes of MlP
M=2 ###number ofhidden nodes of MlP
K=1##number of output nodes of MlP
alpha=np.random.rand(M,L)
alpha_bias=np.random.rand(M,1)
beta_weights=np.random.rand(K,M)
beta_bias=np.random.rand(K,1)
train_error1=adagrad1(alpha, beta_weights,alpha_bias,beta_bias)
train_error2=ad_grad(alpha, beta_weights,alpha_bias,beta_bias)
plt.subplot(121)
plt.plot(train_error1,label='train error')
plt.xlabel('epochs')
plt.ylabel('train error1')
plt.subplot(122)
plt.plot(train_error2,label='train error')
plt.xlabel('epochs')
plt.ylabel('train error2')