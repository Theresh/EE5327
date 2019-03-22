import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv(r'''C:\ML ASSIGNMENT\Xsvm1.csv''');
df2 = pd.read_csv(r'''C:\ML ASSIGNMENT\ysvm1.csv''');
x1= list(df1.x1) #conversion of x1into list
x2=list(df1.x2)  #conversion of x2 into list
X1=list(zip(x1,x2))
X=np.matrix(X1)
Y1=list(df2.y)
Y2=np.matrix(Y1)
Y=Y2.T

for i in range(500):
    if(Y[i]<0):
        Y[i]=0

number_samples=500

L=2###number of input nodes of MlP
M=2 ###number ofhidden nodes of MlP
K=1##number of output nodes of MlP
alpha=np.random.rand(M,L)
alpha_bias=np.random.rand(M,1)
beta_weights=np.random.rand(K,M)
beta_bias=np.random.rand(K,1)
Y_hat=np.zeros((500,1))
li=np.zeros((1,L))
alpha_grad=np.zeros((M,L,number_samples))
sigma_beta_grad=np.zeros((K,M,number_samples))
g_dash=np.zeros((number_samples,1))
ent_dash=np.zeros((number_samples,1))
error=np.zeros((number_samples,1))
epsilon=0.2
delta1=0.000002
rho1=0.9
rho2=0.99



def sigmoid(x):
        sh=x.shape
        for i in range(sh[0]):
            for j in range(sh[1]):
                x[i][j]=1/(1+np.exp(- x[i][j]))
        return x 
def sigmoid_differentiation(x):
    y=np.zeros((x.shape))
    y=np.multiply(x,(1-x))
    return y
def update_of_beta(X12,beta_weights,r_beta,s_beta,t_beta): ### gamma is learning rate
    #Y=np.zeros((K,M))
    Y21=(np.sum(X12,axis=2))/number_samples
    Z21=np.multiply(Y21,Y21)
    s_beta1=rho1*s_beta+(1-rho1)*Y21
    r_beta1=(rho2*r_beta)+((1-rho2)*Z21)
    #d1=math.pow((1-rho1),t_beta)
    #d2=math.pow((1-rho2),t_beta)
    d1=1-math.pow(rho1,t_beta)
    d2=1-math.pow(rho2,t_beta)
    s_cap_beta=np.divide(s_beta1,d1)
    r_cap_beta=np.divide(r_beta1,d2)
    r_beta2=np.sqrt(r_cap_beta)+delta1
    #r_beta2=np.sqrt(r_beta21)
    del_beta=np.divide(s_cap_beta,r_beta2)
    del_beta2=np.multiply(del_beta,epsilon)
    t_beta=t_beta+1
    beta_weights1=beta_weights-del_beta2
    return beta_weights1,r_beta1,s_beta1,t_beta
def update_of_alpha(x12,alpha_weights,r_alpha,s_alpha,t_alpha):
   
    #Y12=np.zeros((M,L))
    Y12=(np.sum(x12,axis=2))/number_samples
    Z12=np.multiply(Y12,Y12)
    #print('alpha')
    s_alpha1=rho1*s_alpha+(1-rho1)*Y12
    r_alpha1=rho2*r_alpha+(1-rho2)*Z12
    #d1=math.pow((1-rho1),t_alpha)
    #d2=math.pow((1-rho2),t_alpha)
    d1=1-math.pow(rho1,t_alpha)
    d2=1-math.pow(rho2,t_alpha)
    s_cap_alpha=np.divide(s_alpha1,d1)
    r_cap_alpha=np.divide(r_alpha1,d2)
    #print(r_alpha1)
    r_alpha2=np.sqrt(r_cap_alpha)+delta1
    del_alpha=np.divide(s_cap_alpha,r_alpha2)
    del_alpha2=np.multiply(del_alpha,epsilon)
    t_alpha=t_alpha+1
    
    alpha_weights1=alpha_weights-del_alpha2
    
    return alpha_weights1,r_alpha1,s_alpha1,t_alpha
def MLP(input_NN,M,K,L,alpha,alpha_bias,beta_weights,beta_bias):
        
        #alpha_weights=alpha
        
        input1=input_NN.T
        
        Z1=np.matmul(alpha,input1)+alpha_bias
        Z=sigmoid(Z1)
        #print(Z)
        # beta_weights=beta_weights1
        #beta_bias=beta_bias1
        H1=(np.matmul(beta_weights,Z)+beta_bias)
        #print(H1 )
        Y11=sigmoid(H1)
       # print(Y)
           
#           
        return(input1,Y11,Z,H1)
def backpropagation(t11,t22,s11,s22,r11,r22,X,M,K,L,alpha,alpha_bias,beta_weights,beta_bias):
    #print(Y)
    
    for i in range(number_samples):
         IN,ZX,Z,H1= MLP(X[i],M,K,L,alpha,alpha_bias,beta_weights,beta_bias)
         Y_hat[i]=np.transpose(ZX)
         #print(Y_hat[i])
         g_dash[i]=sigmoid_differentiation(ZX)
         ent_dash[i]=-2*(Y[i]-Y_hat[i])
         error[i]=(Y[i]-Y_hat[i])**2
         #print(error[i])
         delta=np.multiply(g_dash[i],ent_dash[i])
         #print(delta)
         beta_grad=np.multiply(delta,Z.T)
         sigma_beta_grad[:,:,i]=beta_grad
         
         li1=delta*beta_weights
         li=li1.reshape(M,1)
         #sm_1=np.sum(li,axis=0) 
         sigma_dash=sigmoid_differentiation(Z)
         a_g=np.multiply(sigma_dash,li)
         alpha_grad[:,:,i]=np.matmul(a_g,IN.T)
         
    #print(alpha_grad.shape)
    alpha_updated,r11,s11,t11=update_of_alpha(alpha_grad,alpha,r11,s1,t11)
    beta_updated,r22,s22,t22=update_of_beta(sigma_beta_grad,beta_weights,r22,s22,t22)
   # print(alpha_updated.shape)
    #alpha=alpha_updated
    #beta_weights=beta_updated
    train_error=np.sum(error)
    print('train_error after each epoch')
    print(train_error)
    
    return alpha_updated,beta_updated,train_error,r11,r22,s11,s22,t11,t22
#MLP()
epoch=0
t1=1
t2=1
r1=np.zeros((M,L))
r2=np.zeros((K,M))
s1=np.zeros((M,L))
s2=np.zeros((K,M))
alpha_updated,beta_updated,train_error,r11,r22,s11,s22,t11,t22= backpropagation(t1,t2,s1,s2,r1,r2,X,M,K,L,alpha,alpha_bias,beta_weights,beta_bias)   
#print(alpha_updated)
TRAIN_ERROR=[]

TRAIN_ERROR.append(train_error)

while(train_error>2):
   # print('swetha')
    alpha=alpha_updated
    beta_weights=beta_updated
    alpha_updated,beta_updated,train_error,r11,r22,s11,s22,t11,t22= backpropagation(t11,t22,s11,s22,r11,r22,X,M,K,L,alpha,alpha_bias,beta_weights,beta_bias)
    TRAIN_ERROR.append(train_error)
    
    epoch=epoch+1
plt.subplot(121)
plt.plot(TRAIN_ERROR,label='train error')
plt.xlabel('epochs')
plt.ylabel('train error')
#plt.subplot(122)
#plt.legend()


#plt.tight.layout()
#plt.legend
#plt.show()
#print(total_error)
#print(alpha_updated)
#print(beta_updated)
     

