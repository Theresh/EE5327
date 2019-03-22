import numpy as np

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

number_samples=400

L=2###number of input nodes of MlP
M=2 ###number ofhidden nodes of MlP
K=1##number of output nodes of MlP
alpha=np.random.rand(M,L)
alpha_bias=np.random.rand(M,1)
beta_weights=np.random.rand(K,M)
beta_bias=np.random.rand(K,1)

error1=np.zeros((number_samples,1))


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
def update_of_beta(X12,gamma,beta_weights): ### gamma is learning rate
    #Y=np.zeros((K,M))
    
    
    beta_weights1=beta_weights-gamma*X12
    
    return beta_weights1
def update_of_alpha(x12,gamma,alpha_weights):
   
    
   
    
    alpha_weights1=alpha_weights-gamma*x12
    
    return alpha_weights1
def MLP(input_NN,M,K,L,alpha1,alpha_bias,beta_weights1,beta_bias):
        
        #alpha_weights=alpha
        
        input1=input_NN.T
        
        Z1=np.matmul(alpha1,input1)+alpha_bias
        Z=sigmoid(Z1)
        #print(Z)
        # beta_weights=beta_weights1
        #beta_bias=beta_bias1
        H1=(np.matmul(beta_weights1,Z)+beta_bias)
        #print(H1 )
        Y11=sigmoid(H1)
       # print(Y)
           
#           
        return(input1,Y11,Z,H1)
def backpropagation(X_INP,Y_OT,M,K,L,alpha1,alpha_bias,beta_weights1,beta_bias):

#print(Y)
     Xi=X_INP
     IN,ZX,Z,H1= MLP(Xi,M,K,L,alpha1,alpha_bias,beta_weights1,beta_bias)
     Y_hat=ZX
     #print(Y_hat[i])
     g_dash=sigmoid_differentiation(ZX)
     ent_dash=-2*(Y_OT-Y_hat)
     
     error=pow((Y_OT-Y_hat),2)
     #print(error[i])
     delta=np.multiply(g_dash,ent_dash)
     #print(delta)
     beta_grad=np.multiply(delta,Z.T)
     
     beta_updated=update_of_beta(beta_grad,0.1,beta_weights1)
     li1=delta*beta_weights1
     li=li1.reshape(M,1)
     #sm_1=np.sum(li,axis=0) 
     sigma_dash=sigmoid_differentiation(Z)
     
     a_g=np.multiply(sigma_dash,li)
     alpha_grad=np.matmul(a_g,Xi)
     
     alpha_updated=update_of_alpha(alpha_grad,0.1,alpha1)

     return alpha_updated,beta_updated,error
#MLP()
epoch=0
alpha_updated,beta_updated,error1[0]= backpropagation(X[0],Y[0],M,K,L,alpha,alpha_bias,beta_weights,beta_bias)   
#print(alpha_updated)
TRAIN_ERROR=[]
train_error=np.sum(error1)

TRAIN_ERROR.append(train_error)

while(train_error>0.05):
   # print('swetha')
   for i in range(number_samples):
        #alpha=alpha_updated
        #beta_weights=beta_updated
        alpha_updated,beta_updated,error1[i]= backpropagation(X[i],Y[i],M,K,L,alpha_updated,alpha_bias,beta_updated,beta_bias)
   train_error=np.sum(error1)
   print('error after each epoch')
   print(train_error)
   TRAIN_ERROR.append(train_error)
    
    #epoch=epoch+1
plt.subplot(121)
plt.plot(TRAIN_ERROR,label='train error')
plt.xlabel('epochs')
plt.ylabel('train error')
plt.show()
