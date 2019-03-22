
# coding: utf-8

# In[1]:

import numpy as np
import random as rd
import math as math
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

"""
Load the MNIST dataset into numpy arrays
Author: Alexandre Drouin
License: BSD
"""
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST data/", one_hot=True)
X_train = np.vstack([img.reshape((28, 28)) for img in mnist.train.images])
y_train = mnist.train.labels
X_test = np.vstack([img.reshape(28, 28) for img in mnist.test.images])
y_test = mnist.test.labels
del mnist


# In[4]:

print('shape of X train=',X_train.shape)
print('shape of y train=',y_train.shape)
print('shape of X test=',X_test.shape)
print('shape of y test=',X_test.shape)


# In[5]:

##################################################################
# reshaping numpy array into volume  for train and test set 
#################################################################
X_train=np.reshape(X_train,(-1,784))
y_train=np.reshape(y_train,(-1,10))
X_test=np.reshape(X_test,(-1,784))
y_test=np.reshape(y_test,(-1,10))
print('shape of train input volume=',X_train.shape)
print('shape of train labels=',y_train.shape)
print('shape of test input volume=',X_test.shape)
print('shape of test labels=',y_test.shape)


# In[ ]:

#############################################################
# subsample of dataset
###########################################################
X_train=X_train[0:500,:]
y_train=y_train[0:500,:]
X_test=X_test[0:100,:]
y_test=y_test[0:100,:]
print('shape of train input volume=',X_train.shape)
print('shape of train labels=',y_train.shape)
print('shape of test input volume=',X_test.shape)
print('shape of test labels=',y_test.shape)


# In[6]:

class ThreeLayerNet(object):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        self.parameters={}
        self.parameters['W1']=1e-4*np.random.rand(input_size,hidden_size1)
        self.parameters['W2']=1e-4*np.random.rand(hidden_size1,hidden_size2)
        self.parameters['W3']=1e-4*np.random.rand(hidden_size2,output_size)
        self.parameters['b1']=np.zeros(hidden_size1)
        self.parameters['b2']=np.zeros(hidden_size2)
        self.parameters['b3']=np.zeros(output_size)
        
    def forward_pass(self, X, y=None, reg=0.0):
        # Unpack variables from the parameters dictionary
        W1, b1 = self.parameters['W1'], self.parameters['b1']
        W2, b2 = self.parameters['W2'], self.parameters['b2']
        W3, b3 = self.parameters['W3'], self.parameters['b3']
        N, D = X.shape
        
        relu=lambda x: np.maximum(0,x)
        h1=relu(np.dot(X,W1)+b1)
        h2=relu(np.dot(h1,W2)+b2)
        h3=np.dot(h2,W3)+b3
        scores=h3
        return h1,h2,scores
        
    def losses(self, X, y=None, reg=0.0):
        h1,h2,scores=self.forward_pass( X, y=None, reg=0.0)
        h3=scores
        # Unpack variables from the parameters dictionary
        W1 = self.parameters['W1']
        W2 = self.parameters['W2']
        W3 = self.parameters['W3']
        # if labels are not given(i.e. test case)
        if y is None:
            return scores
        ##############################################################
        # loss calculation
        ############################################################
        N, D = X.shape
        loss = None
        scores -=np.reshape(np.max(scores,axis=1),(N,1))
        p = np.exp(scores)/np.reshape(np.sum(np.exp(scores),axis=1),(N,1)) 
        index_correct_class=[range(N),np.argmax(y,axis=1)] #indexes of correct class 
        correct_class_score= np.reshape(p[index_correct_class[0],index_correct_class[1]],(N,1))
        loss = np.sum(-np.log(correct_class_score))
        loss /= N
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        
        ###############################################################
        # backward pass
        ##############################################################
        grads = {}
        # gradient on scores
        dscores=p
        dscores[index_correct_class] -=1
        dscores/=N

        #gradient on W3
        grads['W3']=np.dot(h2.T,dscores)/N
        #gradient on b3
        grads['b3']=np.sum(dscores,axis=0)/N
        

        #backprop to hidden layer2
        dhidden2= np.dot(dscores, W3.T)
        dhidden2[h2 <=0] =0
        
        #gradient on W2
        grads['W2']=np.dot(h1.T,dhidden2)/N
        #gradient on b2
        grads['b2']=np.sum(dhidden2,axis=0)/N
        
        #backprop to hidden layer2
        dhidden1= np.dot(dhidden2, W2.T)
        dhidden1[h1 <=0] =0

        #gradient on W2
        grads['W1']= np.dot(X.T,dhidden1)/N
        #gradient on b2
        grads['b1']=np.sum(dhidden1,axis=0)/N

        #adding regularization
        grads['W3']+=reg*W3
        grads['W2']+=reg*W2
        grads['W1']+=reg*W1
        
        
        return loss, grads, h3
    ###########################################################################################################
    # stochastic gradient descent
    ###########################################################################################################
    def SGD_train(self, X, y=None,iters=10,learning_rate=1e-4,reg=0.25):
        N,D=X.shape
        
        loss_his=[]# store training loss values
        train_acc_his=[] # store training accuracy
        for j in range(iters):
            A=0
            for i in range(N):
                q=np.reshape(X[i],(1,D))
                u=np.reshape(y[i],(1,10))
                loss,grads,h3=self.losses( X=q, y=u, reg=0.25)# loss and gradient calculation
                A=A+loss
                # updating parameters
                for k in range(1,4):
                    self.parameters["W"+str(k)]=self.SGD(self.parameters["W"+str(k)],grads["W"+str(k)],learning_rate)
                    self.parameters["b"+str(k)]=self.SGD(self.parameters["b"+str(k)],grads["b"+str(k)],learning_rate) 
            if j%10==0:  
                h3=self.losses( X, y=None, reg=0.25)
                train_acc = (np.argmax(h3,1) == np.argmax(y,1)).mean()
                print('iteration %d / %d: loss %f training accuracy: %f' % (j,iters, A/N,train_acc))
                loss_his=np.append(loss_his,A/N)
                train_acc_his=np.append(train_acc_his,train_acc)
            
        return loss_his/N,train_acc_his
    
    def SGD(self,W,dW,learning_rate):# stochastic gradient descent method for upadating parameters
        
        W = W - learning_rate*dW
        return W 
    ###########################################################################################################
    # momentum
    ###########################################################################################################
    
    def momentum_train(self, X, y=None,iters=10,momentum=0.9,learning_rate=1e-4,reg=0.25):
        N,D=X.shape
        loss_his=[]# store training loss values
        train_acc_his=[] # store training accuracy
        g=[]
        
        v={}# velocity
        # initializing velocity
        for k in range(3):
            v["W"+str(k+1)] = np.zeros((self.parameters["W"+str(k+1)].shape[0],self.parameters["W"+str(k+1)].shape[1]))
            v["b"+str(k+1)] = np.zeros(self.parameters["b"+str(k+1)].shape[0])
            
        for j in range(iters):
            loss,grads,h3=self.losses( X, y, reg=0.25)# loss and gradient calculation
            g=np.append(g,loss)
            
            # updating parameters
            for k in range(3):
                self.parameters["W"+str(k+1)],v["W"+str(k+1)]=self.momentum(self.parameters["W"+str(k+1)],grads["W"+str(k+1)],learning_rate,momentum,v["W"+str(k+1)])
                self.parameters["b"+str(k+1)],v["b"+str(k+1)]=self.momentum(self.parameters["b"+str(k+1)],grads["b"+str(k+1)],learning_rate,momentum,v["b"+str(k+1)])
            
            if j%10==0:
                train_acc = (np.argmax(h3,1) == np.argmax(y,1)).mean()
                print('iteration %d / %d: loss %f training accuracy: %f' % (j,iters, loss,train_acc))
                loss_his=np.append(loss_his,loss)
                train_acc_his=np.append(train_acc_his,train_acc)
        return loss_his,train_acc_his,g
    
    def momentum(self,W,dW,learning_rate,momentum,velocity):# MOMENTUM method for upadating parameters
        velocity= momentum*velocity - learning_rate*dW
        next_w = W + velocity
        return next_w,velocity
    
    ################################################################################################
    # nesterov
    ###############################################################################################
    def nesterov_train(self, X, y=None,iters=10,momentum=0.9,learning_rate=1e-4,reg=0.25):
        N,D=X.shape
        loss_his=[]# store training loss values
        train_acc_his=[] # store training accuracy
        g=[]
        
        v={}# velocity
        # initializing velocity
        for k in range(3):
            v["W"+str(k+1)] = np.zeros((self.parameters["W"+str(k+1)].shape[0],self.parameters["W"+str(k+1)].shape[1]))
            v["b"+str(k+1)] = np.zeros(self.parameters["b"+str(k+1)].shape[0])
            
        for j in range(iters): 
            # interim update of parameters
            for k in range(3):
                self.parameters["W"+str(k+1)]= self.parameters["W"+str(k+1)] + momentum*v["W"+str(k+1)]
                self.parameters["b"+str(k+1)]= self.parameters["b"+str(k+1)] + momentum*v["b"+str(k+1)]

            loss,grads,h3=self.losses( X, y, reg=0.25)# loss and gradient calculation
            g=np.append(g,loss)
            
            # updating parameters and velocity
            for k in range(3):
                self.parameters["W"+str(k+1)],v["W"+str(k+1)]=self.momentum(self.parameters["W"+str(k+1)],grads["W"+str(k+1)],learning_rate,momentum,v["W"+str(k+1)])
                self.parameters["b"+str(k+1)],v["b"+str(k+1)]=self.momentum(self.parameters["b"+str(k+1)],grads["b"+str(k+1)],learning_rate,momentum,v["b"+str(k+1)])
            
            if j%10==0:
                train_acc = (np.argmax(h3,1) == np.argmax(y,1)).mean()
                print('iteration %d / %d: loss %f training accuracy: %f' % (j,iters, loss,train_acc))
                loss_his=np.append(loss_his,loss)
                train_acc_his=np.append(train_acc_his,train_acc)
        return loss_his,train_acc_his,g
    
    ################################################################################################
    # AdaGrad
    ###############################################################################################
    
    def adagrad_train(self, X, y=None,iters=10,momentum=0.9,learning_rate=1e-4,reg=0.25,delta=1e-7):
        N,D=X.shape
        loss_his=[]# store training loss values
        train_acc_his=[] # store training accuracy
        g=[]
        r={} 
        # initialize gradient accumulation variable 'r'
        for k in range(3):
            r["W"+str(k+1)] = np.zeros((self.parameters["W"+str(k+1)].shape[0],self.parameters["W"+str(k+1)].shape[1]))
            r["b"+str(k+1)] = np.zeros(self.parameters["b"+str(k+1)].shape[0])

        for j in range(iters):
            loss,grads,h3=self.losses( X, y, reg=0.0)# loss and gradient calculation
            g=np.append(g,loss) 

            for k in range(3):
                # accumulate squared gradient
                r["W"+str(k+1)] = r["W"+str(k+1)] + grads["W"+str(k+1)] * grads["W"+str(k+1)]
                r["b"+str(k+1)] = r["b"+str(k+1)] + grads["b"+str(k+1)] * grads["b"+str(k+1)]
                # compute parameter update
                self.parameters["W"+str(k+1)] -= (learning_rate/(delta + np.sqrt(r["W"+str(k+1)]))) * grads["W"+str(k+1)]
                self.parameters["b"+str(k+1)] -= (learning_rate/(delta + np.sqrt(r["b"+str(k+1)]))) * grads["b"+str(k+1)]

            if j%10==0:
                train_acc = (np.argmax(h3,1) == np.argmax(y,1)).mean()
                print('iteration %d / %d: loss %f training accuracy: %f' % (j,iters, loss,train_acc))
                loss_his=np.append(loss_his,loss)
                train_acc_his=np.append(train_acc_his,train_acc)
                
        return loss_his,train_acc_his,g
    
    ################################################################################################
    # RMSprop
    ###############################################################################################
    
    def rmsprop_train(self, X, y=None,iters=10,momentum=0.9,learning_rate=1e-4,reg=0.25,delta=1e-7,decay_rate=0.999):
        N,D=X.shape
        loss_his=[]# store training loss values
        train_acc_his=[] # store training accuracy
        g=[]
        r={}
        # initialize gradient accumulation variable 'r'
        for k in range(3):
            r["W"+str(k+1)] = np.zeros((self.parameters["W"+str(k+1)].shape[0],self.parameters["W"+str(k+1)].shape[1]))
            r["b"+str(k+1)] = np.zeros(self.parameters["b"+str(k+1)].shape[0])

        for j in range(iters):
            loss,grads,h3=self.losses( X, y, reg=0.25)# loss and gradient calculation
            g=np.append(g,loss) 
            
            for k in range(3):
                # accumulate squared gradient using momentum
                r["W"+str(k+1)] = decay_rate*r["W"+str(k+1)] + (1-decay_rate)*(grads["W"+str(k+1)] * grads["W"+str(k+1)])
                r["b"+str(k+1)] = decay_rate*r["b"+str(k+1)] + (1-decay_rate)*(grads["b"+str(k+1)] * grads["b"+str(k+1)])
                # compute parameter update
                self.parameters["W"+str(k+1)] -= (learning_rate/(delta + np.sqrt(r["W"+str(k+1)]))) * grads["W"+str(k+1)]
                self.parameters["b"+str(k+1)] -= (learning_rate/(delta + np.sqrt(r["b"+str(k+1)]))) * grads["b"+str(k+1)]

            if j%10==0:
                train_acc = (np.argmax(h3,1) == np.argmax(y,1)).mean()
                print('iteration %d / %d: loss %f training accuracy: %f' % (j,iters, loss,train_acc))
                loss_his=np.append(loss_his,loss)
                train_acc_his=np.append(train_acc_his,train_acc)
                
        return loss_his,train_acc_his,g
    
    ################################################################################################
    # Adam
    ###############################################################################################
    def adam(self, X, y=None,iters=10,momentum=0.9,learning_rate=1e-4,reg=0.25,delta=1e-8,decay_rate1=0.9,decay_rate2=0.999):
        N,D=X.shape
        loss_his=[]# store training loss values
        train_acc_his=[] # store training accuracy
        g=[]
        r={}
        s={}
        rt={}
        st={}
        t=0 # initialize time step
        for k in range(3):
            # initialize 1st moment variable
            r["W"+str(k+1)] = np.zeros((self.parameters["W"+str(k+1)].shape[0],self.parameters["W"+str(k+1)].shape[1]))
            r["b"+str(k+1)] = np.zeros(self.parameters["b"+str(k+1)].shape[0])
            # initialize 2nd moment variable
            s["W"+str(k+1)] = np.zeros((self.parameters["W"+str(k+1)].shape[0],self.parameters["W"+str(k+1)].shape[1]))
            s["b"+str(k+1)] = np.zeros(self.parameters["b"+str(k+1)].shape[0])

        for j in range(iters):
            loss,grads,h3=self.losses( X, y, reg=0.0)# loss and gradient calculation
            g=np.append(g,loss)

            t=t+1
            for k in range(3):
                # Update biased ﬁrst moment estimate
                s["W"+str(k+1)] = decay_rate1*s["W"+str(k+1)] + (1-decay_rate1)*(grads["W"+str(k+1)])
                s["b"+str(k+1)] = decay_rate1*s["b"+str(k+1)] + (1-decay_rate1)*(grads["b"+str(k+1)])
                
                # Update biased second moment estimate
                r["W"+str(k+1)] = decay_rate2*r["W"+str(k+1)] + (1-decay_rate2)*(grads["W"+str(k+1)] * grads["W"+str(k+1)])
                r["b"+str(k+1)] = decay_rate2*r["b"+str(k+1)] + (1-decay_rate2)*(grads["b"+str(k+1)] * grads["b"+str(k+1)])
                
                # Correct bias in ﬁrst moment
                st["W"+str(k+1)]= s["W"+str(k+1)]/(1 - decay_rate1**t)
                st["b"+str(k+1)]= s["b"+str(k+1)]/(1 - decay_rate1**t)
                
                # Correct bias in second moment
                rt["W"+str(k+1)]= r["W"+str(k+1)]/(1 - decay_rate2**t)
                rt["b"+str(k+1)]= r["b"+str(k+1)]/(1 - decay_rate2**t)

                # updating parameters
                self.parameters["W"+str(k+1)] -= ((learning_rate)/(delta + np.sqrt(rt["W"+str(k+1)]))) * st["W"+str(k+1)]
                self.parameters["b"+str(k+1)] -= ((learning_rate)/(delta + np.sqrt(rt["b"+str(k+1)]))) * st["b"+str(k+1)]

            if j%10==0:
                train_acc = (np.argmax(h3,1) == np.argmax(y,1)).mean()
                print('iteration %d / %d: loss %f training accuracy: %f' % (j,iters, loss,train_acc))
                loss_his=np.append(loss_his,loss)
                train_acc_his=np.append(train_acc_his,train_acc)
                
        return loss_his,train_acc_his,g
    
    
    


# In[ ]:

########################### SGD
input_size=784
hidden_size1=1000
hidden_size2=1000
output_size=10

validation=ThreeLayerNet(input_size,hidden_size1,hidden_size2,output_size)

print("STOCHASTIC GRADIENT DESCENT")
loss_sgd,train_sgd=validation.SGD_train(X_train, y_train,iters=100,learning_rate=1e-4,reg=0.25)
   
plt.subplot(2,1,1)
plt.title('STOCHASTIC GRADIENT DESCENT')
plt.plot(np.arange(0,100,10),loss_sgd,'-o',label='MOMENTUM')
plt.xlabel('epoch')
plt.ylabel('training loss')
#plt.ylim((2.304080,2.302280))
plt.subplot(2,1,2)
plt.plot(np.arange(0,100,10),train_sgd,'-o',label='MOMENTUM')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
#plt.ylim((0.096000,0.112500))
#plt.legend()
plt.show()


# In[7]:

########################### MOMENTUM
input_size=784
hidden_size1=1000
hidden_size2=1000
output_size=10

validation=ThreeLayerNet(input_size,hidden_size1,hidden_size2,output_size)

print("MOMENTUM")
loss_momentum,train_momentum,h_m=validation.momentum_train(X_train, y_train,iters=200,momentum=0.9,learning_rate=1e-2,reg=0.25)
   
plt.subplot(2,1,1)
plt.title('MOMENTUM')
plt.plot(np.arange(0,200,20),loss_momentum,'-o',label='MOMENTUM')
plt.xlabel('epoch')
plt.ylabel('training loss')
#plt.ylim((2.304080,2.302280))
plt.subplot(2,1,2)
plt.plot(np.arange(0,200,20),train_momentum,'-o',label='MOMENTUM')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
#plt.ylim((0.096000,0.112500))
#plt.legend()
plt.show()


# In[8]:

plt.subplot(2,1,1)
plt.title('MOMENTUM')
plt.plot(np.arange(0,200,10),loss_momentum,'-o',label='MOMENTUM')
plt.xlabel('epoch')
plt.ylabel('training loss')
#plt.ylim((2.304080,2.302280))
plt.subplot(2,1,2)
plt.plot(np.arange(0,200,10),train_momentum,'-o',label='MOMENTUM')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
#plt.ylim((0.096000,0.112500))
#plt.legend()
plt.show()


# In[9]:

plt.subplot(2,1,1)
plt.title('MOMENTUM')
plt.plot(np.arange(0,200,1),h_m,label='MOMENTUM')
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.show()


# In[10]:

########################### NESTEROV
input_size=784
hidden_size1=1000
hidden_size2=1000
output_size=10

validation=ThreeLayerNet(input_size,hidden_size1,hidden_size2,output_size)

print("NESTEROV")
loss_nesterov,train_nesterov,h_n=validation.nesterov_train(X_train, y_train,iters=200,momentum=0.9,learning_rate=1e-2,reg=0.25)
   
plt.subplot(2,1,1)
plt.title('NESTEROV')
plt.plot(np.arange(0,200,10),loss_nesterov,'-o',label='nesterov')
plt.xlabel('epoch')
plt.ylabel('training loss')
#plt.ylim((2.304080,2.302280))
plt.subplot(2,1,2)
plt.plot(np.arange(0,200,10),train_nesterov,'-o',label='nesterov')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
#plt.ylim((0.096000,0.112500))
#plt.legend()
plt.show()


# In[11]:

########################### AdaGrad
input_size=784
hidden_size1=1000
hidden_size2=1000
output_size=10

validation=ThreeLayerNet(input_size,hidden_size1,hidden_size2,output_size)

print("AdaGrad")
loss_adagrad,train_adagrad,h_ada=validation.adagrad_train(X_train, y_train,iters=200,momentum=0.9,learning_rate=1e-2,reg=0.25,delta=1e-7)
plt.subplot(2,1,1)
plt.title('AdaGrad')
plt.plot(np.arange(0,200,10),loss_adagrad,'-o',label='adagrad')
plt.xlabel('epoch')
plt.ylabel('training loss')
#plt.ylim((2.304080,2.302280))
plt.subplot(2,1,2)
plt.plot(np.arange(0,200,10),train_adagrad,'-o',label='adagrad')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
#plt.ylim((0.096000,0.112500))
#plt.legend()
plt.show()


# In[12]:

########################### RMSprop
input_size=784
hidden_size1=1000
hidden_size2=1000
output_size=10

validation=ThreeLayerNet(input_size,hidden_size1,hidden_size2,output_size)

print("RMSprop")
loss_rmsprop,train_rmsprop,h_rms=validation.rmsprop_train(X_train, y_train,iters=200,momentum=0.9,learning_rate=1e-2,reg=0.25,delta=1e-7,decay_rate=0.999)

plt.subplot(2,1,1)
plt.title('RMSprop')
plt.plot(np.arange(0,200,10),loss_rmsprop,'-o',label='RMSprop')
plt.xlabel('epoch')
plt.ylabel('training loss')
#plt.ylim((2.304080,2.302280))
plt.subplot(2,1,2)
plt.plot(np.arange(0,200,10),train_rmsprop,'-o',label='RMSprop')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
#plt.ylim((0.096000,0.112500))
#plt.legend()
plt.show()


# In[13]:

########################### Adam
input_size=784
hidden_size1=1000
hidden_size2=1000
output_size=10

validation=ThreeLayerNet(input_size,hidden_size1,hidden_size2,output_size)

print("Adam")
loss_adam,train_adam,h_adam=validation.adam(X_train, y_train,iters=200,momentum=0.9,learning_rate=1e-2,reg=0.25,delta=1e-8,decay_rate1=0.9,decay_rate2=0.999)
   
plt.subplot(2,1,1)
plt.title('Adam')
plt.plot(np.arange(0,200,10),loss_adam,'-o',label='adam')
plt.xlabel('epoch')
plt.ylabel('training loss')
#plt.ylim((2.304080,2.302280))
plt.subplot(2,1,2)
plt.plot(np.arange(0,200,10),train_adam,'-o',label='adam')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
#plt.ylim((0.096000,0.112500))
#plt.legend()
plt.show()


# In[17]:

################## plot training loss of all algorithm for comparison
plt.title('MOMENTUM')
plt.plot(np.arange(0,200,10),loss_momentum,'-o',label='MOMENTUM')

plt.title('NESTEROV')
plt.plot(np.arange(0,200,10),loss_nesterov,'-o',label='nesterov')

plt.title('AdaGrad')
plt.plot(np.arange(0,200,10),loss_adagrad,'-o',label='adagrad')

plt.title('RMSprop')
plt.plot(np.arange(0,200,10),loss_rmsprop,'-o',label='RMSprop')

plt.title('Adam')
plt.plot(np.arange(0,200,10),loss_adam,'-o',label='adam')

plt.xlabel('epoch')
plt.ylabel('training loss')
plt.legend()
plt.show()


# In[21]:

#plt.title('MOMENTUM')
plt.plot(np.arange(0,200,1),h_m,label='MOMENTUM')

#plt.title('NESTEROV')
plt.plot(np.arange(0,200,1),h_n,label='nesterov')

#plt.title('AdaGrad')
plt.plot(np.arange(0,200,1),h_ada,label='adagrad')

#plt.title('RMSprop')
#plt.plot(np.arange(0,200,1),h_rms,label='RMSprop')

#plt.title('Adam')
plt.plot(np.arange(0,200,1),h_adam,label='adam')
#plt.xlim((0,100))
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.legend()
plt.show()


# In[ ]:



