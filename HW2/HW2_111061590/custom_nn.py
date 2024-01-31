# -*- coding: utf-8 -*-


import numpy as np
import random
class custom_NN():
    def __init__(self):
        self.input = 2 # feature numbers
        self.output = 3 # class number 
        self.hidden_units = 512 # single layer
        seed = 123
        np.random.seed(seed)
        random.seed(seed)
        self.w1 = np.random.randn(self.input, self.hidden_units)*0.01
        self.w2 = np.random.randn(self.hidden_units, self.output) *0.01
        self.b1 = np.zeros(self.hidden_units)
        self.b2 = np.zeros(self.output)
    
    def relu(self,X):
        
        for i in range(0,X.shape[0]):
            if(X[i] <= 0):
                X[i] = 0
        return X
    
    def softmax2(self,z3):
        ans = []
        
        denom = sum(np.exp(z3))
        for i in range(0,z3.shape[0]):
            ans.append(np.exp(z3[i])/denom)
        ans = np.array(ans)

        return ans

    def forward_pass(self, X):
        
        self.z2 = np.dot(X, self.w1)+self.b1
        self.a2 = self.relu(self.z2) 
        self.z3 = np.dot(self.a2, self.w2) + self.b2
        self.a3 = self.softmax2(self.z3)

        return self.a3
    #CEL
    def loss(self,y_hat, y):

        log_probs= 0;
        if y_hat[int(y)] == 0:
            y_hat[int(y)] = 10**-10
        log_probs = -np.log2(y_hat[int(y)])
        loss = log_probs
        return loss
   
    def backward_pass(self,y_pre, X, y):
        delta3 = y_pre
        delta3[int(y)] -= 1
        self.dw2 = np.dot(self.a2.reshape(self.hidden_units,1),delta3.reshape(3,1).T)
        self.db2 = np.sum(delta3,axis = 0)
        
        self.dz1 = np.dot(delta3,self.w2.T) * self.relu_prime(self.a2)
        self.dw1 = np.dot(X.reshape(2,1),self.dz1.reshape(self.hidden_units,1).T)
        self.db1 = np.sum(self.dz1,axis = 0)
        
    
    def relu_prime(self,z):
        ans = z;
        for j in range(0,z.shape[0]):
            if ans[j] <=0:
                ans[j] = 0
            else:
                ans[j] = 1
        return ans
        
    def _update(self, learning_rate=0.01):
        self.w1 = self.w1 - learning_rate*self.dw1
        self.b1 = self.b1 - learning_rate*self.db1
        self.w2 = self.w2 - learning_rate*self.dw2
        self.b2 = self.b2 - learning_rate*self.db2
               
    def train(self, X, epoch=3):
        tot_loss = []
        for _ in range(0,epoch):
            for i in range(0,1470):
                y_hat = self.forward_pass(X[i,0:2])
                loss = self.loss(y_hat, X[i,2])
                self.backward_pass(y_hat,X[i,0:2],X[i,2])
                self._update()
                tot_loss.append(loss)
        return np.array(tot_loss)
                
                
    def predict(self, X):      
        ans = []
        for i in range(0,X.shape[0]):
            y_hat = self.forward_pass(X[i,0:2])
            ans.append(np.argmax(y_hat))
        return np.array(ans)
    
    def score(self, predict, y):
        cnt = np.sum(predict==y)
        return (cnt/len(y))*100
    