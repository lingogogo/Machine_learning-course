# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:11:51 2023

@author: Gordon
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def basis_function_gau(X,mu,s):
    phi = np.exp(-((X - mu)**2)/(2*s**2))
    return phi

def basis_function_sigmoid(X,mu,s):
    a = (X - mu)/s
    phi = 1/(1 + np.exp(-a))
    return phi

def design_matrix(X,X_mean,X_std):
    #15000 data input  feature
    phi = np.ones([X.shape[0],X.shape[1]])
    
    for i in range(0,phi.shape[1]):
        for j in range(0,phi.shape[0]):
            phi[j][i] = basis_function_sigmoid(X[j][i], X_mean[i],X_std[i] )
    return phi

def change_sextonum(X):
    for i in range(0,X.shape[0]):
        if X[i,0] == 'male':
            X[i,0] = 1
        else:
            X[i,0] = 0
    return X

def BLR(train_data,train_label,test_data,train_mean,train_std):
    train_phi = design_matrix(train_data, train_mean, train_std)
    Weights = np.linalg.inv(np.identity(train_phi.shape[1]) + train_phi.T @ train_phi) @ train_phi.T @ train_label
    y_prediction = design_matrix(test_data,train_mean,train_std) @ Weights
    return y_prediction

def MLR(train_data,train_label,test_data,train_mean,train_std):
    train_phi = design_matrix(train_data, train_mean, train_std)
    Weights = np.linalg.inv(train_phi.T @ train_phi) @ train_phi.T @ train_label
    y_prediction = design_matrix(test_data,train_mean,train_std) @ Weights
    return y_prediction

def drawlinearline(slope, intercept,linef):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, linestyle = linef)
    
def main():
    
        ## Data input and make them together
    filename_1 = 'exercise.csv'
    filename_2 = 'calories.csv'

    read_ex = pd.read_csv(filename_1)
    read_cal = pd.read_csv(filename_2)
    merge_cal_ex = pd.concat([read_ex,read_cal.iloc[:,1]],axis= 1)
    
    train = merge_cal_ex.iloc[0:10500,0:].to_numpy()
    merge_cal_ex.iloc[0:10500,0:].to_csv('train.csv')
    validation = merge_cal_ex.iloc[10500:12000,0:].to_numpy()
    merge_cal_ex.iloc[10500:12000,0:].to_csv('validation.csv')
    test = merge_cal_ex.iloc[12000:15000,0:].to_numpy()
    merge_cal_ex.iloc[12000:15000,0:].to_csv('test.csv')
    # train_mean and train_std from pd.describe()
    train_mean = merge_cal_ex.iloc[0:10500,0:].describe().iloc[1,1:7].to_numpy()
    train_std = merge_cal_ex.iloc[0:10500,0:].describe().iloc[2,1:7].to_numpy()

    train_data = train[:,2:8]
    train_label = train[:,-1]
    test_data = test[:,2:8]
    test_label = test[:,-1]
    validation_data = validation[:,2:8]
    validation_label = validation[:,-1]

    y_pred = BLR(train_data,train_label,validation_data,train_mean,train_std)
    y_pred_MLR = MLR(train_data,train_label,test_data,train_mean,train_std)
    mse_BLR = (1/validation.shape[0])*sum(pow(validation_label - y_pred,2))
    mse_MLR = (1/test.shape[0])*sum(pow(test_label - y_pred_MLR,2))
    
    ## Q3
    ## Use public package
    # import statsmodels.api as sm
    # import statsmodels.formula.api as smf
    # import pymc3 as pm
    
    # input_x = merge_cal_ex.iloc[10500:12000,:] # test data
    # mod = smf.ols(formula='Calories ~  Duration',data = input_x)
    # res = mod.fit()
    # plt.figure()
    # intercept_ol = res.params['Intercept']
    # Duration = res.params['Duration']
    # plt.scatter(validation[:,5],validation_label[:],marker = ',') # Duration
    # drawlinearline(Duration,intercept_ol,'dashed')
    # input_x = input_x.iloc[:,5].to_numpy()
     
    
    # with pm.Model() as linear_model:
    #     intercept = pm.Normal('Intercept', mu = 0, sd = 10)
    #     slope = pm.Normal('slope', mu = 0, sd = 10)
    #     sigma = pm.HalfNormal('sigma', sd = 10)
    #     mean = intercept + slope * merge_cal_ex.iloc[10500:12000,1:].loc[:, 'Duration']
    #     Y_obs = pm.Normal('Y_obs', mu = mean, sd = sigma, observed = merge_cal_ex.iloc[10500:12000,1:].loc[:, 'Calories'])
    #     step = pm.NUTS()
    #     # Posterior distribution
    #     linear_trace = pm.sample(100, step)
       
    # pm.plot_posterior_predictive_glm(linear_trace, samples = 100, eval=np.linspace(2, 30, 100), linewidth = 1, 
    #                                   color = 'red',  lm = lambda x, sample: sample['Intercept'] + sample['slope'] * x);
                                   
    
    # plt.title('Posterior Predictions', size = 20);
    # plt.ylabel('Calories', size = 18);
    
    ## Use custom function
    # def MLR_beta_intercept(X, Y):
    #     coe = np.linalg.inv(X.T @ X) @ X.T @ Y
    #     return coe

    # def BLR_beta_intercept(X, Y):
    #     coe = np.linalg.inv(np.identity(X.shape[1]) + X.T @ X) @ X.T @ Y
    #     return coe
    
    # plot_data_X = merge_cal_ex.iloc[10500:12000,:]
    # plot_data_X['intercept'] = 1
    # plot_data_X = plot_data_X.loc[:,['Duration','intercept']]
    # plot_data_Y = merge_cal_ex.iloc[10500:12000,:].loc[:,'Calories']
    # print(plot_data_X)
    # cm = MLR_beta_intercept(plot_data_X, plot_data_Y)
    # cb = BLR_beta_intercept(plot_data_X, plot_data_Y)
    # print(cm)
    
    
    
    # plt.figure()
    # plt.scatter(validation[:,5],validation_label[:],marker = ',')
    # drawlinearline(cb[0],cb[1],'solid')
    # drawlinearline(cm[0],cm[1],'dashed')
    # plt.title('Calories and Duration')
    # plt.xlabel('Duration')
    # plt.ylabel('Calories')
    
    ## Q4
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    
    lr = LinearRegression()
    lr.fit(train_data,train_label)
    pred = lr.predict(test_data)
    mse_LR = 1/test.shape[0]*sum(pow(test_label - pred,2))
    
    lr = BayesianRidge()
    lr.fit(train_data,train_label)
    pred = lr.predict(test_data)
    mse_BR = 1/test.shape[0]*sum(pow(test_label - pred,2))
    
    lr = Lasso()
    lr.fit(train_data,train_label)
    pred = lr.predict(test_data)
    mse_L = 1/test.shape[0]*sum(pow(test_label - pred,2))
    
    lr = ElasticNet()
    lr.fit(train_data,train_label)
    pred = lr.predict(test_data)
    mse_EN = 1/test.shape[0]*sum(pow(test_label - pred,2))
    
    print("LinearRegression: ", mse_LR)
    print("BayesianRidge: ",mse_BR)
    print("Lasso: ",mse_L)
    print("ElasticNet: ",mse_EN)
    
    return mse_BLR,mse_MLR

if __name__ == '__main__':
      
      [mse,mse_2] = main()
      print("mse_BLR: ", mse)
      print("mse_MLR: ",mse_2)
