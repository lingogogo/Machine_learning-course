# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:06:35 2023

@author: Gordon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'wine.csv'
df = pd.read_csv(filename)
#build up dataset
df_0 = pd.DataFrame(columns=list(df.columns))
df_1 = pd.DataFrame(columns=list(df.columns))
df_2 = pd.DataFrame(columns=list(df.columns))
# take all the different target into the DataFrame
df_0 = df.loc[df['target'] == 0]
df_1 = df.loc[df['target'] == 1]
df_2 = df.loc[df['target'] == 2]
# random the dataset
df_0 = df_0.reindex(np.random.permutation(df_0.index))
df_1 = df_1.reindex(np.random.permutation(df_1.index))
df_2 = df_2.reindex(np.random.permutation(df_2.index))
# set all the dataset with 1 to last index
df_0.reset_index(drop=True, inplace=True)
df_1.reset_index(drop=True, inplace=True)
df_2.reset_index(drop=True, inplace=True)

# build up train and test
train = pd.DataFrame(columns=list(df.columns))
test = pd.DataFrame(columns=list(df.columns))

test = pd.concat([df_0.loc[0:19,:],df_1.loc[0:19,:],df_2.loc[0:19,:]],axis = 0)
test.reset_index(drop=True, inplace=True)

train = pd.concat([df_0.loc[20:len(df_0),:],df_1.loc[20:len(df_1),:],df_2.loc[20:len(df_2),:]],axis = 0)
train.reset_index(drop=True, inplace=True)

train.to_csv('train.csv')
test.to_csv('test.csv')

# problem 2
data_0 = train.loc[train['target'] == 0]
data_1 = train.loc[train['target'] == 1]
data_2 = train.loc[train['target'] == 2]
data_0 = np.array(data_0.describe())[:,1:]
data_1 = np.array(data_1.describe())[:,1:]
data_2 = np.array(data_2.describe())[:,1:]
mean_0 = data_0[1,:]
mean_1 = data_1[1,:]
mean_2 = data_2[1,:]
std_0 = data_0[2,:]
std_1 = data_1[2,:]
std_2 = data_2[2,:]

import scipy
testd = np.array(test)
#scipy.stats.norm(distance, R).pdf(z[i])
p_0 = 175/483
p_1 = 205/483
p_2 = 103/483

ans = []
j = 0
while(j < np.shape(testd)[0]):
    
    posteroir_0 = p_0
    posteroir_1 = p_1
    posteroir_2 = p_2
    for i in range(0,13):
        posteroir_0 *= scipy.stats.norm(mean_0[i], std_0[i]).pdf(testd[j,i+1])
        posteroir_1 *= scipy.stats.norm(mean_1[i], std_1[i]).pdf(testd[j,i+1])
        posteroir_2 *= scipy.stats.norm(mean_2[i], std_2[i]).pdf(testd[j,i+1])
    evi =  posteroir_0+ posteroir_1+posteroir_2
    posteroir_0/=evi;posteroir_1/=evi;posteroir_2/=evi;
    
    if (posteroir_0 > posteroir_1) & (posteroir_0 > posteroir_2):
        ans.append(0)
        
    elif(posteroir_1 > posteroir_0) & (posteroir_1 > posteroir_2):
        ans.append(1)
    elif(posteroir_2 > posteroir_1) & (posteroir_2 > posteroir_0):
        ans.append(2)
        
    j+=1

count = 0
for i in range(0,np.shape(testd)[0]):
    if(int(testd[i,0]) == int(ans[i])):
        count+=1
        
print("accuracy: ",count/60)


# posterior_0, posterior_1, posterior_2
from sklearn.decomposition import PCA
test_pca = testd[:,1:]
pca = PCA(n_components = 3)
pca.fit(test_pca)
test_new = pca.transform(test_pca)
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(test_new[:20,0], test_new[:20,1],test_new[:20,2],color = 'b')
ax.scatter(test_new[20:40,0], test_new[20:40,1],test_new[20:40,2],color = 'r')
ax.scatter(test_new[40:60,0], test_new[40:60,1],test_new[40:60,2],color = 'g')

test_pca2 = testd[:,1:]
pca2 = PCA(n_components = 2)
pca.fit(test_pca2)
test_new2 = pca.transform(test_pca2)
fig = plt.figure()

plt.scatter(test_new[:20,0], test_new[:20,1],color = 'b')
plt.scatter(test_new[20:40,0], test_new[20:40,1],color = 'r')
plt.scatter(test_new[40:60,0], test_new[40:60,1],color = 'g')

