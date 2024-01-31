# -*- coding: utf-8 -*-
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import glob

images_1 = [];images_2 = [];images_3 = []
label1 = [0]*490
label2 = [1]*490
label3 = [2]*490
path_1 = glob.glob("Data_train/Carambula/*.png")
path_2 = glob.glob("Data_train/Lychee/*.png")
path_3 = glob.glob("Data_train/Pear/*.png")

for img in path_1:
    n = np.array(cv2.imread(img,cv2.IMREAD_GRAYSCALE))
    images_1.append(n)

for img in path_2:
    n = np.array(cv2.imread(img,cv2.IMREAD_GRAYSCALE))
    images_2.append(n)

for img in path_3:
    n = np.array(cv2.imread(img,cv2.IMREAD_GRAYSCALE))
    images_3.append(n)
    

images_1 = np.array(images_1)/255
images_2 = np.array(images_2)/255
images_3 = np.array(images_3)/255

ima1= images_1.reshape(490,1024)
ima2 = images_2.reshape(490,1024)
ima3 = images_3.reshape(490,1024)

label = np.array([[0]*490,[1]*490,[2]*490]).reshape(1470,1)
tot_img = np.r_[ima1,ima2,ima3]

pca = PCA(n_components=2)
train_data = pca.fit_transform(tot_img)

train_data = np.hstack((train_data,label))

# Neural Network
from custom_nn import custom_NN
#from Hw2_batch import custom_NN3


images_5 = [];images_6 = [];images_4 = []
path_4 = glob.glob("Data_test/Carambula/*.png")
path_5 = glob.glob("Data_test/Lychee/*.png")
path_6 = glob.glob("Data_test/Pear/*.png")
for img in path_4:
    n = np.array(cv2.imread(img,cv2.IMREAD_GRAYSCALE))
    images_4.append(n)

for img in path_5:
    n = np.array(cv2.imread(img,cv2.IMREAD_GRAYSCALE))
    images_5.append(n)

for img in path_6:
    n = np.array(cv2.imread(img,cv2.IMREAD_GRAYSCALE))
    images_6.append(n)

images_4 = np.array(images_4)/255
images_5 = np.array(images_5)/255
images_6 = np.array(images_6)/255

test_number = 166
ima4 = images_4.reshape(test_number,1024)
ima5 = images_5.reshape(test_number,1024)
ima6 = images_6.reshape(test_number,1024)


test_label = np.array([[0]*test_number,[1]*test_number,[2]*test_number]).reshape(test_number*3,1)
test_data = np.r_[ima4,ima5,ima6]

test_data_pca = pca.transform(test_data)
test_data_pca = np.hstack((test_data_pca,test_label))
np.random.shuffle(train_data)

# layer 2
NN = custom_NN()
# training epoch
episode_1 = 3
# For plot
x_1 = []
for i in range(0,episode_1*1470):
    x_1.append(i)

# training
loss = NN.train(train_data,episode_1)
x_1 = np.array(x_1)
plt.figure()
plt.plot(x_1,loss)

pre_train = NN.predict(train_data)
print("2 layer train score: ",NN.score(pre_train, train_data[:,2]))
NN_pre = NN.predict(test_data_pca)
print("2 layer test score: ",NN.score(NN_pre, test_data_pca[:,2]))


# layer 3
from custom_nn3 import custom_NN3
# training epoch
episode_2 = 3

x_2 = []
for i in range(0,episode_2*1470):
    x_2.append(i)
x_2 = np.array(x_2)
# training
NN3 = custom_NN3()
loss_3 = NN3.train(train_data,episode_2)
NN3_pre_train = NN3.predict(train_data)
print("3 layer train score: ",NN3.score(NN3_pre_train, train_data[:,2]))
NN3_pre = NN3.predict(test_data_pca)
print("3 layer test score: ",NN3.score(NN3_pre, test_data_pca[:,2]))

plt.figure()
plt.plot(x_2,loss_3)

# decision region preprocess
sample = 200
nx, ny = (sample, sample)
x_min = np.min(train_data[:,0])
x_max = np.max(train_data[:,0])
y_min = np.min(train_data[:,1])
y_max = np.max(train_data[:,1])
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
xv, yv = np.meshgrid(x, y)
xv = xv.reshape(sample*sample,1)
yv = yv.reshape(sample*sample,1)
xyv = np.hstack((xv,yv))
# Use 2 layer predict
d_r = NN.predict(xyv)
# plot
plt.figure()
plt.contourf(xv.reshape(sample,sample), yv.reshape(sample,sample), d_r.reshape(sample,sample), 10, alpha=.3, cmap=plt.cm.jet)
for i in range(0,train_data.shape[0]):
    if train_data[i,2] == 0:
        kx = plt.scatter(train_data[i,0],train_data[i,1],color = 'b')
    if train_data[i,2] == 1:
        yx = plt.scatter(train_data[i,0],train_data[i,1],color = 'g')
    if train_data[i,2] == 2:
        zx = plt.scatter(train_data[i,0],train_data[i,1],color = 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision regions of training data(2 layers)')


# Use 3 layer predict
d_r3 = NN3.predict(xyv)
# plot
plt.figure()
plt.contourf(xv.reshape(sample,sample), yv.reshape(sample,sample), d_r3.reshape(sample,sample), 10, alpha=.3, cmap=plt.cm.jet)
for i in range(0,train_data.shape[0]):
    if train_data[i,2] == 0:
        kx = plt.scatter(train_data[i,0],train_data[i,1],color = 'b')
    if train_data[i,2] == 1:
        yx = plt.scatter(train_data[i,0],train_data[i,1],color = 'g')
    if train_data[i,2] == 2:
        zx = plt.scatter(train_data[i,0],train_data[i,1],color = 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision regions of training data(3 layers)')


# decision region for testing data (2 layers)
# sample = 200
# nx, ny = (sample, sample)
# x_min = np.min(test_data_pca[:,0])
# x_max = np.max(test_data_pca[:,0])
# y_min = np.min(test_data_pca[:,1])
# y_max = np.max(test_data_pca[:,1])
# x = np.linspace(x_min, x_max, nx)
# y = np.linspace(y_min, y_max, ny)
# xv, yv = np.meshgrid(x, y)
# xv = xv.reshape(sample*sample,1)
# yv = yv.reshape(sample*sample,1)
# xyv = np.hstack((xv,yv))
# # Use 2 layer predict
# d_r = NN.predict(xyv)
# plot
# plt.figure()
# plt.contourf(xv.reshape(sample,sample), yv.reshape(sample,sample), d_r.reshape(sample,sample), 10, alpha=.3, cmap=plt.cm.jet)
# for i in range(0,NN_pre.shape[0]):
#     if test_data_pca[i,2] == 0:
#         kx = plt.scatter(test_data_pca[i,0],test_data_pca[i,1],color = 'b')
#     if test_data_pca[i,2] == 1:
#         yx = plt.scatter(test_data_pca[i,0],test_data_pca[i,1],color = 'g')
#     if test_data_pca[i,2] == 2:
#         zx = plt.scatter(test_data_pca[i,0],test_data_pca[i,1],color = 'r')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Decision regions of testing data(2 layers)')

## decision region for testing data (3 layers)
# d_r3 = NN3.predict(xyv)
# # plot
# plt.figure()
# plt.contourf(xv.reshape(sample,sample), yv.reshape(sample,sample), d_r3.reshape(sample,sample), 10, alpha=.3, cmap=plt.cm.jet)
# for i in range(0,NN3_pre.shape[0]):
#     if test_data_pca[i,2] == 0:
#         kx = plt.scatter(test_data_pca[i,0],test_data_pca[i,1],color = 'b')
#     if test_data_pca[i,2] == 1:
#         yx = plt.scatter(test_data_pca[i,0],test_data_pca[i,1],color = 'g')
#     if test_data_pca[i,2] == 2:
#         zx = plt.scatter(test_data_pca[i,0],test_data_pca[i,1],color = 'r')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Decision regions of testing data(3 layers)')

## put two decision region together
# d_r = NN.predict(xyv)
# d_r_3 = NN3.predict(xyv)
# plt.figure()
# plt.contourf(xv.reshape(sample,sample), yv.reshape(sample,sample), d_r_3.reshape(sample,sample), 10, alpha=.8, cmap=plt.cm.jet)
# plt.contourf(xv.reshape(sample,sample), yv.reshape(sample,sample), d_r.reshape(sample,sample), 10, alpha=.3, cmap=plt.cm.jet)