import numpy as np
from math import *
import random

def load_dataSet():
    text = open('testSet.txt').readlines()
    num = len(text)
    data_matrix = np.zeros((num,3))
    label_list = [0] * num
    data_matrix[:,0] += 1.0
    for i in range(num):
        line = text[i].strip().split('\t')
        data_matrix[i,1:] = line[:-1]
        label_list[i] = int(line[-1])
    return data_matrix,label_list

def sigmoid(inX):
    return (1.0 / (1 + exp(-inX)))

def gradient_ascent(data_matrix,label_list):
    data_matrix = np.mat(data_matrix)
    label_list = np.mat(label_list).T
    m,n = np.shape(data_matrix)
    a = 0.001
    max_iter = 500
    weights = np.mat(np.ones((n,1)))
    y = np.mat(np.ones((m,1)))
    for i in range(max_iter):
        for j in range(m):
            y[j] = sigmoid((data_matrix * weights)[j,0])
        error = label_list - y
        weights = weights + 0.001 * data_matrix.T * error
    return weights

def plot_best_fit(data_matrix,label_list,weight):
    import matplotlib.pyplot as plt
    num = len(label_list)
    xcord1 = []; ycord1 = [];
    xcord2 = []; ycord2 = [];
    for i in range(num):
        if label_list[i] == 1:
            xcord1.append(data_matrix[i,1])
            ycord1.append(data_matrix[i,2])
        else:
            xcord2.append(data_matrix[i,1])
            ycord2.append(data_matrix[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=50,c='blue',marker='s')
    ax.scatter(xcord2,ycord2,s=50,c='red')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weight[0] - weight[1] * x) / weight[2]
    # y = np.array(y.tolist()[0])
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.xlabel('X2')

def stoc_gradient_ascent0(data_matrix,label_list):
    m,n = np.shape(data_matrix)
    a = 0.01
    weights = np.ones(n)
    for j in range(100):
        for i in range(m):
            y = sigmoid(sum(data_matrix[i] * weights))
            error = label_list[i] - y
            weights = weights + a * data_matrix[i] * error
    return weights

def stoc_gradient_ascent1(data_matrix,label_list,iter_num = 150):
    m,n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(iter_num):
        data_index = range(m)
        for i in range(m):
            a = 4.0/(1.0+i+j) + 0.01
            index = int(random.uniform(0,len(data_index)))
            del(data_index[index])
            y = sigmoid(sum(data_matrix[index] * weights))
            error = label_list[index] - y
            weights = weights + a * data_matrix[index] * error

    return weights