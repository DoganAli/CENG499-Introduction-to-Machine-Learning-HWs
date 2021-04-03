#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:24:59 2020

@author: cedgn
"""
import matplotlib.pyplot as plt
import numpy as np
import os

path = os.path.join('hw2_data','knn')

train_data = np.load(os.path.join(path,"train_data.npy"))
train_labels = np.load(os.path.join(path,"train_labels.npy"))
test_data = np.load(os.path.join(path,"test_data.npy"))
test_labels = np.load(os.path.join(path,"test_labels.npy"))
train_labels = train_labels.reshape(train_labels.shape[0],1)
test_labels = test_labels.reshape(test_labels.shape[0],1)

def euclidian(v1,v2):
    return (np.sum((v1-v2)**2))**0.5


def knn(train_data,train_labels,test_data,k):
    '''
    train_data : numpy array (N,D)
    train_labels: numpy array (N,)
    test_data : numpy array (M,D)
    k : number of nearest neighbors to be looked

    Returns:
    predicted_labels : (M,)

    '''
    predicted_labels = list()
    for test_row in test_data:
        distances = list() # contains tuples (d,l) where d is the distance,l is the label of that training data
        
        for train_index, train_row in enumerate(train_data) :
            dist = euclidian(train_row,test_row)
            distances.append((dist,int(train_labels[train_index])))
            
        distances.sort(key =lambda x: x[0] )
        k_neighbors = [i[1] for i in distances[:k] ] # just taking the labels, dist is no longer important
        prediction = max(set(k_neighbors), key = k_neighbors.count)
        predicted_labels.append(prediction)
    
    predictions = np.array(predicted_labels)
    predictions = predictions.reshape(predictions.shape[0],1)
    return predictions

def ten_fold_cross_knn(train_data,train_labels,k):
    # divide train data into 10
    #  return 10 validation dataset accuries obtained from 10-fold CV
    random_indices = np.random.permutation(train_data.shape[0])
    val_set_size = int(train_data.shape[0] / 10)
    accuracies = list()
    for i in range(10):
        idx = i*val_set_size
        val_index = random_indices[idx:idx+val_set_size]
        train_index = np.setdiff1d(random_indices,val_index)
        train_set,val_set = train_data[train_index,:], train_data[val_index,:]
        train_label_set,val_label_set = train_labels[train_index,:], train_labels[val_index,:]
        predictions = knn(train_set,train_label_set,val_set,k)
        accuracy = np.mean(predictions == val_label_set)
        accuracies.append(accuracy)
        
    return accuracies    


def train(train_data,train_labels,plot = True):
    knn_accs = list()
    for k in range(1,200,2):
        accs = ten_fold_cross_knn(train_data,train_labels,k)
        avg_acc = np.mean(accs)
        knn_accs.append(avg_acc)
    
    if(plot):
        x_val = [i for i in range(1,200,2)]     
        plt.plot(x_val,knn_accs)
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.title('KNN accuracy on 10-fold Cross Validation')
    return knn_accs
        
def test(train_data,train_labels,test_data,test_labels,k):
    predictions = knn(train_data,train_labels,test_data,k)
    accuracy = np.mean(predictions == test_labels)
    print("accuracy test data : ", accuracy)        
    return predictions 
        
        
        
k_accuracies = train(train_data,train_labels,plot = True)  

idx = np.argmax(np.array(k_accuracies))

test(train_data,train_labels,test_data,test_labels,8)

        
        