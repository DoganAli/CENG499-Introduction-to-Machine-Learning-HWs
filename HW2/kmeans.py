#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 00:47:14 2020

@author: cedgn
"""
import numpy as np 
import os
import matplotlib as plt
path = os.path.join('hw2_data','kmeans')
clustering1 = np.load(os.path.join(path,"clustering1.npy"))
clustering2 = np.load(os.path.join(path,"clustering2.npy")) 
clustering3 = np.load(os.path.join(path,"clustering3.npy")) 
clustering4 = np.load(os.path.join(path,"clustering4.npy"))


def initialize_centroids( data , k ):
    X = data[:,0] # first columns of each data point, i.e, x values
    Y = data[:,1] # second columns of each data point, i.e, y values
    x_min, x_max = min(X), max(X)
    y_min, y_max = min(Y), max(Y)
    
    centroids = [[np.random.uniform(low=x_min,high = x_max), np.random.uniform(low = y_min,high = y_max)] for i in range(k)]
    centroids = np.array(centroids)
    
    
    return centroids


def closest_centroid(data, centroids):
    distances = []
    for x in data:
        min_centroid_idx = 0
        min_distance = np.linalg.norm(x-centroids[0])
        for c_idx,c in enumerate(centroids[1:],1):
            current_dist = np.linalg.norm(x-c)
            if current_dist < min_distance:
                min_centroid_idx = c_idx
                min_distance = current_dist
        distances.append(min_centroid_idx)
        
    return np.array(distances)


def find_new_centroids(data,labels,k,old_centroids):
    # returns the new centroids
    new_centroids = []
    for i in range(k):
        label_i = np.where(labels == i)[0]
        if(len(label_i) >0 ):
            x , y = np.nanmean(data[:,0][label_i]) , np.nanmean(data[:,1][label_i])
            new_centroids.append([x,y])
        else:
            new_centroids.append(old_centroids[i])

    return np.array(new_centroids)


def objective(data ,labels, centroids):
    objective_func = 0 
    for i, label in enumerate(labels):
        centroid = centroids[label]
        objective_func += np.linalg.norm(data[i]-centroid)

    return objective_func

def kmeans(data , k , plot = True , eps = 0 ): # epsilon speeds up because it stops early when the convergence is obvious. For a true kmeans algo, just make it 0.0
    current_centroids = initialize_centroids(data, k)
    labels = closest_centroid(data,current_centroids)
    obj_func = round(objective(data,labels,current_centroids),2) ;
    diff = 1 
    while diff > eps :   
        if(plot):
            plt.pyplot.scatter(data[:,0],data[:,1],c = labels)
            plt.pyplot.scatter(current_centroids[:,0],current_centroids[:,1],c = 'b')
            plt.pyplot.show()
        new_centroids = find_new_centroids(data, labels,k,current_centroids)
        diff = np.linalg.norm(current_centroids-new_centroids)
        current_centroids = new_centroids
        labels = closest_centroid(data,current_centroids)        
        obj_func = round(objective(data,labels,current_centroids),2)
    plt.pyplot.scatter(data[:,0],data[:,1],c = labels)
    plt.pyplot.scatter(current_centroids[:,0],current_centroids[:,1],c = 'b')
    plt.pyplot.show()
    return current_centroids, labels , obj_func


def average_obj(data,k,plot= False):
    avg_obj_func = 0
    for i in range(10):
        _,_,obj_func = kmeans(data,k,plot )
        avg_obj_func += obj_func
    print('Average objection : ',avg_obj_func/10 )
    return avg_obj_func/10

def plot_elbow(data,title = 'Objective - k ',plot = False):
    obj_functions = []
    x = []
    for k in range(1,11):
        print('k : ',k)
        x.append(k)
        obj_functions.append( average_obj(data,k,plot) )
          
    plt.pyplot.plot(x,obj_functions)
    plt.pyplot.xlabel('k')
    plt.pyplot.ylabel('Objective')
    plt.pyplot.title(title)
    plt.pyplot.show()


## PLOTING ELBOWS
plot_elbow(clustering1,title = 'Clustering1 Objective Function')
plot_elbow(clustering2,title = 'Clustering2 Objective Function')
plot_elbow(clustering3,title = 'Clustering3 Objective Function')
plot_elbow(clustering4,title = 'Clustering4 Objective Function')



## KMEAN ALGORITHM wıth best k values
kmeans(clustering1,2)
kmeans(clustering2,3)
kmeans(clustering3,4)
kmeans(clustering4,5)



