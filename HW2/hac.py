#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 19:05:55 2020

@author: cedgn
"""

import numpy as np 
from math import inf
import os
import matplotlib as plt
path = os.path.join('hw2_data','hac')

data1 = np.load(os.path.join(path,"data1.npy"))
data2 = np.load(os.path.join(path,"data2.npy")) 
data3 = np.load(os.path.join(path,"data3.npy")) 
data4 = np.load(os.path.join(path,"data4.npy"))

def plot_clustering(data,clusters,crit):
    labels = np.zeros((data.shape[0],))
    for c_index,cluster in enumerate(clusters):
        for c in cluster:
            labels[c] = c_index
    

    plt.pyplot.scatter(data[:,0],data[:,1],c = labels)
    plt.pyplot.title('HAC with %s Criterion' %crit )
    plt.pyplot.show()


def single_linkage(data,cluster1,cluster2):
    # min dist between clusters
    min_dist = inf
    for c1 in cluster1:
        for c2 in cluster2:
            current_dist = np.linalg.norm(data[c1]-data[c2])
            if( current_dist < min_dist):
                min_dist = current_dist
    return min_dist

def complete_linkage(data,cluster1,cluster2):
    max_dist = 0
    for c1 in cluster1:
        for c2 in cluster2:
            current_dist = np.linalg.norm(data[c1]-data[c2])
            if( current_dist > max_dist):
                max_dist = current_dist
    return max_dist

def average_linkage(data,cluster1,cluster2):
    total_dist = 0
    for c1 in cluster1:
        for c2 in cluster2:
            total_dist += np.linalg.norm(data[c1]-data[c2])
    avg_dist = total_dist/(len(cluster1)*len(cluster2))
    return avg_dist

def centroid(data,cluster1,cluster2):
    c1_centroid = np.mean(data[cluster1],axis = 0 )  
    c2_centroid = np.mean(data[cluster2],axis = 0 ) 
    dist = np.linalg.norm(c1_centroid-c2_centroid)
    return dist
       
def find_closest(dist_matrix):
    n = dist_matrix.shape[0]
    idx = np.argmin(dist_matrix)
    c1,c2 = int(idx/n), idx%n
    return c1,c2

def hac(data , k , criterion = 'Single-Linkage'):
    if(criterion == 'Single-Linkage'):
        dist_func = single_linkage
    elif (criterion == 'Complete-Linkage'):
        dist_func = complete_linkage
    elif (criterion == 'Average-Linkage'):
        dist_func = average_linkage
    elif (criterion == 'Centroid'):
        dist_func = centroid
    
    clusters = [[i] for i in range(data.shape[0])]
    cluster_history = [clusters]
    while(len(clusters) > k ):
        dist_matrix = np.ones((len(clusters),len(clusters)))*np.inf
        for c1_idx,cluster1 in enumerate(clusters):
            for c2_idx,cluster2 in enumerate(clusters):
                if c1_idx == c2_idx:
                    continue
                else:
                    dist_matrix[c1_idx][c2_idx] = dist_func(data,cluster1,cluster2)
                    
        c1,c2 = find_closest(dist_matrix) # c1 < c2
        clusters[c1] = clusters[c1]+clusters[c2]
        del clusters[c2]
        cluster_history.append(clusters)
        print(len(clusters))
        if(len(clusters) < 10):
            plot_clustering(data,clusters,criterion)
    
    
    return clusters

crits = ['Single-Linkage','Complete-Linkage','Average-Linkage','Centroid']

for crit in crits:
    hac(data1,2,crit)
    hac(data2,2,crit)
    hac(data3,2,crit)
    hac(data4,4,crit)
    
    

