#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 20:34:29 2021

@author: cedgn
"""
from tree_node import TreeNode,ID
from dt import info_gain, gain_ratio, gini, avg_gini_index, chi_squared_test
import pickle
import pydot
import random 
import copy
import numpy as np
from IPython.display import Image, display
from dt_id3 import id3,plot_tree,create_graph

with open('hw3_data/dt/data.pkl', 'rb') as f:    
    train_data , test_data , attr_vals_list , attr_names = pickle.load ( f )

    
ig = ID()
indices = [i for i in range(len(train_data))]
random.shuffle(indices)
train_set,val_set = np.array(train_data)[indices[200:]],np.array(train_data)[indices[:200]]
dt = id3(train_set,attr_names,info_gain,attr_vals_list, max ,ig )

def test(test_data,dt,print_acc = False):
    test_set = [d[:-1] for d in test_data]
    labels = [d[-1] for d in test_data]
    n = len(test_data)
    n_correct = 0 
    for i,sample in enumerate(test_data):
        label = labels[i]
        prediction = dt.test_single_sample(sample)
        if(label == prediction):
            n_correct += 1 
    if(print_acc):    
        print('accuracy : ',n_correct,'/', n,' : ', n_correct/n)
    return n_correct/n

def post_prune(dt,val_set):
    val_score = test(val_set,dt)
    pruned_score = 0 
    labels = [d[-1] for d in val_set]
    if(dt.attribute in ['unacc','acc']):
        return
    if(dt.values[0] > dt.values[1]): # unacc
        pruned_score = np.mean(labels == 'unacc')
    else: 
        pruned_score = np.mean(labels == 'acc')
    
    if (val_score > pruned_score):
        for child in dt.children:
            post_prune(child,val_set)
        
    else: # pruned score has more accuracy
        dt.prune()



post_prune(dt,val_set)


test(test_data,dt,True)

create_graph(dt)


