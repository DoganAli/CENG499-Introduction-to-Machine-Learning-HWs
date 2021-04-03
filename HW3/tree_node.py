#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:51:19 2020

@author: cedgn
"""

import pickle


class ID:
    def __init__(self):
        self.id = 0 
    def get_id(self):
        self.id +=1
        return str(self.id)
    
attribute_indices = {'buying':0, 'maint':1, 'doors':2, 'persons':3, 'lug_boot':4, 'safety':5}

with open('hw3_data/dt/data.pkl', 'rb') as f:    
    train_data , test_data , attr_vals_list , attr_names = pickle.load ( f )
    
class TreeNode:
    
    def __init__(self, attribute, data,node_id): # attribute : buying,doors,safety,...
        self.id = node_id
        self.attribute = attribute
        self.children = []
        self.parent = None
        self.n_samples = len(data)
        if(attribute not in ['acc','unacc']):
            self.attr_val_list = attr_vals_list[attr_names.index(attribute)] 
        labels = [d[-1] for d in data]
        self.values = [labels.count('unacc'),labels.count('acc')]
        if(attribute in ['buying','maint','doors']):
            self.children = [None for i in range(4)]
        elif(attribute in ['persons','lug_boot','safety']):
            self.children = [None for i in range(3)]
        elif(attribute in ['acc','unacc']):
            self.children = None # no children, it is a leaf
            
        
          
    def get_level(self): # use this to trace the edge attributes, for example up to 3rd level, 4 person 2 doors and high buying selected.. maybe implement this using dictionary
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent

        return level

   
    
    def print_tree(self): # change it to print it using graphiz or smthing
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.attribute + ' ', self.values)
        if self.children:
            for child in self.children:
                child.print_tree()
    
                    
    def add_child(self, child, i ):
        child.parent = self
        self.children[i] = child
        
    def node_shape(self):
        if (self.attribute in ['acc','unacc']):
            return 'ellipse'
        else :
            return 'box'
    def node_color(self):
        if (self.attribute == 'acc'):
            return 'green'
        elif (self.attribute =='unacc') :
            return 'red'
        else : 
            return 'black'
    
    def test_single_sample(self,sample):
        if (self.attribute in ['acc','unacc']):
            return self.attribute
        else:
            attr_index = attribute_indices[self.attribute]
            value = sample[attr_index]
            idx = self.attr_val_list.index(value)
            child = self.children[idx]
            return child.test_single_sample(sample)
        
    def prune(self):
        self.children =  None
        if(self.values[0] > self.values[1]):
            self.attribute = 'unacc'
        else:
            self.attribute = 'acc'
        
        

'''


        
        
G = pydot.Dot(graph_type="digraph")
deneme.plot_tree(G)
from IPython.display import Image, display
im = Image(G.create_png())
display(im)
G.create_png()
problem = [d for d in train_data if d[0] == 'low' and d[2] == '2' and d[3] == 'more' and d[5] == 'high']'''



'''low = [d for d in train_data if d[5] == 'low']
med = [d for d in train_data if d[5] == 'med']
high = [d for d in train_data if d[5] == 'high']

g,sp_data = info_gain(train_data,0,[['low','med','high'],['acc','unacc']])

p2 = [d for d in med if d[3] == '2' ]
p4 = [d for d in med if d[3] == '4' ]
pmore = [d for d in med if d[3] == 'more' ]

buying = [ d for d in p4 if d[0] == 'vhigh']

maint_med = [d for d in buying if d[1] == 'med']'''
