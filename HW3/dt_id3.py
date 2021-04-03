#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:22:59 2021

@author: cedgn
"""
from tree_node import TreeNode,ID
from dt import info_gain, gain_ratio, gini, avg_gini_index, chi_squared_test
import pickle
import pydot
from IPython.display import Image, display

with open('hw3_data/dt/data.pkl', 'rb') as f:    
    train_data , test_data , attr_vals_list , attr_names = pickle.load ( f )

attribute_indices = {'buying':0, 'maint':1, 'doors':2, 'persons':3, 'lug_boot':4, 'safety':5}

def id3( data , attributes, attr_func, attr_vals_list, min_max,id_g ) :
     # attributes : kalan attributelerin isimleri 
     
     labels = [d[-1] for d in data]
     
     if(len(set(labels)) == 1 ):
         # only one label left, so it is a leaf
         return TreeNode(labels[0],data,id_g.get_id()) #leaf
     
     if(len(attributes) == 0 ):
         # return the most common label as root
         if(labels.count('acc') >= labels.count('unacc')):
             return TreeNode('acc',data,id_g.get_id())
         else:
             return TreeNode('unacc',data,id_g.get_id())
        
     gain_values = {} # gini index also included
     buckets = {}
     for attr in attributes:
         attr_index = attribute_indices[attr]
         gain, splitted_data = attr_func(data, attr_index, attr_vals_list)
         gain_values[attr] = gain
         buckets[attr] = splitted_data
    
     selected_attribute = min_max(gain_values,key = gain_values.get)
     selected_split = buckets[selected_attribute]
     #gain_value = gain_values[selected_attribute]
     
     root = TreeNode(selected_attribute,data,id_g.get_id())
     remained_attributes = attributes.copy()
     remained_attributes.remove(selected_attribute)
     for j,subset in enumerate(selected_split):
         if(len(subset) > 0):
             child = id3(subset,remained_attributes,attr_func,attr_vals_list,min_max,id_g)
             root.add_child(child,j)
         else:
             if(labels.count('acc') >= labels.count('unacc')):
                 root.add_child(TreeNode('acc',subset,id_g.get_id()),j)
             else:
                 root.add_child(TreeNode('unacc',subset,id_g.get_id()),j)
         
     return root
        
def test(test_data,dt):
    test_set = [d[:-1] for d in test_data]
    labels = [d[-1] for d in test_data]
    n = len(test_data)
    n_correct = 0 
    for i,sample in enumerate(test_data):
        label = labels[i]
        prediction = dt.test_single_sample(sample)
        if(label == prediction):
            n_correct += 1 
        
    print('accuracy : ',n_correct,'/', n,' : ', n_correct/n)
    return n_correct/n

def plot_tree( root, G ):
        node = pydot.Node(root.id,label = root.attribute + str(root.values),shape = root.node_shape(),color = root.node_color() )
        G.add_node(node)
        if(root.children):
            for c_idx,child in enumerate(root.children):
                if (child):
                    node = pydot.Node(child.id, label = child.attribute,shape = child.node_shape() ,color = child.node_color())
                    G.add_node(node)
                    idx =attribute_indices[root.attribute]
                    att_val = attr_vals_list[idx][c_idx]
                    edge = pydot.Edge(root.id,child.id,label = att_val)
                    G.add_edge(edge)
                    plot_tree(child,G)   
    
def create_graph(dt):
    G = pydot.Dot(graph_type = 'digraph')
    plot_tree(dt,G)
    im = Image(G.create_png())
    display(im)
    G.create_png()
    
    
ig = ID()
dt_info = id3(train_data,attr_names,info_gain,attr_vals_list, max ,ig )
dt_gain_ratio = id3(train_data,attr_names,gain_ratio,attr_vals_list, max ,ig )
dt_avg_gini = id3(train_data,attr_names,avg_gini_index,attr_vals_list, min ,ig )



test(test_data,dt_info)
test(test_data,dt_gain_ratio)
test(test_data,dt_avg_gini)

create_graph(dt_info)
create_graph(dt_gain_ratio)
create_graph(dt_avg_gini)

