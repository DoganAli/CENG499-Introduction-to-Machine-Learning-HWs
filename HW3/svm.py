#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 01:07:25 2021

@author: cedgn
"""
import random
import numpy as np
from sklearn.svm import SVC
from draw import draw_svm 
from sklearn.model_selection import  cross_validate
from sklearn.metrics import confusion_matrix


# PART 1
train_data = np.load("hw3_data/linsep/train_data.npy")
train_labels = np.load("hw3_data/linsep/train_labels.npy")
C_values = [0.01, 0.1, 1, 10, 100.]
x1_min,x1_max = min(train_data[:,0]) , max(train_data[:,0])
x2_min,x2_max = min(train_data[:,1]) , max(train_data[:,1])
for i,c in enumerate(C_values):   
    clf = SVC(C=c, kernel ='linear')
    clf.fit(train_data,train_labels)
    target = 'svm/part_1/c_'+str(i)
    draw_svm(clf,train_data,train_labels,x1_min,x1_max,x2_min,x2_max,target)



# PART 2 

train_data_2 = np.load("hw3_data/nonlinsep/train_data.npy")
train_labels_2 = np.load("hw3_data/nonlinsep/train_labels.npy")
x1_min,x1_max = min(train_data_2[:,0]) , max(train_data_2[:,0])
x2_min,x2_max = min(train_data_2[:,1]) , max(train_data_2[:,1])
kernels = ['linear','rbf','poly','sigmoid']

for kernel in kernels:
    clf = SVC(kernel = kernel)
    clf.fit(train_data_2,train_labels_2)
    target ='svm/part_2/'+kernel
    draw_svm(clf,train_data_2,train_labels_2,x1_min,x1_max,x2_min,x2_max,target)



# PART 3 
train_data_3 = np.load("hw3_data/catdog/train_data.npy") 
train_labels_3 = np.load("hw3_data/catdog/train_labels.npy") 
test_data_3 = np.load("hw3_data/catdog/test_data.npy") 
test_labels_3 = np.load("hw3_data/catdog/test_labels.npy")

max_value = train_data_3.max() # same as test_data_3.max(), 255
train_data_3 = train_data_3 / max_value 
test_data_3 = test_data_3 / max_value


#clf = GridSearchCV(SVC(), parameters)
#clf.fit(train_data_3,train_labels_3)
'''
clf = SVC()
clf.fit(train_data_3,train_labels_3)
preds = clf.predict(test_data_3)
np.mean(preds == test_labels_3)


clf = SVC()
cv_results = cross_validate(clf,train_data_3,train_labels_3,cv = 5 )
'''
cv_results_grid = {}

for kernel in ['linear', 'poly', 'rbf','sigmoid']:
    for c in [0.01, 0.1, 1, 10, 100.] :
        if kernel == 'linear' :
            clf = SVC(C=c , kernel = kernel)
            cv_results = cross_validate(clf,train_data_3,train_labels_3,cv=5)
            print(kernel,c,cv_results['test_score'],cv_results['fit_time'],cv_results['score_time'])
            key = kernel + str(c)
            cv_results_grid[key] = cv_results['test_score']
        else :
            for gamma in [0.00001,0.0001,0.001,0.01,0.1,1] :
                clf = SVC(C=c , kernel = kernel , gamma = gamma )
                cv_results = cross_validate(clf,train_data_3,train_labels_3,cv = 5 )
                print(kernel,c,gamma,cv_results['test_score'],cv_results['fit_time'],cv_results['score_time'])
                key = kernel +'-' + str(c) + '-' + str(gamma)
                cv_results_grid[key] = cv_results['test_score']


f = open('log.txt','a+')

for kernel in ['linear', 'poly', 'rbf','sigmoid']:
    for c in [0.01, 0.1, 1, 10, 100.] :
      if kernel == 'linear' :
            key = kernel + str(c)
            cv_results = cv_results_grid[key] 
            log = key + ' - ' + str(round(np.mean(cv_results),2) ) +'\n'
            f.write(log)
      else :
            for gamma in [0.00001,0.0001,0.001,0.01,0.1,1] :
                key = kernel +'-' + str(c) + '-' + str(gamma)
                cv_results = cv_results_grid[key] 
                log = key + ' - ' + str(round(np.mean(cv_results),2) ) + '\n'
                f.write(log)
         
            
# best hyper parameter test
clf_best = SVC(C = 100, kernel = 'rbf',gamma = 0.01)
clf_best.fit(train_data_3,train_labels_3)
preds = clf_best.predict(test_data_3)
print('accuracy on testset : ', np.mean(preds==test_labels_3))



# PART 4
    
train_data_4 = np.load("hw3_data/catdogimba/train_data.npy") 
train_labels_4 = np.load("hw3_data/catdogimba/train_labels.npy") 
test_data_4 = np.load("hw3_data/catdogimba/test_data.npy") 
test_labels_4 = np.load("hw3_data/catdogimba/test_labels.npy")

max_value = train_data_4.max() # same as test_data_3.max(), 255
train_data_4 = train_data_4 / max_value 
test_data_4 = test_data_4 / max_value

clf = SVC(C=1.0 , kernel='rbf')
clf.fit(train_data_4,train_labels_4)
predictions = clf.predict(test_data_4)
accuracy = np.mean(predictions == test_labels_4)
print('without handling the imbalance problem: accuracy on test set : ', round(accuracy,2))

print(confusion_matrix(test_labels_4,predictions))


# OVER SAMPLE #

minority = np.where(train_labels_4 == 0)
minority_class = train_data_4[minority]
minority_class_labels = train_labels_4[minority]
over_sampled_data = train_data_4
over_sampled_labels = train_labels_4 

for i in range(4):
    over_sampled_data = np.vstack((over_sampled_data,minority_class))
    over_sampled_labels = np.hstack((over_sampled_labels,minority_class_labels))


clf = SVC(C=1.0 , kernel='rbf')
clf.fit(over_sampled_data,over_sampled_labels)
predictions = clf.predict(test_data_4)
accuracy = np.mean(predictions == test_labels_4)
print('oversampling: accuracy on test set : ', round(accuracy,2))

print(confusion_matrix(test_labels_4,predictions))
#####



# UNDER SAMPLE # 
minority = np.where(train_labels_4 == 0)
minority_class = train_data_4[minority]
minority_class_labels = train_labels_4[minority]
majority = list(np.where(train_labels_4 == 1 )[0])
random.shuffle(majority)
reduced_majority = majority[:215]
random_under_sampled_data = train_data_4[reduced_majority]
random_under_sampled_labels = train_labels_4[reduced_majority]
under_sampled_data = np.vstack((random_under_sampled_data,minority_class))
under_sampled_labels = np.hstack((random_under_sampled_labels,minority_class_labels))

clf = SVC(C=1.0 , kernel='rbf')
clf.fit(under_sampled_data,under_sampled_labels)
predictions = clf.predict(test_data_4)
accuracy = np.mean(predictions == test_labels_4)
print('undersampling: accuracy on test set : ', round(accuracy,2))

print(confusion_matrix(test_labels_4,predictions))

#####

# class_weight # 

clf = SVC(C=1.0 , kernel='rbf',class_weight = 'balanced' )
clf.fit(train_data_4,train_labels_4)
predictions = clf.predict(test_data_4)
accuracy = np.mean(predictions == test_labels_4)
print('class_weight balance: accuracy on test set : ', round(accuracy,2))

print(confusion_matrix(test_labels_4,predictions))

###

