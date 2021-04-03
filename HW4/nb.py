import numpy as np 
import time


def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    vocab = []
    for sample in data:
        for word in sample:
            if word not in vocab:
                vocab.append(word)
    return set(vocab)


def train(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta, pi. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all of the words in vocab and the values are their estimated probabilities.
             pi is a dictionary. Its keys are class names and values are their probabilities.
    """

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    label_set = list(set(train_labels))
    pi = {label:0 for label in  label_set }
    theta = {}
    
    for label in label_set:
        pi[label] = np.sum(train_labels == label) / len(train_labels)

    theta_c = {}
    for c in label_set:
        c_indices = np.array(train_labels == c)
        c_data = train_data[c_indices]
        theta_cj = {}
        for j in vocab:
            nom = sum([s.count(j) for s in c_data])
            dinom = sum([len(s) for s in c_data ])
            theta_cj[j] = (nom+1) / (dinom+len(vocab))
        
        theta_c[c] = theta_cj 
        
    theta = theta_c
            
        
    

    return theta, pi


def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    
    classes = pi.keys()
    scores = []
    
    for test_sample in test_data:
        score_test_sample = []
        for c in classes:
            pi_c = np.log(pi[c])
            second_part = 0 
            for word in test_sample:
                if word in vocab:
                    second_part +=np.log(theta[c][word])
            score = pi_c + second_part
            
            score_test_sample.append((score,c))
        scores.append(score_test_sample)
            
                     
    return scores

def test_accuracy(theta, pi, vocab, test_data, test_labels):
    scores = test(theta, pi, vocab, test_data)
    predicted_labels = []
    for score in scores:
        _ , label = max((prob,cls) for prob,cls in score)
        predicted_labels.append(label)
        
    predicted_labels = np.array(predicted_labels)
    
    accuracy = np.mean(predicted_labels == np.array(test_labels))
    correct = np.sum(predicted_labels==np.array(test_labels))
    total = len(test_labels)
    print('accuracy : ',accuracy,' (',correct,'/',total,')')
    
    return accuracy


def get_data(filename):
    f = open(filename,'r')
    data = []
    for line in f:
        data.append(line)
        
    return data
def split_data(data):
    splitted_data = []
    for sample in data:
        new_sample = sample.split(' ')
        splitted_data.append(new_sample)
    return splitted_data
    

def news_classify():
    train_data = get_data('hw4_data/news/train_data.txt')
    train_labels = get_data('hw4_data/news/train_labels.txt')
    test_data = get_data('hw4_data/news/test_data.txt')
    test_labels = get_data('hw4_data/news/test_labels.txt')
    
    
    train_data = split_data(train_data)
    test_data = split_data(test_data)
    
    vocab = vocabulary(train_data)
    theta,pi = train(train_data,train_labels,vocab)
    acc = test_accuracy(theta, pi, vocab, test_data, test_labels)
    
    return acc

''' 
start = time.time()
news_classify()
print("total time : ",time.time()-start)
'''
