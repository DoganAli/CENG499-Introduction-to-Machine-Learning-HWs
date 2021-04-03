import torch
from torch.utils.data import  DataLoader

import torchvision.transforms as T
## below is the model
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import MnistDataset
from model import NET
from cnn import CNN
import os 

torch . manual_seed (1234)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(0.2978,0.1777)
    ])

### DATASET EXTRACTION
train_dataset = MnistDataset('data','train', transforms )
loss_fun = nn.CrossEntropyLoss()

def accuracy_test(model,train_data_loader,val_data_loader,lr):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images,labels in train_data_loader:
          out = model(images)
          _, predicted_labels = torch.max(out,1)
          correct += (predicted_labels == labels).sum()
          total += labels.size(0) 
        train_acc = ((correct)/(total+1))
        print('Train Percent correct: %.4f ' %train_acc)
        
        
        correct = 0
        total = 0 
        for images, labels in val_data_loader :
            out = model(images)
            _, predicted_labels = torch.max(out,1)
            correct += (predicted_labels == labels).sum()
            total += labels.size(0) 
        val_acc = ((correct)/(total+1))
        print('Validation Percent correct: %.4f ' %val_acc)
        print('lr : ' + str(lr))
        print(model)
        
        return train_acc,val_acc
     
def non_decreasing(Losses):
    return all(x<=y for x, y in zip(Losses, Losses[1:]))


def train(model, optimizer, loss_fun ,  train_data_loader,val_data_loader,lr,h1,h2, num_epochs, calculate_val_loss = True ):
    model.train()
    loss_list = []
    val_loss_list =[]
    n_epoch = num_epochs
    for epoch in range(num_epochs) :
        if(len(val_loss_list ) > 3 and non_decreasing(val_loss_list[-3:])) :
            n_epoch = epoch
            break
        for i, (images,labels) in enumerate(train_data_loader):
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fun(pred, labels)
            loss.backward()
            optimizer.step()
            
            if( (i+1)%5 == 0  ) :
                print('Epoch [%d/%d], Step [%d/40], Loss: %.4f'
                %(epoch+1, num_epochs, i+1, loss.item()))
                if(i+1) % 40 == 0 :
                    loss_list.append(loss.item())
           
            
        if(calculate_val_loss):
            for val_images,val_labels in val_data_loader:
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_images)
                    val_loss_list.append(loss_fun(val_pred,val_labels))
                
    plt.plot(loss_list, label = 'train loss')
    if(calculate_val_loss and h1 == 0  ):
        plt.plot(val_loss_list, label = 'validation loss')
        plt.title("1-layer(0-hidden-layer) with lr:" + str(lr)) 
        plt.legend() 
        plt.show()
    
    elif(calculate_val_loss and h2 == 0 ):
        plt.plot(val_loss_list, label = 'validation loss')
        plt.title("2-layer(1-hidden-layer) with HS:" + str(h1) + " and lr:" + str(lr)) 
        plt.legend() 
        plt.show()
    elif(calculate_val_loss ):
        plt.plot(val_loss_list, label = 'validation loss')
        plt.title("3-layer(2-hidden-layer) with HS:" + str(h1) + '-'+str(h2) +  " and lr:" + str(lr)) 
        plt.legend() 
        plt.show()
    return n_epoch
    
    



def hyper_parameter_tune(train_dataset,a):
    '''
    Parameters
    ----------
    train_dataset : MnistDataset
        training data set.
    a : int
        number of layers,  it gets 1,2,3 for three difeerent networks.

    Returns
    -------
    models : Dictionary
        dictionary of models that were trained, the keys are added regarding the parameters
        e.g, model1 = Models['128-0.01'] where it represents 'hidden layer 1 size ' - 'learning rate'.
    x_layer_val : 
        dict of validation set accuracies.
    x_layer_train :
        dict of training set accuracies.

    '''
    learning_rates = [0.03]#,0.01,0.003,0.001,0.0003,0.0001]
    h1 = [256]#,512,1024]
    h2 = [128]#,256,512,1024]

    one_layer_val = {}
    one_layer_train = {}
    two_layer_val = {}
    two_layer_train={}
    three_layer_val={}
    three_layer_train={}
    
    models = {}
    # 1-layer network hyperparameters test
    if(a == 1 ):
        
        for learning_rate in learning_rates:
            train_set , val_set = torch.utils.data.random_split(train_dataset, [8000,2000]) 
            train_data_loader = DataLoader(train_set, batch_size = 200 , shuffle = True)
            val_data_loader = DataLoader(val_set, batch_size = 2000)
            model = NET()
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)   
            n_epoch = train(model,optimizer, loss_fun, train_data_loader,val_data_loader,learning_rate,0,0, num_epochs = 150)
            train_acc, val_acc = accuracy_test(model,train_data_loader,val_data_loader,learning_rate)
            one_layer_val[str(learning_rate)] = round(float(val_acc),4)
            one_layer_train[str(learning_rate)]=round(float(train_acc),4)
            save(train_acc, val_acc,learning_rate,0,0,n_epoch)
            key =str(learning_rate)
            models[key] = model
        return models,one_layer_val, one_layer_train
    
    elif(a == 2 ):
        for learning_rate in learning_rates:
            for hidden_1 in h1 :
                train_set , val_set = torch.utils.data.random_split(train_dataset, [8000,2000]) 
                train_data_loader = DataLoader(train_set, batch_size = 400 , shuffle = True)
                val_data_loader = DataLoader(val_set, batch_size = 2000)
                model = NET(hidden_1)
                optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 
                n_epoch = train(model,optimizer, loss_fun, train_data_loader,val_data_loader,learning_rate,hidden_1,0, num_epochs = 150)
                train_acc, val_acc = accuracy_test(model,train_data_loader,val_data_loader,learning_rate)
                key = str(hidden_1)+'-'+str(learning_rate)
                two_layer_val[key] = round(float(val_acc),4)
                two_layer_train[key] = round(float(train_acc),4)
                models[key] = model 
                save(train_acc, val_acc,learning_rate,hidden_1,0,n_epoch)
        return models, two_layer_val, two_layer_train
        
    elif(a == 3) :       
        for learning_rate in learning_rates :
            for hidden_1 in h1 :
                for hidden_2 in h2 :
                    train_set , val_set = torch.utils.data.random_split(train_dataset, [8000,2000]) 
                    train_data_loader = DataLoader(train_set, batch_size = 200 , shuffle = True)
                    val_data_loader = DataLoader(val_set, batch_size = 2000)
                    model = NET(hidden_1,hidden_2)
                    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)   
                    n_epoch = train(model,optimizer, loss_fun, train_data_loader,val_data_loader,learning_rate,hidden_1,hidden_2, num_epochs = 150)
                    train_acc, val_acc = accuracy_test(model,train_data_loader,val_data_loader,learning_rate)
                    key = str(hidden_1)+'-'+ str(hidden_2)+'-'+str(learning_rate)
                    three_layer_val[key] = round(float(val_acc),4)
                    three_layer_train[key] = round(float(train_acc),4)
                    models[key] = model 
                    save(train_acc, val_acc,learning_rate,hidden_1,hidden_2,n_epoch)
        return models, three_layer_val, three_layer_train
    
    
    
    
def save(train_acc, val_acc,learning_rate,hidden_1,hidden_2,n_epoch) :
    file = open('log_3_layer.txt','a')
    if hidden_2 > 0 :
        log = str(train_acc) +' - '+ str(val_acc) +' - '+ str(learning_rate)+' - '+ str(hidden_1) +' - '+ str(hidden_2) +' - '+ str(n_epoch)
    else:
        log = str(train_acc) +' - '+ str(val_acc) +' - '+ str(learning_rate)+' - '+ str(hidden_1) +' - '+ str(n_epoch)
    file.write(log + '\n')
    

def sanity_check():
    model = NET(1024,128)
    t = DataLoader(train_dataset, batch_size = 10000 , shuffle = True)
    for images,labels in t:
        pred = model(images)
        loss = loss_fun(pred, labels)
        loss.backward()  
    print("loss without training : " + str(loss.item()))
   
    accuracy_test(model, t, t, 0) # to see the accuracy with randomized, untrained model


def add_zeros():
    images_path = os.path.join('data', 'test' )
    file = open(os.path.join(images_path,'labels.txt'), 'w+')
    for i in range(0,10000):
        file.write(str(i) + '.png '+  str(0) +'\n' )
## TEST PART ##
def test(model):
    add_zeros()
    test_dataset = MnistDataset('data','test', transforms) 
    test_data_loader = DataLoader(test_dataset, batch_size = 10000)
       
    correct = 0
    total = 0
    for images,labels in test_data_loader:
       out = model(images)
       _, predicted_labels = torch.max(out,1)
       correct += (predicted_labels == labels).sum()
       total += labels.size(0)
          
        
    file = open('labels.txt','w+')
    for i in range(0,10000):
        file.write(str(i)+'.png '+ str(int(predicted_labels[i])) + '\n')
            
    file.close()


def train_run_example():
    train_set , val_set = torch.utils.data.random_split(train_dataset, [8000,2000]) 
    train_data_loader = DataLoader(train_set, batch_size = 200 , shuffle = True)
    val_data_loader = DataLoader(val_set, batch_size = 2000)
    model = NET(1024,128)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)   
    train(model,optimizer, loss_fun, train_data_loader,val_data_loader,0.001,1024,128, num_epochs = 150)
    accuracy_test(model,train_data_loader,val_data_loader,0.001)
    test(model)
    return model






