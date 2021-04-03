
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as T
## below is the model
import torch.nn as nn
import matplotlib.pyplot as plt



class MnistDataset(Dataset):
    def __init__(self, dataset_path, split, transforms) :
        images_path = os.path.join(dataset_path, split )
        self.data = []
        with open(os.path.join(images_path,'labels.txt'), 'r')   as f :
            for line in f:
                image_name, label = line.split() 
                image_path = os.path.join(images_path , image_name )
                label = int(label)
                self.data.append((image_path, label))
            
        self.transforms = transforms
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = self.data[index][0]
        label = self.data[index][1]
        image = Image.open(image_path)
        image = self.transforms(image)
        return image , label
        

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.fc1 = nn.Linear(2600, 1024)
        self.fc2 = nn.Linear(1024,128)
        self.fc3 = nn.Linear(128, 100)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.drop  = nn.Dropout(p=0.75)
        self.drop2  = nn.Dropout(p=0.5)
        self.conv_bn = nn.BatchNorm2d(20)
        self.conv2_bn = nn.BatchNorm2d(40)
        self.fc_bn = nn.BatchNorm1d(1024)
        self.fc2_bn = nn.BatchNorm1d(128)
    def forward(self, x ):
        x = self.pool(self.conv_bn(self.relu(self.conv1(x))))        
        x = self.pool(self.conv2_bn(self.relu(self.conv2(x))))
        x = x.view(-1, 2600)
        x = self.fc_bn(self.relu(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2_bn(self.relu(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x
    

'''
def train(model, optimizer, loss_fun ,  train_data_loader, num_epochs):
    model.train()
    loss_list = []
    for epoch in range(num_epochs) :
        for i, (images,labels) in enumerate(train_data_loader):
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fun(pred, labels)
            loss.backward()
            optimizer.step()
                
            
            print('Epoch [%d/%d], Step [%d/80], Loss: %.4f'
            %(epoch+1, num_epochs, i+1, loss.item()))
            loss_list.append(loss.item())
                
    plt.plot(loss_list)
    
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([
    T.ToTensor() ,
    T.Normalize(0.2978,0.1777)])


train_dataset = MnistDataset('data','train', transforms )
train_set , val_set = torch.utils.data.random_split(train_dataset, [9000,1000]) 
train_data_loader = DataLoader(train_set, batch_size = 256 , shuffle = True)
val_data_loader = DataLoader(val_set, batch_size = 100)
model = CNN()

loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)



train(model,optimizer, loss_fun, train_data_loader, num_epochs = 30 )


correct = 0
total = 0
model.eval()
for images,labels in train_data_loader:
  out = model(images)
  _, predicted_labels = torch.max(out,1)
  correct += (predicted_labels == labels).sum()
  total += labels.size(0)

print('Train Percent correct: %.3f %%' %((100*correct)/(total+1)))


correct = 0
total = 0 
for images, labels in val_data_loader :
    out = model(images)
    _, predicted_labels = torch.max(out,1)
    correct += (predicted_labels == labels).sum()
    total += labels.size(0)



print('Validation Percent correct: %.3f %%' %((100*correct)/(total+1)))








## TEST PART ##
test_dataset = MnistDataset('data','test', transforms) 

test_data_loader = DataLoader(test_dataset, batch_size = 10000)

correct = 0
total = 0
for images,labels in test_data_loader:
  out = model(images)
  _, predicted_labels = torch.max(out,1)
  correct += (predicted_labels == labels).sum()
  total += labels.size(0)
  

print('test Percent correct: %.3f %%' %((100*correct)/(total+1)))





len(predicted_labels)
file = open('labels.txt','w+')
for i in range(0,10000):
    file.write(str(i)+'.png '+ str(int(predicted_labels[i])) + '\n')
    
file.close()
    '''