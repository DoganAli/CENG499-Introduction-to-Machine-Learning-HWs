import torch.nn as nn

class NET(nn.Module):
    
    def __init__(self,num_hidden_neurons_1 = 0,num_hidden_neurons_2 = 0):
        super(NET, self).__init__()
        self.h1 =num_hidden_neurons_1
        self.h2 = num_hidden_neurons_2
        if(num_hidden_neurons_1 > 0 and num_hidden_neurons_2 > 0  ): 
            self.fc1 = nn.Linear(1*32*64, num_hidden_neurons_1)
            self.fc2 = nn.Linear(num_hidden_neurons_1, num_hidden_neurons_2)
            self.fc3 = nn.Linear(num_hidden_neurons_2,100)
            self.drop1  = nn.Dropout(p=0.75)
            self.drop2  = nn.Dropout(p=0.5)
            self.bnorm1 = nn.BatchNorm1d(num_hidden_neurons_1)
            self.bnorm2 = nn.BatchNorm1d(num_hidden_neurons_2)
            
        elif(num_hidden_neurons_1 > 0 and num_hidden_neurons_2 <= 0):        
            self.fc1 = nn.Linear(1*32*64, num_hidden_neurons_1)
            self.fc2 = nn.Linear(num_hidden_neurons_1,100)
            self.drop1  = nn.Dropout(p=0.5)
            self.bnorm1 = nn.BatchNorm1d(num_hidden_neurons_1)
        else :
            self.fc1 = nn.Linear(1*32*64, 100)
            
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x ):
        x = x.view(x.size(0), -1 )
        
        if(self.h1 > 0 and self.h2 > 0  ): 
            x = self.bnorm1(self.fc1(x))
            x = self.relu(x)
            x = self.drop1(x)
            x = self.bnorm2(self.fc2(x))
            x = self.relu(x)
            x = self.drop2(x)
            x = self.fc3(x)
        elif(self.h1 > 0 and self.h2 <= 0):   
            x = self.bnorm1(self.fc1(x))
            x = self.relu(x)
            x = self.drop1(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
            
        return x 