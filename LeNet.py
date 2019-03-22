import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')


batch_size =64
train_dataset = datasets.FashionMNIST(root='./FashionMNIST/',
                               train=True,
                               transform=transforms.ToTensor())

test_dataset = datasets.FashionMNIST(root='./FashionMNIST/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#In original paper input size is 32x32 , but we have taken input size as 28x28

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size=5)
        self.conv2 = nn.Conv2d(6,16,kernel_size=3)        
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)  # original paper there is average pooling
        self.l1 = nn.Linear(400,120)
        self.l2 = nn.Linear(120,10)
        
        
    def forward(self,x):
        
        x = self.pooling(self.conv1(x))
        x = self.pooling(self.conv2(x))
        
        x = x.view(x.size(0),-1)
        x = self.l1(x)
        
        x = self.l2(x)
        
        return F.log_softmax(x)

model = LeNet()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(10):
    for i ,(data,target) in enumerate(train_loader):
        data,target = Variable(data),Variable(target)
        optimizer.zero_grad()
        y_pred = model.forward(data)
        loss = criterion(y_pred,target)
        
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))

test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = Variable(data), Variable(target)
    output = model.forward(data)
    # sum up batch loss
    test_loss += criterion(output, target).item()
    # get the index of the max
    pred = output.data.max(dim=1,keepdim=True)[1]
    
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))