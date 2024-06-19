from utils import train
from setup import *
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
from torchvision import models
from models import model1
import os


# Data loading

data_path = "/Users/atoukoffikougbanhoun/Desktop/AMMI/CV projects/XAI_Project1/xai-in-cnns/dataset/data"
train_path = data_path+"/train" 
test_path = data_path+"/test"



train_transform0 , test_transform0 = get_tranformer()

root_dir_train = train_path
dataset_train = datasets.ImageFolder(root=root_dir_train, transform=train_transform0)


root_dir_test =test_path
dataset_test = datasets.ImageFolder(root=root_dir_test, transform=test_transform0)



batch_size = 32
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)



display_(train_loader,dataset_train=dataset_train)

print(" hello")

# model 

model = models.resnet18(pretrained=True)

    # Freeze the base layers
for param in model.parameters():
    param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 16)




# training

# criterion and optimizer

# optimizer and loss
Criterion = nn.CrossEntropyLoss()

learn_rate = 1e-3
optimizer = torch.optim.SGD(model1.parameters(),lr=learn_rate)




# training
num_epochs = 10
train(model1,Criterion, train_loader, optimizer, num_epochs=num_epochs)




#grad_cam



filepaths = os.listdir("/Users/atoukoffikougbanhoun/Desktop/AMMI/CV projects/XAI_Project1/xai-in-cnns/dataset/data/test/california_poppy")
image_path = "/Users/atoukoffikougbanhoun/Desktop/AMMI/CV projects/XAI_Project1/xai-in-cnns/dataset/data/test/california_poppy/"+filepaths[20]
grad_cam = GradCAM(model1)
visualize_results(image_path, model1, grad_cam,dataset_train=dataset_train)
