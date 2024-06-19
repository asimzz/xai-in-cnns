
import torch.nn as nn 

from torchvision import models





def model1():

    model = models.resnet18(pretrained=True)

    # Freeze the base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 16)

    return model

model1 = model1()
print(model1)