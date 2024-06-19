import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def train_model(dataloaders, dataset_sizes, device, num_epochs=5):
    model = models.inception_v3(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 16)  # Adjust this to match the number of classes
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
    
            running_loss = 0.0
            running_corrects = 0
    
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, aux_outputs = outputs
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train' and isinstance(outputs, tuple):
                        loss += 0.4 * criterion(aux_outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
    
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    torch.save(model.state_dict(), 'inception_flower_classification_model.pth')
    print("Training complete!")
    return model
