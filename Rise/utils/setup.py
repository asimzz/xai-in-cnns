
from torchvision import datasets ,transforms

from torch.utils.data import DataLoader

import torch

from tqdm import tqdm



def get_tranformer():

    ''' Description: funtion that provide transform for the purpose of
     data augmentation
     arg: None
     return : train_transformer  , test_transformer
     
     '''
    

    train_transform0 = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(336),                                     
    transforms.RandomHorizontalFlip(),
    
    transforms.Resize((224 ,224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    


    test_transform0 = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    

     ])
    
    return  train_transform0 , test_transform0


def give_data(train_path,test_path,train_transform,test_transform,batch_size = 32):

    root_dir_train = train_path
    dataset_train = datasets.ImageFolder(root=root_dir_train, transform=train_transform)


    root_dir_test =test_path
    dataset_test = datasets.ImageFolder(root=root_dir_test, transform=test_transform)


    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)


    return train_loader , test_loader




def get_device(mps = False):
    if mps == True:

        device = "mps:0" if torch.cuda.is_available() else "cpu" 

        mps = device =='mps:0'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cuda = device == "cuda"

    return (device, cuda)



device,_ = get_device(mps = True)


def train(model, criterion, data_loader, optimizer, num_epochs):
    """Simple training loop for a PyTorch model.""" 
    
    # Make sure model is in training mode.
    model.train()
    
    # Move model to the device (CPU or GPU).
    model.to(device)
    
    # Exponential moving average of the loss.
    ema_loss = None

    print('----- Training Loop -----')
    # Loop over epochs.
    for epoch in range(num_epochs):
        
      # Loop over data.
      for batch_idx, (features, target) in tqdm(enumerate(data_loader), leave = False):
            
          # Forward pass.
        output = model(features.to(device))
        loss = criterion(output, target.to(device))

          # Backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      # NOTE: It is important to call .item() on the loss before summing.
        if ema_loss is None:
            ema_loss = loss.item()
        else:
            ema_loss += (loss.item() - ema_loss) * 0.01 

      
      #scheduler.step()
        
      print('Epoch: {} \tLoss: {:.6f}'.format(epoch, ema_loss),)
    


    
    
def model_evaluation(data_loader,model):
    
    model.eval()
    model.to(device)
    
    total = 0

    correct = 0

    for data in data_loader:
        image_data, labels = data


        #image_data.to(mps_device)
        out =  model(image_data.to(device))
        max_values,pred_class = torch.max(out.to(device),1)

        total += labels.to(device).shape[0]

        correct +=(pred_class == labels.to(device)).sum().item()

    acuracy = (100*correct)/total

    print(f"acc  = {round(acuracy,4)}%")
