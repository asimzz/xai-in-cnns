import torch

from tqdm import tqdm


# scheduler = lr_scheduler.OneCycleLR(
#                                     optimizer,
#                                     max_lr=0.001,
#                                     epochs=30,
#                                     steps_per_epoch=int(len(train_loader) / batch_size),
#                                     pct_start=0.1,
#                                     anneal_strategy='cos',
#                                     final_div_factor=10**5)



# Device configuration
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
      for batch_idx, (features, target) in tqdm(enumerate(data_loader),desc="Evoluation", leave = False):
            
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



