import torch


calculate_losses_avg = lambda losses: sum(losses) / len(losses)


def train(
    classifier,
    optimizer,
    criterion,
    scheduler,
    data_loaders,
    device,
    epochs,
    origin_path,
):
    """Simple training loop for a PyTorch model."""

    # Make sure model is in training mode.
    classifier.train()

    # Move model to the device (CPU or MPS or GPU).
    classifier.to(device)

    # Exponential moving average of the loss.
    train_losses = []

    print("----- Training Loop -----")
    # Loop over epochs.
    for epoch in range((epochs)):
        for fold, (train_loader, valid_loader) in enumerate((data_loaders)):
            # Loop over data.
            for _, (features, target) in enumerate(train_loader):

                # Forward pass.
                output = classifier(features.to(device))
                train_loss = criterion(output.to(device), target.to(device))
                train_losses.append(train_loss.item())

                # Backward pass.
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                scheduler.step()

            correct = 0  # Reset correct for each fold
            train_loss_avg = calculate_losses_avg(train_losses)

            validation_losses = evaluate(classifier, criterion, valid_loader, device)
            valid_loss_avg = calculate_losses_avg(validation_losses)

            print(
                f"Epoch: {epoch}, Fold: {fold}, Train Loss: {train_loss_avg:.6f}, Validation Loss: {valid_loss_avg:.6f}"
            )

            percent = 100.0 * correct / len(valid_loader.dataset)
            print(
                f"Validation accuracy: {correct} / {len(valid_loader.dataset)} ({percent:.0f}%)"
            )
            torch.save(classifier.state_dict(), f"{origin_path}/resnet18_model.pth")
            # Reset model to training mode
            classifier.train()

    return train_losses


def evaluate(classifier, criterion, data_loader, device):
    # Evaluate on the validation set
    classifier.eval()
    validation_losses = []
    with torch.no_grad():
        for _, (features, target) in enumerate(data_loader):
            output = classifier(features.to(device))
            valid_loss = criterion(output.to(device), target.to(device))
            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)
            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
            validation_losses.append(valid_loss.item())
    return correct, validation_losses


def predict(classifier, data_loader, device):
    classifier.eval()
    correct = 0
    for _, (images, target) in enumerate(data_loader):
        # Forward pass.
        output = classifier(images.to(device))
        # Get the label corresponding to the highest predicted probability.
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    percent = 100.0 * correct / len(data_loader.dataset)
    print(f"Test Accuracy: {correct} / {len(data_loader.dataset)} ({percent:.0f}%)")


# Device configuration
def get_device(mps = False):
    if mps == True:

        device = "mps:0" if torch.cuda.is_available() else "cpu" 

        mps = device =='mps:0'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cuda = device == "cuda"

    return (device, cuda)
