import torch
from utils import get_device
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset


def get_data_loaders(dataset: Dataset, batch_size: int):

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    data_loaders = []
    for _, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        train_dataset = Subset(dataset, train_ids)
        val_dataset = Subset(dataset, val_ids)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        valid_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

        data_loaders.append((train_loader, valid_loader))

    return data_loaders


def train(
    classifier, optimizer, criterion, scheduler, data_loaders, save_path, epochs=5
):
    """Simple training loop for a PyTorch model."""

    device, _ = get_device()
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
            correct = 0  # Reset correct for each fold
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

        train_loss_avg = sum(train_losses) / len(train_losses)
        # Evaluate on the validation set
        classifier.eval()
        validation_losses = []
        with torch.no_grad():
            for batch_idx, (features, target) in enumerate(valid_loader):
                output = classifier(features.to(device))
                valid_loss = criterion(output.to(device), target.to(device))
                # Get the label corresponding to the highest predicted probability.
                pred = output.argmax(dim=1, keepdim=True)
                # Count number of correct predictions.
                correct += pred.cpu().eq(target.view_as(pred)).sum().item()
                validation_losses.append(valid_loss.item())

        valid_loss_avg = sum(validation_losses) / len(validation_losses)

        print(
            f"Epoch: {epoch}, Fold: {fold}, Train Loss: {train_loss_avg:.6f}, Valid Loss: {valid_loss_avg:.6f}"
        )
        # Print test accuracy.
        percent = (
            100.0 * correct / len(valid_loader.dataset)
        )  # Use valid_loader.dataset instead of data_loader.dataset
        print(
            f"Validation accuracy: {correct} / {len(valid_loader.dataset)} ({percent:.0f}%)"
        )
        torch.save(classifier.state_dict(), f"{save_path}/resnet18_model.pth")
        # Reset model to training mode
        classifier.train()


def predict(classifier, data_loader):
    device, _ = get_device()
    classifier.eval()
    correct = 0
    for _, (images, target) in enumerate(data_loader):
        # Forward pass.
        output = classifier(images.to(device))
        # Get the label corresponding to the highest predicted probability.
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    percent = 100. * correct / len(data_loader.dataset)
    print(f'Test Accuracy: {correct} / {len(data_loader.dataset)} ({percent:.0f}%)')