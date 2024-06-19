import torch
import numpy as np


calculate_losses_avg = lambda losses: sum(losses) / len(losses)


def train(
    classifier,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    val_loader,
    device,
    epochs,
):
    classifier.train()
    classifier.to(device)
    train_losses = []

    print("----- Training Loop -----")
    for epoch in range(epochs):
        for _, (images, target) in enumerate(train_loader):

            # Forward pass.
            output = classifier(images.to(device))
            train_loss = criterion(output.to(device), target.to(device))
            train_losses.append(train_loss.item())

            # Backward pass.
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss_avg = np.mean(train_losses)
        # Evaluate on the validation set
        correct, val_losses = evaluate(classifier, criterion, val_loader, device)
        valid_loss_avg = np.mean(val_losses)

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss_avg:.6f}, Valid Loss: {valid_loss_avg:.6f}"
        )
        percent = (
            100.0 * correct / len(val_loader.dataset)
        )  
        print(
            f"Validation accuracy: {correct} / {len(val_loader.dataset)} ({percent:.0f}%)"
        )
        # Reset model to training mode
        classifier.train()

    return train_losses, val_losses


def evaluate(classifier, criterion, data_loader, device):
    # Evaluate on the validation set
    classifier.eval()
    validation_losses = []
    with torch.no_grad():
        correct = 0
        for _, (images, target) in enumerate(data_loader):
            output = classifier(images.to(device))
            valid_loss = criterion(output.to(device), target.to(device))
            pred = output.argmax(dim=1, keepdim=True)
            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
            validation_losses.append(valid_loss.item())
    return correct, validation_losses


def predict(classifier, data_loader, device):
    classifier.eval()
    correct = 0
    for _, (images, target) in enumerate(data_loader):
        output = classifier(images.to(device))
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    percent = 100.0 * correct / len(data_loader.dataset)
    print(f"Test Accuracy: {correct} / {len(data_loader.dataset)} ({percent:.0f}%)")


# Device configuration
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cuda = device == "cuda"

    return (device, cuda)
