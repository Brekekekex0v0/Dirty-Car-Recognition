import torch
from torch import nn

num_classes = 0

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    correct, train_loss = 0, 0
    

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate accuracy
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Average loss and accuracy for this epoch
    train_loss /= len(dataloader)
    correct /= size

    # Log the training loss and accuracy
    print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")


def test_loop(dataloader, model, loss_fn, device):
    global best_accuracy  # Keep track of the highest accuracy
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    test_loss, correct = 0, 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            test_loss += loss_fn(pred, y).item()
            correct += (predicted == y).sum().item()

            # Per-class accuracy
            for label in range(num_classes):
                class_correct[label] += ((predicted == y) & (y == label)).sum().item()
                class_total[label] += (y == label).sum().item()

    
    test_loss /= num_batches
    correct /= size
    
    # Log the test loss and accuracy
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    for label in range(num_classes):
        if class_total[label] > 0:
            acc = 100 * class_correct[label] / class_total[label]
            print(f"Accuracy for class {label}: {acc:.2f}% ({class_correct[label]}/{class_total[label]})")
        else:
            print(f"No samples for class {label}")
    
    return test_loss