import torch
from tqdm import tqdm

def calc_accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1)  # Fing the index of the max value, prediction
    correct = torch.eq(y_true, y_pred).sum().item() # Compare prediction with true label, get value
    accuracy = (correct / len(y_pred)) * 100
    
    return accuracy

def check_last_n(values: list, n:int = 3, delta:int = 0.5) -> bool:
    '''
    Check if the last n values in a list are within value of each other
    
    args:
    - values: list
    - n: number of values to check
    - delta: threshold
    
    return: boolean
    '''
    
    last_three = values[-n:]
    if max(last_three) - min(last_three) <= delta and len(values) >= n:
        return True
    else:
        return False


def train(model, train_loader, val_loader, epochs, criterion, optimizer, device, patience=3, min_delta=0.5):
    '''
    Train a PyTorch model with added verbosity using tqdm.
    
    args:
    - model: PyTorch model
    - train_loader: DataLoader
    - val_loader: DataLoader
    - epochs: number of epochs
    - criterion: loss function
    - optimizer: optimizer
    - device: cuda or cpu
    - patience: number of epochs to wait before early stopping
    - min_delta: minimum change in loss to qualify as an improvement, percent
    
    return: train_loss, val_loss, train_accuracy, val_accuracy
    '''
    
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_train_loss, epoch_train_acc = 0, 0
        size = len(train_loader)
        
        # Train
        model.train()
        
        # tqdm progress bar
        with tqdm(total=size, desc="Training", unit="batch") as pbar:
            for image, label in train_loader:
                image, label = image.to(device), label.to(device)  # move data to device
                
                y_logits = model(image)  # forward pass
                
                # calculate loss, acc
                loss = criterion(y_logits, label)
                acc = calc_accuracy(y_logits, label)
                
                optimizer.zero_grad()  # zero grad
                loss.backward()  # backward pass
                optimizer.step()  # update weights
                
                epoch_train_loss += loss.item()
                epoch_train_acc += acc
                
                pbar.set_postfix(loss=loss.item(), acc=acc)
                pbar.update(1)
        
        train_loss.append(epoch_train_loss / size)
        train_acc.append(epoch_train_acc / size)
        
        # Validation
        model.eval()
        epoch_val_loss, epoch_val_acc = 0, 0
        size = len(val_loader)
        
        with torch.inference_mode():
            with tqdm(total=size, desc="Validating", unit="batch") as pbar:
                for image, label in val_loader:
                    image, label = image.to(device), label.to(device)
                    y_logits = model(image)
                    loss = criterion(y_logits, label)
                    acc = calc_accuracy(y_logits, label)
                    
                    epoch_val_loss += loss.item()
                    epoch_val_acc += acc
                    
                    pbar.set_postfix(loss=loss.item(), acc=acc)
                    pbar.update(1)
        
        val_loss.append(epoch_val_loss / size)
        val_acc.append(epoch_val_acc / size)
        
        print(f"\nEpoch: {epoch + 1} | Train Loss: {train_loss[-1]:.5f} | Train Acc: {train_acc[-1]:.2f}% | Val Loss: {val_loss[-1]:.5f} | Val Acc: {val_acc[-1]:.2f}%")
        
        # Early stopping
        if check_last_n(val_loss, patience, min_delta):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    return train_loss, val_loss, train_acc, val_acc
