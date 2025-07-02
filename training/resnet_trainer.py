import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import models, transforms
from torch.utils.data import DataLoader
from processing.MelSpectrogramDataset import MelSpectrogramDataset

# Training function
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,batch_size,num_epochs=40):
    since = time.time()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if scheduler is None:
        scheduler = ExponentialLR(optimizer, gamma=0.95)

    device = next(model.parameters()).device
    model.to(device)
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            n_batches = dataset_sizes[phase] // batch_size
            for it, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print(
                    f"Epoch: {epoch+1}/{num_epochs} Iter: {it+1}/{n_batches}",
                    end="\r",
                    flush=True,
                )
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            if phase == 'validation':
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())
                scheduler.step(epoch_loss)  # Reduce LR if val_loss does not improve
                last_lr = scheduler.get_last_lr()
                print(f"Epoch {epoch+1}, Learning Rate: {last_lr}")
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accuracies,
        "val_acc": val_accuracies
    }
    with open("Resnet/SGD_training_history.json", "w") as f:
        json.dump(history, f)
    torch.save(best_model_wts, "Resnet/SGD_best_model.pth")
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model