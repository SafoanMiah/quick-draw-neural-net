import os
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
from scipy.ndimage import rotate

import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

IMAGES_DIR = "numpy_bitmap/"

cat_a = os.listdir(IMAGES_DIR)
print(f'Total categories: {len(cat_a)}')

cat_s = cat_a[12:22]
print(f'Sample categories: {cat_s}')

bmap_split = pd.DataFrame()
bmap = pd.DataFrame()
bitmaps = []

for category in cat_s:
    data = pd.DataFrame(np.load(IMAGES_DIR + category))
    data["category"] = category
    bitmaps.append(data)
    
bmap_split = pd.concat(bitmaps, ignore_index=True)
bmap_split.sample(5)

bmap["y"] = bmap_split["category"].apply(lambda x: x.split(".")[0])
bmap["X"] = bmap_split.iloc[:, :-1].apply(lambda x: np.array(x).reshape(28,28), axis=1)

def show_images(df, n_images, category=None):
    fig_h = 1.5 * (n_images//10)
    fig, axs = plt.subplots(n_images//10, 10, figsize=(12, fig_h))
    axs = axs.flatten()
    
    if not category:
        sample = df.sample(n_images)
    else:
        sample = df[df['y'] == category].sample(n_images)
    
    label = sample['y'].values
    image = sample['X'].values
    
    for n in range(n_images):
        axs[n].imshow(image[n])
        axs[n].set_title(label[n])
        axs[n].axis('off')
    
    plt.tight_layout()
    plt.show()

show_images(bmap, 30)

def category_heatmap(df):
    categories = df['y'].unique()
    n_cat = len(categories)
    
    fig_h = 1.5 * (n_cat//5)
    fig, axs = plt.subplots(n_cat//5, 5, figsize=(7, fig_h))
    axs = axs.flatten()
    
    for n in range(n_cat):
        arrs = df[df['y'] == categories[n]]['X']
        heatmap = np.stack(arrs).sum(axis=0)
      
        axs[n].imshow(heatmap)
        axs[n].set_title(categories[n])
        axs[n].axis('off')
        
    plt.tight_layout()
    plt.show()

category_heatmap(bmap)

plt.bar(bmap['y'].unique(), bmap['y'].value_counts(), color='gold')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Number of Drawings per Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(15, 3))
[ax.axis('off') for ax in axs]
image = bmap.iloc[0, 1]

axs[0].imshow(image)
axs[0].set_title('Original')

axs[1].imshow(rotate(image, 15))
axs[1].set_title('Rotate 15')

axs[2].imshow(image[:,::-1])
axs[2].set_title('Flip Horizontal')

axs[3].imshow(image[::-1,:])
axs[3].set_title('Flip Vertical')

plt.show()

torch.manual_seed(0)

train_df, test_df = train_test_split(bmap, test_size=0.2, random_state=0)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

print(f'Train: {train_df.shape[0]}, Test: {test_df.shape[0]}, Validation: {val_df.shape[0]}')

batch_size = 64

train_loader = DataLoader(train_df, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_df, batch_size=batch_size, shuffle=False)

class QuickDrawCNN_V1(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), n_classes=10):
        super().__init__()
        
        self.conv_set_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )   
        
        self.conv_set_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential( 
            nn.Linear(8*8*32, n_classes),
        )
        
        self.stack = nn.Sequential(
            self.conv_set_1,
            self.conv_set_2,
            self.flatten,
            self.fc
        )
        
    def forward(self, x):
        x = self.stack(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

model_1 = QuickDrawCNN_V1().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001)

def calc_accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1)
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train(model, train_loader, val_loader, epochs, loss_fn, optimizer, device):
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    
    for epoch in range(epochs):
        
        epoch_train_loss, epoch_train_acc = 0, 0
        size = len(train_loader)
        
        for image, label in train_loader:
            model.train()
            
            image, label = image.to(device), label.to(device)
            
            y_logits = model(image)
            
            loss = loss_fn(y_logits, label)
            acc = calc_accuracy(y_logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_acc += acc
        
        train_loss.append(epoch_train_loss / size)
        train_acc.append(epoch_train_acc / size)
        
        model.eval()
        epoch_train_loss, epoch_train_acc = 0, 0
        size = len(val_loader)
        
        with torch.inference_mode():
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                y_logits = model(image)
                loss = loss_fn(y_logits, label)
                acc = calc_accuracy(y_logits, label)
                
                epoch_train_loss += loss.item()
                epoch_train_acc += acc
                
        val_loss.append(epoch_train_loss / size)
        val_acc.append(epoch_train_acc / size)
        
        print(f'Epoch: {epoch+1} | Val Loss {val_loss[-1]:.5f} | Val Acc {val_acc[-1]:.2f}%')
        
    return train_loss, val_loss, train_acc, val_acc

train_loss, val_loss, train_acc, val_acc = train(model=model_1,
                                                train_loader=train_loader,
                                                val_loader=val_loader,
                                                epochs=5,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                device=device)


