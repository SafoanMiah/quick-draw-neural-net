import torch
import os

from data.process_data import load_data, augment_data, preprocess_data, to_tensors_batch, split_batch
from models.quickdraw_cnn import QuickDrawCNN_V1, QuickDrawCNN_V2
from training.model_train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
image_dir = "data/numpy_bitmap/"
categories = os.listdir(image_dir)

# Split the categories into two halves
# Otherwise I'd need 64+ GB of RAM to load all the data at once
half_point = len(categories) // 2
first_half = categories[:half_point]
second_half = categories[half_point:]

def train_on_files(categories_subset, model, save_path, epochs=10):
    print(f"Processing {len(categories_subset)} files...")
    
    # Load and preprocess data
    bmap = preprocess_data(load_data(image_dir, file_standardize=False, sample=False, sample_size=1))
    bmap = bmap[bmap["category"].isin(categories_subset)]
    
    features, labels, labels_map = to_tensors_batch(bmap.copy())
    bmap = list(zip(features, labels))
    
    # Split data into train, test, and validation
    train_loader, test_loader, val_loader = split_batch(bmap, batch_size=64)
    
    # Initialize model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_loss, val_loss, train_acc, val_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.")
    
    return labels_map, train_loss, val_loss, train_acc, val_acc
    
# Initialize model outside
model = QuickDrawCNN_V2().to(device)
labels_map, train_loss, val_loss, train_acc, val_acc = train_on_files(first_half, model, "models/quickdraw_cnn_first_half.pth", epochs=10)
model.load_state_dict(torch.load("models/quickdraw_cnn_first_half.pth"))
labels_map, train_loss, val_loss, train_acc, val_acc = train_on_files(second_half, model, "models/quickdraw_cnn_final.pth", epochs=10)

print("Training complete!")

# Save labels_map, train_loss, val_loss, train_acc, val_acc
import pickle

with open("models/labels_map.pkl", "wb") as f:
    pickle.dump(labels_map, f)
    pickle.dump(train_loss, f)
    pickle.dump(val_loss, f)
    pickle.dump(train_acc, f)
    pickle.dump(val_acc, f)