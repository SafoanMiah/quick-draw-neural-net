import torch
import pickle
import os

from utils.model_train import train
from utils.processing import load_data, augment_data, to_tensors, split_batch, process_data, incremental_save
from utils.quickdraw_cnn import QuickDrawCNN_V2

# Set varaibles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = "data/numpy_bitmap/"
print(f"Device: {device} \nUsing {image_dir} as source")

# Load data spit into two (memory issues)
categories = os.listdir(image_dir)
p1, p2 = categories[:len(categories)//2], categories[len(categories)//2:]

features_p1, labels_p1 = load_data(p1, image_dir, file_standardize=False) # files standarized prior to this
features_p2, labels_p2 = load_data(p2, image_dir, file_standardize=False)

# Process data (reshape, normalize, etc)
features, labels = process_data(features_p1, features_p2, labels_p1, labels_p2, portion=0.4)

# Augment data – temporarily disabled
# features, labels = augment_data(features, labels, rot=15, h_flip=True, v_flip=False)

# Convert the features and labels to tensors
features, labels, labels_map = to_tensors(features, labels, device)

# Split into train test val then create loaders (Batch size 64)
train_loader, test_loader, val_loader = split_batch(features, labels, batch_size=64)

# Creating the model based on the V2, with more layers
model = QuickDrawCNN_V2().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model with an early stop that stops model once the improvement is minimal
train_loss, val_loss, train_acc, val_acc = train(model = model,
                                                  train_loader = train_loader,
                                                  val_loader = val_loader,
                                                  epochs = 10,
                                                  criterion = criterion,
                                                  optimizer = optimizer,
                                                  device = device)   

# Save model and variables
parent_path = 'saves'
varaibles_saved = incremental_save(f'{parent_path}/vars/qd_cnn_v2')
with open(varaibles_saved, "wb") as f:
    pickle.dump(labels_map, f)
    pickle.dump(train_loss, f)
    pickle.dump(val_loss, f)
    pickle.dump(train_acc, f)
    pickle.dump(val_acc, f)

model_saved = incremental_save(f'{parent_path}/model/qd_cnn_v2', data=model)