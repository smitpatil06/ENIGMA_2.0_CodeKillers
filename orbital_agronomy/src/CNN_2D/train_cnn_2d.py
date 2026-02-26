import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
import time

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIRECTORY = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'train')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

PATCH_SIZE = 16
EPOCHS = 10
BATCH_SIZE = 128 # Increased batch size because we are lazy loading
STRIDE = 8

# --- ARCHITECTURE ---
class SpatialStressCNN(nn.Module):
    """Adaptive 2D CNN that handles dynamic band counts."""
    def __init__(self, num_bands, num_classes=3):
        super(SpatialStressCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(num_bands, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2) 
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2) 
        
        self.flatten = nn.Flatten()
        flat_size = 64 * (PATCH_SIZE // 4) * (PATCH_SIZE // 4)
        
        self.fc1 = nn.Linear(flat_size, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- LAZY LOADING DATA LOADER ---
class TIFFPatchDatasetLazy(Dataset):
    """Memory-efficient Dataset that loads patches from disk on-the-fly."""
    def __init__(self, root_dir, patch_size=16, stride=8):
        self.patch_info = [] # Stores ONLY metadata: (filepath, row, col, label_idx)
        self.patch_size = patch_size
        
        print("Pass 1: Scanning dataset metadata...")
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.le = LabelEncoder().fit(classes)
        
        max_bands = 0
        global_max_val = 0.0 

        # Scan files to build the index, DO NOT load images into RAM
        for label_name in classes:
            class_dir = os.path.join(root_dir, label_name)
            label_idx = self.le.transform([label_name])[0]
            
            for img_name in os.listdir(class_dir):
                if not img_name.endswith('.tif'): continue
                img_path = os.path.join(class_dir, img_name)
                
                try:
                    with rasterio.open(img_path) as src:
                        # 1. Update max bands
                        max_bands = max(max_bands, src.count)
                        
                        # 2. Update global max value (read very briefly)
                        data_temp = src.read()
                        local_max = np.percentile(data_temp, 99.0)
                        global_max_val = max(global_max_val, local_max)
                        
                        h, w = data_temp.shape[1], data_temp.shape[2]
                        del data_temp # Free RAM immediately
                        
                        # 3. Pre-calculate all valid window coordinates
                        pad_h = max(0, patch_size - h)
                        pad_w = max(0, patch_size - w)
                        padded_h = h + pad_h
                        padded_w = w + pad_w

                        for r in range(0, padded_h - patch_size + 1, stride):
                            for c in range(0, padded_w - patch_size + 1, stride):
                                # Store the recipe, not the data!
                                self.patch_info.append((img_path, r, c, label_idx))
                                
                except Exception as e:
                    print(f"Skipping {img_name}: {e}")
                    continue
        
        self.num_bands = max_bands
        self.max_val = global_max_val if global_max_val > 0 else 1.0
        
        print(f"Indexed {len(self.patch_info)} patches without loading them into RAM.")
        print(f"Max Bands: {self.num_bands} | Global Max Val: {self.max_val:.2f}")

    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        # PyTorch calls this function ONLY when it needs an image for the batch.
        img_path, r, c, label_idx = self.patch_info[idx]
        
        with rasterio.open(img_path) as src:
            img_data = src.read().astype(np.float32)
            channels, h, w = img_data.shape
            
            # Spatial Padding
            if h < self.patch_size or w < self.patch_size:
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)
                img_data = np.pad(img_data, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
                
            # Channel Padding
            if channels < self.num_bands:
                img_data = np.pad(img_data, ((0, self.num_bands - channels), (0,0), (0,0)), mode='constant')
            elif channels > self.num_bands:
                img_data = img_data[:self.num_bands, :, :]
                
            # Extract patch
            patch = img_data[:, r:r+self.patch_size, c:c+self.patch_size]
            
            # Scale
            patch = np.clip(patch / self.max_val, 0, 1)
            
            # Fallback for empty patches
            if patch.shape != (self.num_bands, self.patch_size, self.patch_size):
                patch = np.zeros((self.num_bands, self.patch_size, self.patch_size), dtype=np.float32)

        return torch.tensor(patch), torch.tensor(label_idx, dtype=torch.long)

# --- TRAINING ---
def train():
    print("--- STARTING 2D LAZY-LOAD TRAINING ---")
    dataset = TIFFPatchDatasetLazy(TRAIN_DIRECTORY, patch_size=PATCH_SIZE, stride=STRIDE)
    if len(dataset) == 0: 
        print("Dataset empty!")
        return

    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    # Use num_workers for parallel loading from disk
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = SpatialStressCNN(num_bands=dataset.num_bands, num_classes=len(dataset.le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                _, pred = torch.max(model(inputs.to(device)), 1)
                correct += (pred == labels.to(device)).sum().item()
                
        acc = 100 * correct / len(test_ds)
        elapsed = round(time.time() - t0, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}% | {elapsed}s")
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'stress_cnn.pt'))
    joblib.dump(dataset.le, os.path.join(MODEL_DIR, 'cnn_label_encoder.pkl'))
    joblib.dump(dataset.max_val, os.path.join(MODEL_DIR, 'cnn_scaler.pkl'))
    print("--- 2D CNN TRAINING COMPLETE ---")

if __name__ == "__main__":
    train()