import torch
import torch.nn as nn
import rasterio
import numpy as np
import joblib
import os

# --- RE-DEFINE ARCHITECTURE ---
class SpatialStressCNN(nn.Module):
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
        # Patch size 16 results in 4x4 after two pools
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_cnn_stressed_coordinates(image_path, model_dir):
    """
    Zero-Loss 2D Inference: Standardizes new images to match training dimensions.
    """
    model_path = os.path.join(model_dir, 'stress_cnn.pt')
    encoder_path = os.path.join(model_dir, 'cnn_label_encoder.pkl')
    scaler_path = os.path.join(model_dir, 'cnn_scaler.pkl')
    
    if not all(os.path.exists(p) for p in [model_path, encoder_path, scaler_path]):
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    le = joblib.load(encoder_path)
    max_val = joblib.load(scaler_path)  # Load max_val as joblib pickle
    
    # Identify the 'Stressed' label index
    try:
        rust_idx = list(le.classes_).index('Rust')
    except ValueError:
        rust_idx = 1 # Default fallback

    patch_size = 16
    stride = 8 # Sliding window with overlap

    with rasterio.open(image_path) as src:
        img_data = src.read().astype(np.float32)
        c, h, w = img_data.shape
        
        # 1. DYNAMIC PADDING TO MATCH TRAINING
        # Load the model first to see how many bands it was trained on
        temp_model_state = torch.load(model_path, map_location='cpu')
        trained_bands = temp_model_state['conv1.weight'].shape[1]
        
        # Spatial Padding for small images
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            img_data = np.pad(img_data, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
            _, h, w = img_data.shape
            
        # Channel Padding to match trained band count
        if c < trained_bands:
            img_data = np.pad(img_data, ((0, trained_bands - c), (0,0), (0,0)), mode='constant')
        elif c > trained_bands:
            img_data = img_data[:trained_bands, :, :] # Trim if extra bands exist

        # 2. SCALING & MODEL INIT
        img_data = np.clip(img_data / max_val, 0, 1)
        model = SpatialStressCNN(num_bands=trained_bands, num_classes=len(le.classes_)).to(device)
        model.load_state_dict(temp_model_state)
        model.eval()

        # 3. SLIDING WINDOW INFERENCE
        stressed_points = []
        patches = []
        coords = []

        for r in range(0, h - patch_size + 1, stride):
            for col in range(0, w - patch_size + 1, stride):
                patches.append(img_data[:, r:r+patch_size, col:col+patch_size])
                coords.append((r + patch_size//2, col + patch_size//2))
                
                if len(patches) >= 1024:
                    batch = torch.tensor(np.array(patches)).to(device)
                    with torch.no_grad():
                        preds = torch.argmax(model(batch), dim=1).cpu().numpy()
                    for i, p in enumerate(preds):
                        if p == rust_idx:
                            lon, lat = src.xy(coords[i][0], coords[i][1])
                            stressed_points.append([lat, lon, 1.0])
                    patches, coords = [], []

        # Final batch processing
        if patches:
            batch = torch.tensor(np.array(patches)).to(device)
            with torch.no_grad():
                preds = torch.argmax(model(batch), dim=1).cpu().numpy()
            for i, p in enumerate(preds):
                if p == rust_idx:
                    lon, lat = src.xy(coords[i][0], coords[i][1])
                    stressed_points.append([lat, lon, 1.0])

    return stressed_points