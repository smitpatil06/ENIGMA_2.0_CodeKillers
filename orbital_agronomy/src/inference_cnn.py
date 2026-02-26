import torch
import torch.nn as nn
import rasterio
import pandas as pd
import numpy as np
import joblib
import os

# 1. Define the exact same CNN architecture used in training
# (PyTorch needs this to load the saved weights properly)
class StressCNN(nn.Module):
    def __init__(self, num_bands=12): # Change 12 to match your actual number of bands
        super(StressCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (num_bands // 4), 64) 
        self.fc2 = nn.Linear(64, 2) # Assuming 0 = Healthy, 1 = Stressed (Rust)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_cnn_stressed_coordinates(image_path, model_path, scaler_path):
    """
    Runs PyTorch 1D-CNN inference on a satellite image and returns stressed coordinates.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Missing CNN model or Scaler!")
        return []

    # Setup device (Use GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Scaler
    scaler = joblib.load(scaler_path)

    # Load Model
    with rasterio.open(image_path) as src:
        num_bands = src.count
        model = StressCNN(num_bands=num_bands).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode!
        
        # Read and flatten image
        img_data = src.read()
        _, height, width = img_data.shape
        flattened_data = img_data.reshape(num_bands, -1).transpose()
        
        # Scale the data (CRITICAL FOR CNN)
        scaled_data = scaler.transform(flattened_data)
        
        # Convert to PyTorch Tensor: Shape (Batch, Channels, Sequence) -> (N, 1, Bands)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(1).to(device)
        
        print("Running Deep Learning Inference...")
        
        # Predict in batches to avoid running out of RAM/VRAM
        batch_size = 10000 
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(tensor_data), batch_size):
                batch = tensor_data[i:i+batch_size]
                outputs = model(batch)
                # Get the predicted class (0 or 1)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                
        # Reshape predictions back to the image grid
        prediction_grid = np.array(all_preds).reshape((height, width))
        
        stressed_points = []
        stride = 10 # Sample every 10th pixel to prevent Folium from lagging
        
        for row in range(0, height, stride):
            for col in range(0, width, stride):
                # Assuming 1 is the label for 'Stressed' or 'Rust'
                if prediction_grid[row, col] == 1: 
                    lon, lat = src.xy(row, col)
                    stressed_points.append([lat, lon, 1.0])
                    
    return stressed_points