import rasterio
import joblib
import pandas as pd
import numpy as np
import os
import torch

def get_rf_stressed_coordinates(image_path, model_path):
    """
    Random Forest Inference: Finds pixels classified as 'Rust' (Stressed).
    Returns list of [lat, lon, intensity] tuples for heatmap.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return []

    model = joblib.load(model_path)
    stressed_points = []
    
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
            num_bands, height, width = img_data.shape
            
            # Flatten and predict
            flattened_data = img_data.reshape(num_bands, -1).transpose()
            column_names = [f"Band_{i+1}" for i in range(num_bands)]
            df_temp = pd.DataFrame(flattened_data, columns=column_names)
            
            # Get predictions
            predictions = model.predict(df_temp)
            
            # Get prediction probabilities for intensity
            try:
                probs = model.predict_proba(df_temp)
                max_prob = np.max(probs, axis=1)
            except:
                max_prob = np.ones_like(predictions)
            
            # Reshape predictions back to image grid (height x width)
            prediction_grid = predictions.reshape((height, width))
            prob_grid = max_prob.reshape((height, width))
            
            # Find pixels labeled 'Rust' (or encoded as 2 if using label encoding)
            # HACKATHON OPTIMIZATION: Sample every 10th pixel to prevent Folium crash
            stride = 10 
            for row in range(0, height, stride):
                for col in range(0, width, stride):
                    # Check if this pixel is classified as stressed (Rust)
                    if prediction_grid[row, col] == 'Rust' or prediction_grid[row, col] == 2:
                        # Convert the row/col index back into real-world Lat/Lon
                        lon, lat = src.xy(row, col)
                        # Append [lat, lon, intensity] where intensity is prediction confidence
                        intensity = float(prob_grid[row, col])
                        stressed_points.append([lat, lon, min(intensity, 1.0)])
    except Exception as e:
        print(f"Error processing RF inference: {e}")
    
    return stressed_points


def get_cnn_stressed_coordinates(image_path, model_path, scaler_path):
    """
    CNN (Deep Learning) Inference: 1D-CNN for stress detection.
    CRITICAL: Must use the same scaler that was used in training!
    
    Args:
        image_path: Path to satellite image (TIF)
        model_path: Path to trained CNN model (.pt file)
        scaler_path: Path to StandardScaler used during training
    
    Returns:
        List of [lat, lon, intensity] tuples for heatmap
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"CNN model or scaler missing")
        return []
    
    stressed_points = []
    
    try:
        # Load the scaler (CRUCIAL for CNN!)
        scaler = joblib.load(scaler_path)
        
        # Load the CNN model
        device = torch.device('cpu')  # Use CPU for inference in Streamlit
        model = torch.load(model_path, map_location=device)
        model.eval()
        
        with rasterio.open(image_path) as src:
            img_data = src.read()
            num_bands, height, width = img_data.shape
            
            # Flatten
            flattened_data = img_data.reshape(num_bands, -1).transpose()
            
            # CRITICAL: Scale the data with the same scaler used in training
            # If you skip this, the CNN will predict garbage!
            flattened_data_scaled = scaler.transform(flattened_data)
            
            # Convert to tensor
            X_tensor = torch.tensor(flattened_data_scaled, dtype=torch.float32)
            
            # Inference
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                confidences = torch.max(probs, dim=1)[0].cpu().numpy()
            
            # Reshape back to image grid
            prediction_grid = predictions.reshape((height, width))
            confidence_grid = confidences.reshape((height, width))
            
            # Find stressed pixels (class 2 = Rust, assuming 0=Health, 1=Other, 2=Rust)
            stride = 10
            for row in range(0, height, stride):
                for col in range(0, width, stride):
                    if prediction_grid[row, col] == 2:  # Rust class
                        lon, lat = src.xy(row, col)
                        intensity = float(confidence_grid[row, col])
                        stressed_points.append([lat, lon, min(intensity, 1.0)])
    
    except Exception as e:
        print(f"Error in CNN inference: {e}")
    
    return stressed_points
