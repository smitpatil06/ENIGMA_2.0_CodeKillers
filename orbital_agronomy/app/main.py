from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import base64
import torch
import os
import segmentation_models_pytorch as smp

# --- Global Model Loading ---
# Pointing directly to your saved weights
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'stress_vision_weights.pth')

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the trained U-Net Segmentation model"""
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # Architecture must match the Kaggle training script EXACTLY
            model = smp.Unet(
                encoder_name="resnet34", 
                encoder_weights=None, # No need to download imagenet weights for inference
                in_channels=4,        # RGB (3) + NIR (1)
                classes=1,            # 1 class: Stress
                activation='sigmoid'  # Outputs probability 0 to 1
            )
            
            # Load the state dict
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            print(f"✅ U-Net 'Stress-Vision' Model loaded successfully from {MODEL_PATH}!")
            return True
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            return False
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}. Using enhanced simulation mode.")
        return False

def create_stress_heatmap_cnn(rgb_img, nir_img=None):
    """Generate stress heatmap using the U-Net model"""
    h, w = rgb_img.shape[:2]
    
    if model is not None and nir_img is not None:
        try:
            # 1. Resize images to 512x512 (the size the model was trained on)
            rgb_resized = cv2.resize(rgb_img, (512, 512))
            nir_resized = cv2.resize(nir_img, (512, 512))
            
            # Convert RGB back to standard format if needed (FastAPI decodes as BGR)
            rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
            
            # 2. Stack into 4 Channels: R, G, B, NIR
            combined = np.dstack([rgb_resized, nir_resized]).astype(np.float32) / 255.0
            
            # 3. Convert to PyTorch Tensor format (Batch, Channels, Height, Width)
            tensor = torch.tensor(combined.transpose(2, 0, 1)).unsqueeze(0).to(device)
            
            # 4. Run Inference
            with torch.no_grad():
                prediction = model(tensor)
                mask = prediction.squeeze().cpu().numpy() # Extract the 512x512 2D mask
                
            # 5. Resize mask back to original image dimensions for the overlay
            stress_map = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return stress_map
            
        except Exception as e:
            print(f"CNN inference error: {e}, falling back to simulation")
            return create_enhanced_simulation(rgb_img, nir_img)
    else:
        # Fallback if model fails to load
        return create_enhanced_simulation(rgb_img, nir_img)

def create_enhanced_simulation(rgb_img, nir_img):
    """Fallback simulation if weights aren't found"""
    h, w = rgb_img.shape[:2]
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (15, 40, 40), (45, 255, 255))
    value_channel = hsv[:, :, 2]
    dark_mask = (value_channel < 100).astype(np.uint8) * 255
    stress_indicators = cv2.addWeighted(yellow_mask, 0.6, dark_mask, 0.4, 0)
    stress_map = cv2.GaussianBlur(stress_indicators, (15, 15), 0).astype(np.float32) / 255.0
    return stress_map

def calculate_metrics(stress_map):
    """Calculate health metrics from stress map (Thresholds matched to UI)"""
    total_pixels = stress_map.size
    
    # Using the exact thresholds from our UI components
    healthy = np.sum(stress_map < 0.3)
    moderate = np.sum((stress_map >= 0.3) & (stress_map < 0.7))
    severe = np.sum(stress_map >= 0.7)
    
    stressed_pixels = moderate + severe
    
    return {
        "healthy_percent": round((healthy / total_pixels) * 100, 1),
        "stressed_percent": round((stressed_pixels / total_pixels) * 100, 1),
        "severity": {
            "healthy": int(healthy),
            "moderate": int(moderate),
            "severe": int(severe)
        }
    }

def create_heatmap_overlay(rgb_img, stress_map):
    """Create color-coded heatmap overlay and blend with original image"""
    # Create a blank red heatmap image
    heatmap = np.zeros_like(rgb_img)
    heatmap[:, :, 2] = 255 # Set Red channel to maximum (OpenCV uses BGR)
    
    # Binarize mask for clean glowing overlay (confidence > 0.5)
    binary_mask = (stress_map > 0.5).astype(np.uint8)
    
    # Mask out only the stressed areas
    stress_overlay = cv2.bitwise_and(heatmap, heatmap, mask=binary_mask)
    
    # Blend the original RGB image with the red stress overlay
    alpha = 0.55
    blended = cv2.addWeighted(stress_overlay, alpha, rgb_img, 1 - alpha, 0)
    
    return blended

# --- FastAPI Application ---
app = FastAPI(title="Orbital Agronomy: Stress-Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/api/analyze")
async def analyze_crop(rgb_file: UploadFile = File(...), nir_file: UploadFile = File(...)):
    """Analyze crop stress from RGB and NIR imagery"""
    try:
        # Read RGB image
        rgb_bytes = await rgb_file.read()
        np_arr = np.frombuffer(rgb_bytes, np.uint8)
        rgb_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if rgb_img is None:
            raise HTTPException(status_code=400, detail="Invalid RGB image")
        
        # Read NIR image
        nir_bytes = await nir_file.read()
        nir_arr = np.frombuffer(nir_bytes, np.uint8)
        nir_img = cv2.imdecode(nir_arr, cv2.IMREAD_GRAYSCALE)
        
        # Generate stress heatmap
        stress_map = create_stress_heatmap_cnn(rgb_img, nir_img)
        
        # Calculate metrics
        metrics = calculate_metrics(stress_map)
        
        # Create visualization
        blended = create_heatmap_overlay(rgb_img, stress_map)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', blended)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "image_base64": f"data:image/jpeg;base64,{img_str}",
            "metrics": metrics,
            "model_type": "U-Net CNN" if model is not None else "Enhanced Simulation"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)