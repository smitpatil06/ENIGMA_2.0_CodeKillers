from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import base64
import torch
import os
import joblib
import segmentation_models_pytorch as smp
import pandas as pd
from rasterio.io import MemoryFile
from typing import Optional

# --- Global Model Loading ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# U-Net Model (CNN 2D)
UNET_PATH = os.path.join(MODEL_DIR, 'CNN_2D', 'stress_vision_weights.pth')

# CNN + Scikit Model (MLP Deep Learning)
CNN_SCIKIT_DIR = os.path.join(MODEL_DIR, 'CNN+scikit')
DEEP_MODEL_PATH = os.path.join(CNN_SCIKIT_DIR, 'deep_stress_model.pkl')
DEEP_ENCODER_PATH = os.path.join(CNN_SCIKIT_DIR, 'label_encoder.pkl')
DEEP_SCALER_PATH = os.path.join(CNN_SCIKIT_DIR, 'data_scaler.pkl')

# Random Forest Model
RF_MODEL_PATH = os.path.join(CNN_SCIKIT_DIR, 'stress_vision_model.pkl')

# Global model containers
unet_model = None
deep_model = None
rf_model = None
deep_scaler = None
deep_encoder = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_unet_model():
    """Load U-Net segmentation model"""
    global unet_model
    if os.path.exists(UNET_PATH):
        try:
            unet_model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=4,
                classes=1,
                activation='sigmoid'
            )
            unet_model.load_state_dict(torch.load(UNET_PATH, map_location=device))
            unet_model.to(device)
            unet_model.eval()
            print(f"✅ U-Net Model loaded from {UNET_PATH}")
            return True
        except Exception as e:
            print(f"⚠️ U-Net load failed: {e}")
            return False
    else:
        print(f"⚠️ U-Net model not found at {UNET_PATH}")
        return False

def load_deep_learning_model():
    """Load MLP Deep Learning model (CNN 1D + Scikit)"""
    global deep_model, deep_scaler, deep_encoder
    try:
        if os.path.exists(DEEP_MODEL_PATH) and os.path.exists(DEEP_SCALER_PATH) and os.path.exists(DEEP_ENCODER_PATH):
            deep_model = joblib.load(DEEP_MODEL_PATH)
            deep_scaler = joblib.load(DEEP_SCALER_PATH)
            deep_encoder = joblib.load(DEEP_ENCODER_PATH)
            print(f"✅ Deep Learning Model loaded from {DEEP_MODEL_PATH}")
            return True
        else:
            print(f"⚠️ Deep Learning model files not found")
            return False
    except Exception as e:
        print(f"⚠️ Deep Learning model load failed: {e}")
        return False

def load_random_forest_model():
    """Load Random Forest model"""
    global rf_model
    try:
        if os.path.exists(RF_MODEL_PATH):
            rf_model = joblib.load(RF_MODEL_PATH)
            print(f"✅ Random Forest Model loaded from {RF_MODEL_PATH}")
            return True
        else:
            print(f"⚠️ Random Forest model not found at {RF_MODEL_PATH}")
            return False
    except Exception as e:
        print(f"⚠️ Random Forest model load failed: {e}")
        return False

def load_all_models():
    """Load all available models"""
    load_unet_model()
    load_deep_learning_model()
    load_random_forest_model()


def run_unet_inference(rgb_img, nir_img):
    """
    Returns a float32 probability map in [0, 1] at the original image size.
    Uses test-time augmentation (TTA) with horizontal + vertical flip for
    smoother, more confident predictions — especially at faint/moderate zones.
    """
    h, w = rgb_img.shape[:2]
    rgb_r = cv2.resize(rgb_img, (512, 512))
    nir_r = cv2.resize(nir_img, (512, 512))
    rgb_r = cv2.cvtColor(rgb_r, cv2.COLOR_BGR2RGB)

    def to_tensor(r, n):
        combined = np.dstack([r, n]).astype(np.float32) / 255.0
        return torch.tensor(combined.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        # Original
        p0 = unet_model(to_tensor(rgb_r, nir_r)).squeeze().cpu().numpy()
        # Horizontal flip TTA
        p1 = np.fliplr(unet_model(to_tensor(np.fliplr(rgb_r), np.fliplr(nir_r))).squeeze().cpu().numpy())
        # Vertical flip TTA
        p2 = np.flipud(unet_model(to_tensor(np.flipud(rgb_r), np.flipud(nir_r))).squeeze().cpu().numpy())

    # Average the three predictions — smoother gradients in moderate zones
    prob_map = (p0 + p1 + p2) / 3.0

    # Resize back to original image dimensions
    return cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)


def run_deep_learning_inference(rgb_img, nir_img):
    """
    Run inference using the MLP Deep Learning model (CNN 1D + Scikit)
    Processes the image pixel-by-pixel to generate stress probability map
    """
    h, w = rgb_img.shape[:2]
    
    # Convert images to feature vectors (simplified - using RGB + NIR as bands)
    rgb_float = rgb_img.astype(np.float32)
    nir_float = nir_img.astype(np.float32)
    
    # Stack to create multi-band representation
    # Reshape: (H, W, 3+1) -> (H*W, 4)
    rgb_reshaped = rgb_float.reshape(-1, 3)
    nir_reshaped = nir_float.reshape(-1, 1)
    
    # Create basic spectral features (simulate hyperspectral bands)
    # In real scenario, this would be actual hyperspectral data
    num_pixels = h * w
    num_bands = 126  # Expected by the model
    
    # Create synthetic band features from RGB and NIR
    features = np.zeros((num_pixels, num_bands), dtype=np.float32)
    features[:, :3] = rgb_reshaped
    features[:, 90] = nir_reshaped.squeeze()  # NIR in Band 90
    
    # Calculate NDWI and PRI (as done in training)
    nir_band = features[:, 90] + 0.0001
    swir_band = features[:, 110] + 0.0001
    ndwi = (nir_band - swir_band) / (nir_band + swir_band)
    
    b531 = features[:, 45] + 0.0001
    b570 = features[:, 55] + 0.0001
    pri = (b531 - b570) / (b531 + b570)
    
    # Add engineered features
    features_with_indices = np.column_stack([features, ndwi, pri])
    
    # Clean up
    features_with_indices = np.nan_to_num(features_with_indices, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    features_scaled = deep_scaler.transform(features_with_indices)
    
    # Get probabilities
    probabilities = deep_model.predict_proba(features_scaled)
    
    # Get stress probability (assuming 'Rust' class)
    try:
        rust_idx = list(deep_encoder.classes_).index('Rust')
    except ValueError:
        rust_idx = -1  # Last class if 'Rust' not found
    
    stress_probs = probabilities[:, rust_idx]
    prob_map = stress_probs.reshape((h, w))
    
    return prob_map


def run_random_forest_inference(rgb_img, nir_img):
    """
    Run inference using the Random Forest model
    Similar to deep learning but using RF classifier
    """
    h, w = rgb_img.shape[:2]
    
    # Convert images to feature vectors
    rgb_float = rgb_img.astype(np.float32)
    nir_float = nir_img.astype(np.float32)
    
    # Stack to create multi-band representation
    rgb_reshaped = rgb_float.reshape(-1, 3)
    nir_reshaped = nir_float.reshape(-1, 1)
    
    num_pixels = h * w
    num_bands = len(rf_model.feature_names_in_)
    
    # Create features matching training format
    features = np.zeros((num_pixels, num_bands), dtype=np.float32)
    
    # Map RGB and NIR to appropriate bands
    for i, band_name in enumerate(rf_model.feature_names_in_):
        if 'Band_1' in band_name:
            features[:, i] = rgb_reshaped[:, 2]  # Blue
        elif 'Band_2' in band_name:
            features[:, i] = rgb_reshaped[:, 1]  # Green
        elif 'Band_3' in band_name:
            features[:, i] = rgb_reshaped[:, 0]  # Red (BGR format)
        elif 'Band_90' in band_name:
            features[:, i] = nir_reshaped.squeeze()
    
    # Get probabilities
    probabilities = rf_model.predict_proba(features)
    
    # Get stress probability (assuming 'Rust' class)
    try:
        rust_idx = list(rf_model.classes_).index('Rust')
    except ValueError:
        rust_idx = -1
    
    stress_probs = probabilities[:, rust_idx]
    prob_map = stress_probs.reshape((h, w))
    
    return prob_map


def read_tif_bytes(tif_bytes):
    with MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            img_data = src.read().astype(np.float32)
            height, width = src.height, src.width
    return img_data, height, width


def build_bgr_preview(img_data):
    num_bands = img_data.shape[0]
    if num_bands >= 3:
        red_idx = 40 if num_bands > 40 else min(num_bands - 1, 2)
        green_idx = 25 if num_bands > 25 else min(num_bands - 1, 1)
        blue_idx = 10 if num_bands > 10 else 0
        red_band = img_data[red_idx]
        green_band = img_data[green_idx]
        blue_band = img_data[blue_idx]
    else:
        band = img_data[0]
        red_band = band
        green_band = band
        blue_band = band

    bgr_img = np.dstack((blue_band, green_band, red_band))
    bgr_min, bgr_max = np.min(bgr_img), np.max(bgr_img)
    if bgr_max > bgr_min:
        bgr_img = (bgr_img - bgr_min) / (bgr_max - bgr_min)
    return (bgr_img * 255).clip(0, 255).astype(np.uint8)


def run_deep_learning_inference_tif(img_data):
    num_bands = img_data.shape[0]
    height, width = img_data.shape[1], img_data.shape[2]

    flattened_data = img_data.reshape(num_bands, -1).transpose()
    df_pixels = pd.DataFrame(flattened_data, columns=[f"Band_{i+1}" for i in range(num_bands)])

    for i in range(1, 127):
        band_name = f"Band_{i}"
        if band_name not in df_pixels.columns:
            df_pixels[band_name] = 0
    df_pixels = df_pixels[[f"Band_{i}" for i in range(1, 127)]]

    nir_band = df_pixels["Band_90"] + 0.0001
    swir_band = df_pixels["Band_110"] + 0.0001
    df_pixels["NDWI_Water_Stress"] = (nir_band - swir_band) / (nir_band + swir_band)

    b531 = df_pixels["Band_45"] + 0.0001
    b570 = df_pixels["Band_55"] + 0.0001
    df_pixels["PRI_Photosynthetic_Stress"] = (b531 - b570) / (b531 + b570)

    df_pixels.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_pixels.fillna(0, inplace=True)

    X_scaled = deep_scaler.transform(df_pixels.values)
    probabilities = deep_model.predict_proba(X_scaled)
    try:
        rust_idx = list(deep_encoder.classes_).index("Rust")
    except ValueError:
        rust_idx = -1

    stress_probs = probabilities[:, rust_idx]
    return stress_probs.reshape((height, width))


def run_random_forest_inference_tif(img_data):
    num_bands = img_data.shape[0]
    height, width = img_data.shape[1], img_data.shape[2]

    flattened_data = img_data.reshape(num_bands, -1).transpose()
    df_pixels = pd.DataFrame(flattened_data, columns=[f"Band_{i+1}" for i in range(num_bands)])

    expected_features = rf_model.feature_names_in_
    for col in expected_features:
        if col not in df_pixels.columns:
            df_pixels[col] = 0
    df_pixels = df_pixels[expected_features]

    probabilities = rf_model.predict_proba(df_pixels)
    try:
        rust_idx = list(rf_model.classes_).index("Rust")
    except ValueError:
        rust_idx = -1

    stress_probs = probabilities[:, rust_idx]
    return stress_probs.reshape((height, width))



def create_enhanced_simulation(rgb_img, nir_img):
    """Fallback simulation producing a smooth 3-level probability map."""
    h, w = rgb_img.shape[:2]
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 40, 40), (45, 255, 255)).astype(np.float32) / 255.0
    dark   = (hsv[:, :, 2] < 100).astype(np.float32)
    raw    = yellow * 0.6 + dark * 0.4
    smooth = cv2.GaussianBlur(raw, (21, 21), 0)
    return smooth


def calculate_metrics(stress_map):
    """
    Three tiers using probability thresholds:
      Healthy   : prob < 0.30
      Moderate  : 0.30 <= prob < 0.60
      Severe    : prob >= 0.60
    """
    total = stress_map.size
    # Raised thresholds: model tends to over-predict severity on Agriculture-Vision
    # Moderate: 0.35–0.70 captures the "at-risk" zone without false alarms
    # Severe:   > 0.70 reserved for genuinely high-confidence detections
    healthy  = int(np.sum(stress_map < 0.35))
    moderate = int(np.sum((stress_map >= 0.35) & (stress_map < 0.70)))
    severe   = int(np.sum(stress_map >= 0.70))

    return {
        "healthy_percent":  round(healthy  / total * 100, 1),
        "moderate_percent": round(moderate / total * 100, 1),
        "severe_percent":   round(severe   / total * 100, 1),
        "stressed_percent": round((moderate + severe) / total * 100, 1),
        "severity": {
            "healthy":  healthy,
            "moderate": moderate,
            "severe":   severe,
        },
        "mean_stress": round(float(np.mean(stress_map)), 4),
    }


def create_heatmap_overlay(rgb_img, stress_map):
    """
    Smooth, 3-tier colour-coded overlay:
      Green  (healthy)   : prob < 0.30  — no overlay
      Yellow (moderate)  : 0.30–0.60    — yellow tint proportional to prob
      Red    (severe)    : > 0.60       — red tint proportional to prob

    Alpha for each pixel = probability, so faint detections appear as
    translucent washes rather than being invisible (binary) or fully opaque.
    """
    h, w = rgb_img.shape[:2]
    overlay = rgb_img.copy().astype(np.float32)

    # --- Moderate zone: yellow (BGR 0, 220, 255) ---
    mod_mask = (stress_map >= 0.35) & (stress_map < 0.70)
    mod_alpha = np.where(mod_mask,
                         (stress_map - 0.35) / 0.35,   # 0→1 within the band
                         0.0).astype(np.float32)
    yellow = np.array([0, 220, 255], dtype=np.float32)   # BGR yellow
    for c in range(3):
        overlay[:, :, c] = np.where(
            mod_mask,
            overlay[:, :, c] * (1 - mod_alpha * 0.65) + yellow[c] * mod_alpha * 0.65,
            overlay[:, :, c]
        )

    # --- Severe zone: red (BGR 30, 30, 220) ---
    sev_mask = stress_map >= 0.70
    sev_alpha = np.where(sev_mask,
                         np.clip((stress_map - 0.70) / 0.30, 0, 1),
                         0.0).astype(np.float32)
    red = np.array([30, 30, 220], dtype=np.float32)       # BGR red
    for c in range(3):
        overlay[:, :, c] = np.where(
            sev_mask,
            overlay[:, :, c] * (1 - sev_alpha * 0.70) + red[c] * sev_alpha * 0.70,
            overlay[:, :, c]
        )

    return np.clip(overlay, 0, 255).astype(np.uint8)


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
    load_all_models()


@app.post("/api/analyze")
async def analyze_crop(
    rgb_file: Optional[UploadFile] = File(None),
    nir_file: Optional[UploadFile] = File(None),
    tif_file: Optional[UploadFile] = File(None),
    model_type: str = Form(default="unet")
):
    try:
        stress_map = None
        model_name = None

        if model_type in {"deep", "rf"}:
            if tif_file is None:
                raise HTTPException(status_code=400, detail="Upload a .tif file for the selected model.")

            tif_bytes = await tif_file.read()
            if not tif_bytes:
                raise HTTPException(status_code=400, detail="Empty .tif file uploaded.")

            img_data, height, width = read_tif_bytes(tif_bytes)
            rgb_img = build_bgr_preview(img_data)

            if model_type == "deep" and deep_model is not None:
                try:
                    stress_map = run_deep_learning_inference_tif(img_data)
                    model_name = "MLP Deep Learning (CNN 1D + Scikit)"
                except Exception as e:
                    print(f"Deep Learning inference error: {e}")
                    stress_map = create_enhanced_simulation(rgb_img, None)
                    model_name = "Deep Learning Fallback Simulation"
            elif model_type == "rf" and rf_model is not None:
                try:
                    stress_map = run_random_forest_inference_tif(img_data)
                    model_name = "Random Forest Classifier"
                except Exception as e:
                    print(f"Random Forest inference error: {e}")
                    stress_map = create_enhanced_simulation(rgb_img, None)
                    model_name = "Random Forest Fallback Simulation"
            else:
                stress_map = create_enhanced_simulation(rgb_img, None)
                model_name = f"Enhanced Simulation ({model_type} not available)"

        else:
            if rgb_file is None or nir_file is None:
                raise HTTPException(status_code=400, detail="Upload RGB and NIR images for U-Net analysis.")

            rgb_bytes = await rgb_file.read()
            rgb_img = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
            if rgb_img is None:
                raise HTTPException(status_code=400, detail="Invalid RGB image")

            nir_bytes = await nir_file.read()
            nir_img = cv2.imdecode(np.frombuffer(nir_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

            if model_type == "unet" and unet_model is not None:
                try:
                    if nir_img is not None:
                        stress_map = run_unet_inference(rgb_img, nir_img)
                        model_name = "U-Net (CNN 2D) + TTA"
                    else:
                        stress_map = create_enhanced_simulation(rgb_img, nir_img)
                        model_name = "U-Net Simulation (No NIR)"
                except Exception as e:
                    print(f"U-Net inference error: {e}")
                    stress_map = create_enhanced_simulation(rgb_img, nir_img)
                    model_name = "U-Net Fallback Simulation"
            else:
                stress_map = create_enhanced_simulation(rgb_img, nir_img)
                model_name = f"Enhanced Simulation ({model_type} not available)"

        metrics = calculate_metrics(stress_map)
        blended = create_heatmap_overlay(rgb_img, stress_map)

        _, buf = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_b64 = base64.b64encode(buf).decode('utf-8')

        _, preview_buf = cv2.imencode('.jpg', rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        preview_b64 = base64.b64encode(preview_buf).decode('utf-8')

        return {
            "status":       "success",
            "image_base64": f"data:image/jpeg;base64,{img_b64}",
            "metrics":      metrics,
            "model_type":   model_name,
            "preview_base64": f"data:image/jpeg;base64,{preview_b64}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/models/status")
async def get_model_status():
    """Return which models are currently loaded and available"""
    return {
        "unet": unet_model is not None,
        "deep_learning": deep_model is not None,
        "random_forest": rf_model is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
