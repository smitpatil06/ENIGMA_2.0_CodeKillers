import os
import rasterio
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
from scipy.ndimage import gaussian_filter

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def visualize_deep_learning_stress(image_path, model_path, encoder_path, scaler_path, mode="drone"):
    print(f"Loading hyperspectral data from: {image_path}...")
    
    with rasterio.open(image_path) as src:
        img_data = src.read()
        num_bands = src.count
        height, width = src.height, src.width
        
        # Extract RGB for the base layer
        red_band, green_band, blue_band = img_data[40, :, :], img_data[25, :, :], img_data[10, :, :]
        rgb_img = np.dstack((red_band, green_band, blue_band))
        rgb_min, rgb_max = np.min(rgb_img), np.max(rgb_img)
        if rgb_max > rgb_min:
            rgb_img = (rgb_img - rgb_min) / (rgb_max - rgb_min)

    # --- SATELLITE TOGGLE ---
    if mode.lower() == "satellite":
        print("Engaging Orbital Satellite Simulation (Downsampling)...")
        block_size = 20 
        img_data = img_data[:, ::block_size, ::block_size]
        rgb_img = rgb_img[::block_size, ::block_size, :]
        height, width = img_data.shape[1], img_data.shape[2]

    print("Flattening image data...")
    flattened_data = img_data.reshape(num_bands, -1).transpose()
    df_pixels = pd.DataFrame(flattened_data, columns=[f"Band_{i+1}" for i in range(num_bands)])

    # =========================================================
    # THE HACKATHON ALIGNMENT FIX (Band_126 mismatch)
    # =========================================================
    print("Aligning image bands with Deep Learning memory...")
    for i in range(1, 127): # Forces Band_1 through Band_126 to exist
        band_name = f"Band_{i}"
        if band_name not in df_pixels.columns:
            df_pixels[band_name] = 0
            
    # Force the exact column order: Band_1 to Band_126
    df_pixels = df_pixels[[f"Band_{i}" for i in range(1, 127)]]

    # =========================================================
    # ADVANCED FEATURE ENGINEERING (NDWI & PRI)
    # =========================================================
    print("Calculating scientific Pre-Visual stress indices (NDWI & PRI)...")
    nir_band = df_pixels['Band_90'] + 0.0001
    swir_band = df_pixels['Band_110'] + 0.0001
    df_pixels['NDWI_Water_Stress'] = (nir_band - swir_band) / (nir_band + swir_band)

    b531 = df_pixels['Band_45'] + 0.0001
    b570 = df_pixels['Band_55'] + 0.0001
    df_pixels['PRI_Photosynthetic_Stress'] = (b531 - b570) / (b531 + b570)

    # Clean up math errors (NaNs and Infinities)
    df_pixels.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_pixels.fillna(0, inplace=True)

    # =========================================================
    # DEEP LEARNING PREPROCESSING & PREDICTION
    # =========================================================
    print("Loading Deep Neural Network Ecosystem...")
    deep_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    print("Scaling data to match training conditions...")
    # Now it perfectly matches the 128 expected features
    X_scaled = scaler.transform(df_pixels.values)

    print("Deep Brain is processing spectral signatures...")
    probabilities = deep_model.predict_proba(X_scaled)
    
    rust_class_index = list(label_encoder.classes_).index('Rust')
    stress_probs = probabilities[:, rust_class_index]
    probability_map = stress_probs.reshape((height, width))

    # =========================================================
    # VISUALIZATION & BLENDING
    # =========================================================
    threshold = 0.30
    probability_map[probability_map < threshold] = 0 

    blur_sigma = 1.0 if mode == "drone" else 0.5
    smoothed_prob_map = gaussian_filter(probability_map, sigma=blur_sigma)
    intense_heatmap = ma.masked_where(smoothed_prob_map < (threshold - 0.05), smoothed_prob_map)

    print("Generating High-Contrast Deep Learning Visual...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    PLOT_SETTINGS = {'aspect': 'equal', 'vmin': 0.0, 'vmax': 1.0, 'interpolation': 'bilinear'}

    # Left Side
    ax1.imshow(rgb_img, **PLOT_SETTINGS)
    ax1.set_title("Standard Optical View (RGB Baseline)", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Right Side
    ax2.imshow(rgb_img, **PLOT_SETTINGS) 
    colors = [(1, 1, 0, 0), (1, 1, 0, 0.4), (1, 0.5, 0, 0.8), (0.8, 0, 0, 1.0)] 
    custom_ylorrd = LinearSegmentedColormap.from_list("Custom_YlOrRd", colors, N=256)
    
    heatmap_layer = ax2.imshow(intense_heatmap, cmap=custom_ylorrd, alpha=0.9, **PLOT_SETTINGS)
    ax2.set_title("Deep Neural Network Stress Index (NDWI+PRI)", fontsize=14, fontweight='bold')
    ax2.axis('off')

    cbar = fig.colorbar(heatmap_layer, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Pre-Visual Spectral Stress Severity (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    cbar.set_ticks([0.3, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['30% Early Stress Warning', '50% Moderate Stress', '75% High Stress', '100% Severe Stress'])

    plt.tight_layout()
    plt.show()

# --- RUN THE SCRIPT ---
TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'val', 'val', '110.tif')
DEEP_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'CNN+scikit', 'deep_stress_model.pkl')
ENCODER_PATH = os.path.join(PROJECT_ROOT, 'models', 'CNN+scikit', 'label_encoder.pkl')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'CNN+scikit', 'data_scaler.pkl')
CURRENT_MODE = "drone" 

if __name__ == "__main__":
    print(f"Test Image Path: {TEST_IMAGE_PATH}")
    print(f"Deep Model Path: {DEEP_MODEL_PATH}")
    print(f"Encoder Path: {ENCODER_PATH}")
    print(f"Scaler Path: {SCALER_PATH}")
    visualize_deep_learning_stress(TEST_IMAGE_PATH, DEEP_MODEL_PATH, ENCODER_PATH, SCALER_PATH, mode=CURRENT_MODE)