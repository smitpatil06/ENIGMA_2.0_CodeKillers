import rasterio
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
import os
from scipy.ndimage import gaussian_filter

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def visualize_professional_blended_stress(image_path, model_path, mode="drone"):
    print(f"Loading hyperspectral data from: {image_path}...")
    
    with rasterio.open(image_path) as src:
        img_data = src.read()
        num_bands = src.count
        height = src.height
        width = src.width
        
        # Extract RGB for base. Pick bands that best represent RGB for this camera.
        red_band = img_data[40, :, :]   
        green_band = img_data[25, :, :] 
        blue_band = img_data[10, :, :]  
        
        rgb_img = np.dstack((red_band, green_band, blue_band))
        rgb_min, rgb_max = np.min(rgb_img), np.max(rgb_img)
        if rgb_max > rgb_min:
            rgb_img = (rgb_img - rgb_min) / (rgb_max - rgb_min)

    # =========================================================
    # 🌍 THE ORBITAL SCALE TOGGLE: DRONE -> SATELLITE
    # =========================================================
    if mode.lower() == "satellite":
        print("Engaging Orbital Satellite Simulation (Downsampling)...")
        # Define how big the satellite pixel is (e.g., 20x20 drone pixels = 1 satellite pixel)
        block_size = 20 
        
        # We downsample the images (pixelate them) to simulate large satellite pixels.
        # This is a 'Hackathon Shortcut' to simulate orbital scaling.
        img_data = img_data[:, ::block_size, ::block_size]
        rgb_img = rgb_img[::block_size, ::block_size, :]
        
        # Update the dimensions. The images are now much smaller (e.g., 50x50).
        height, width = img_data.shape[1], img_data.shape[2]
        
    else:
        print("Engaging Drone/UAV Mode (High-Resolution)...")
    # =========================================================

    print("Flattening image data for the AI brain...")
    flattened_data = img_data.reshape(num_bands, -1).transpose()
    column_names = [f"Band_{i+1}" for i in range(num_bands)]
    df_pixels = pd.DataFrame(flattened_data, columns=column_names)

    print("Loading the trained Stress-Vision Model...")
    model = joblib.load(model_path)
    
    # Alignment Fix
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in df_pixels.columns:
            df_pixels[col] = 0
    df_pixels = df_pixels[expected_features]

    print("AI is calculating stress severity gradients...")
    # Get probabilities (0.0 to 1.0)
    probabilities = model.predict_proba(df_pixels)
    rust_class_index = list(model.classes_).index('Rust')
    stress_probs = probabilities[:, rust_class_index]
    probability_map = stress_probs.reshape((height, width))

    # =========================================================
    # ✨ THE PROFESSIONAL VISUAL UPGRADE: SMOOTH & MIX
    # =========================================================
    # 1. MASKING: Only process areas where the AI is at least 30% sure there is stress.
    threshold = 0.30
    probability_map[probability_map < threshold] = 0 # Pre-zero non-stress areas

    # 2. COLOUR MIXING (Gaussian Blur):
    # This takes the distinct pixel probabilities and gently 'spreads' them,
    # making the colors blend and mix with their neighbors like the heat blobs
    # in your reference image. Adjust sigma for more or less blend.
    # sigma=1 is good for Drone Mode, sigma=0.5 for Satellite mode
    print("Blending spectral signatures for smooth color mixing...")
    blur_sigma = 1.0 if mode == "drone" else 0.5
    smoothed_prob_map = gaussian_filter(probability_map, sigma=blur_sigma)
    
    # 3. Apply the mask *after* blurring to keep healthy areas green and crisp.
    intense_heatmap = ma.masked_where(smoothed_prob_map < (threshold - 0.05), smoothed_prob_map)

    print("Generating the final, blending high-contrast visual...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Define common plotting settings
    PLOT_SETTINGS = {
        'aspect': 'equal',
        'vmin': 0.0,
        'vmax': 1.0
    }
    
    # NEW VISUAL UPGRADE: Use Bilinear Interpolation for smooth display
    # This addresses the 'pixelate form' directly by smoothing transitions.
    INTERPOLATION = 'bilinear' 

    # Left Side: Standard RGB
    ax1.imshow(rgb_img, interpolation=INTERPOLATION, **PLOT_SETTINGS)
    ax1.set_title("Standard Optical View (RGB Baseline)", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Right Side: Base Image + Blended Heatmap Overlay
    ax2.imshow(rgb_img, interpolation=INTERPOLATION, **PLOT_SETTINGS) 
    
    # NEW VISUAL UPGRADE: A customized, intensely saturated color map
    # that blends smoothly. YlOrRd_r is the standard, but we'll use a custom one 
    # for precise control to get that 'mix well' effect.
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(1, 1, 0, 0), (1, 1, 0, 0.4), (1, 0.5, 0, 0.8), (0.8, 0, 0, 1.0)] # Custom YlOrRd
    custom_ylorrd = LinearSegmentedColormap.from_list("Custom_YlOrRd", colors, N=256)
    
    heatmap_layer = ax2.imshow(intense_heatmap, cmap=custom_ylorrd, alpha=0.9,
                               vmax=1.0, interpolation=INTERPOLATION)
    
    ax2.set_title("Pre-Visual Stress severity Index (Smoothed)", fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Add the professional scale/colorbar
    # Re-labeled colorbar to reflect general spectral stress severity
    cbar = fig.colorbar(heatmap_layer, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Pre-Visual Spectral Stress Severity (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    cbar.set_ticks([0.3, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['30% Early Stress Warning', '50% Moderate Stress', '75% High Stress', '100% Severe Stress'])

    plt.tight_layout()
    plt.show()

# --- RUN THE SCRIPT ---
# Pick ONE unseen image from your validation folder (val/val/)
TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'val', 'val', '110.tif')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'CNN+scikit', 'stress_vision_model.pkl')

# =========================================================
# CHANGE MODE HERE: set to "drone" or "satellite"
# =========================================================
CURRENT_MODE = "drone" # We recommend keeping drone mode for the main visual

if __name__ == "__main__":
    print(f"Test Image Path: {TEST_IMAGE_PATH}")
    print(f"Model Path: {MODEL_PATH}")
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"ERROR: Cannot find the test image at {TEST_IMAGE_PATH}")
    else:
        print(f"Running visualizer in {CURRENT_MODE.upper()} mode...")
        visualize_professional_blended_stress(TEST_IMAGE_PATH, MODEL_PATH, mode=CURRENT_MODE)