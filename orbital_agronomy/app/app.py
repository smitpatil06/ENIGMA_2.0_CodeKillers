import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import sys
import os
import numpy as np
import joblib

# Ensure Streamlit can find the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.processor import get_rf_stressed_coordinates, get_cnn_stressed_coordinates

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Orbital Agronomy - Stress Detection",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Orbital Agronomy: AI-Powered Crop Stress Detection")
st.markdown("**Real-time satellite imagery analysis for precision agriculture**")

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.radio(
    "Select Active AI Core:",
    ["🧠 Deep Learning 1D-CNN (Primary)", "🌲 Random Forest (Backup/Fast)"],
    index=1  # Default to RF since CNN might not be trained yet
)

# --- SIDEBAR: REGION SELECTION ---
st.sidebar.header("📍 Target Region")
region_name = st.sidebar.selectbox(
    "Select Farm Region:",
    ["Nagpur Farms, India", "Central Valley, California", "Kyiv Region, Ukraine"]
)

# --- REGION COORDINATES ---
region_coords = {
    "Nagpur Farms, India": [21.1458, 79.0882],
    "Central Valley, California": [36.7783, -119.4179],
    "Kyiv Region, Ukraine": [50.4501, 30.5234]
}

# --- STRESS DETECTION FUNCTION ---
def predict_stress(region_name, active_model):
    """Routes to the correct model based on user selection."""
    
    # Define paths relative to app.py
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'stress_rf_model.pkl')
    CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'stress_cnn.pt')
    CNN_SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'cnn_scaler.pkl')
    
    # Route to the correct TIF file based on region selection
    if region_name == "Nagpur Farms, India":
        image_path = os.path.join(PROJECT_ROOT, 'data', 'sample_nagpur.tif')
    elif region_name == "Central Valley, California":
        image_path = os.path.join(PROJECT_ROOT, 'data', 'sample_california.tif')
    else:  # Kyiv
        image_path = os.path.join(PROJECT_ROOT, 'data', 'sample_kyiv.tif')
    
    # Route to the correct inference function
    if "Random Forest" in active_model:
        # Use Random Forest
        if os.path.exists(RF_MODEL_PATH):
            try:
                return get_rf_stressed_coordinates(image_path, RF_MODEL_PATH), "Random Forest"
            except Exception as e:
                st.error(f"RF Inference failed: {e}")
                return generate_fallback_data(region_name), "Random Forest (Fallback)"
        else:
            st.warning("⚠️ Random Forest model not found. Using simulated data.")
            return generate_fallback_data(region_name), "Random Forest (Simulated)"
    else:
        # Use CNN
        if os.path.exists(CNN_MODEL_PATH) and os.path.exists(CNN_SCALER_PATH):
            try:
                return get_cnn_stressed_coordinates(image_path, CNN_MODEL_PATH, CNN_SCALER_PATH), "CNN"
            except Exception as e:
                st.error(f"CNN Inference failed: {e}")
                return generate_fallback_data(region_name), "CNN (Fallback)"
        else:
            st.warning("⚠️ CNN model or scaler not found. Using simulated data.")
            return generate_fallback_data(region_name), "CNN (Simulated)"

def generate_fallback_data(region_name):
    """Generate simulated stress points for demo purposes."""
    center = region_coords.get(region_name, [0, 0])
    return [[center[0] + np.random.normal(0, 0.01), center[1] + np.random.normal(0, 0.01), 0.8] for _ in range(50)]

# --- MAP CREATION ---
def create_map(center_coords, region_name, show_stress=False, active_model="Random Forest"):
    """Creates a Folium map with optional stress heatmap."""
    m = folium.Map(
        location=center_coords,
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    # Add marker for region center
    folium.Marker(
        location=center_coords,
        popup=f"<b>{region_name}</b>",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    
    if show_stress:
        stress_data, model_used = predict_stress(region_name, active_model)
        if stress_data:
            HeatMap(
                stress_data,
                name="Crop Stress Layer",
                min_opacity=0.3,
                radius=15,
                blur=25,
                max_zoom=1
            ).add_to(m)
            st.sidebar.success(f"✅ Stress detection complete using {model_used}")
        else:
            st.sidebar.warning("⚠️ No stress points detected")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📡 {region_name}")
    
    # Toggle for stress visualization
    show_stress = st.checkbox("🔍 Show Crop Stress Visualization", value=True)
    
    # Create and display map
    center = region_coords[region_name]
    m = create_map(center, region_name, show_stress=show_stress, active_model=model_choice)
    st_folium(m, width=700, height=600)

with col2:
    st.subheader("📊 Analysis Summary")
    
    # Display model info
    st.markdown(f"**Active Model:** {model_choice}")
    
    if "CNN" in model_choice:
        st.markdown("""
        🧠 **Deep Learning 1D-CNN**
        - Neural network trained on spectral bands
        - Captures complex patterns
        - Requires normalized data
        """)
    else:
        st.markdown("""
        🌲 **Random Forest Classifier**
        - Ensemble tree-based model
        - Fast inference
        - No preprocessing required
        """)
    
    st.markdown("---")
    
    # Display region stats
    st.markdown(f"**Region:** {region_name}")
    st.markdown(f"**Center Coords:** {center[0]:.4f}, {center[1]:.4f}")
    st.markdown(f"**Zoom Level:** 10")
    
    st.markdown("---")
    
    # Instructions
    st.markdown("""
    ### 📋 How to Use:
    1. Select a farm region
    2. Choose a model (CNN or Random Forest)
    3. Enable stress visualization
    4. Red areas = high stress zones
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
**Orbital Agronomy v1.0** | Powered by Scikit-Learn & PyTorch | 🛰️ Precision Agriculture
""")