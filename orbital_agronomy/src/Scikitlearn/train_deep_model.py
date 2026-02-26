import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_advanced_deep_model(csv_path, model_output_path, encoder_output_path, scaler_output_path):
    print("Loading the massive dataset...")
    df = pd.read_csv(csv_path)

    print("Balancing data for Deep Learning...")
    # Added .copy() to fix the 'fragmented dataframe' warning
    df_sampled = df.groupby('Stress_Label').sample(n=40000, random_state=42, replace=True).copy()

    # ---------------------------------------------------------
    # ADVANCED FEATURE ENGINEERING: SCIENTIFIC INDICES
    # ---------------------------------------------------------
    print("Calculating scientific Pre-Visual stress indices (NDWI & PRI)...")
    nir_band = df_sampled['Band_90'] + 0.0001 
    swir_band = df_sampled['Band_110'] + 0.0001
    df_sampled['NDWI_Water_Stress'] = (nir_band - swir_band) / (nir_band + swir_band)

    b531 = df_sampled['Band_45'] + 0.0001
    b570 = df_sampled['Band_55'] + 0.0001
    df_sampled['PRI_Photosynthetic_Stress'] = (b531 - b570) / (b531 + b570)

    # =========================================================
    # THE HACKATHON FIX: Clean up math errors (NaNs and Infinities)
    # =========================================================
    df_sampled.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sampled.fillna(0, inplace=True)

    X = df_sampled.drop('Stress_Label', axis=1).values
    y_text = df_sampled['Stress_Label'].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    
    print("Scaling hyperspectral wavelengths...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # BUILDING THE DEEP NEURAL NETWORK (MLP)
    # ---------------------------------------------------------
    print("Building Deep Neural Network architecture...")
    deep_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=30,
        verbose=True, 
        random_state=42
    )

    print("Training the Deep Brain (Watch the terminal for progress)...")
    deep_model.fit(X_train, y_train)

    accuracy = deep_model.score(X_test, y_test)
    print(f"\n===================================================")
    print(f"DEEP NEURAL NETWORK ACCURACY: {accuracy * 100:.2f}%")
    print(f"===================================================")

    print("Saving the Deep Learning ecosystem...")
    joblib.dump(deep_model, model_output_path)
    joblib.dump(label_encoder, encoder_output_path)
    joblib.dump(scaler, scaler_output_path)
    print("SUCCESS! Advanced Deep Learning model is ready.")

# --- RUN THE SCRIPT ---
CSV_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'master_training_data.csv')
DEEP_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'CNN+scikit', 'deep_stress_model.pkl')
ENCODER_PATH = os.path.join(PROJECT_ROOT, 'models', 'CNN+scikit', 'label_encoder.pkl')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'CNN+scikit', 'data_scaler.pkl')

if __name__ == "__main__":
    print(f"CSV Path: {CSV_DATA_PATH}")
    print(f"Deep Model Path: {DEEP_MODEL_PATH}")
    print(f"Encoder Path: {ENCODER_PATH}")
    print(f"Scaler Path: {SCALER_PATH}")
    train_advanced_deep_model(CSV_DATA_PATH, DEEP_MODEL_PATH, ENCODER_PATH, SCALER_PATH)