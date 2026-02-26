import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# --- CONFIGURATION ---
# Get the project root directory (two levels up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, 'orbital_agronomy/data/datasets/master_training_data.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'orbital_agronomy/models')

# --- CNN ARCHITECTURE ---
class StressCNN(nn.Module):
    """1D-CNN for crop stress classification from spectral bands."""
    def __init__(self, input_size, num_classes=3):
        super(StressCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        flat_size = 64 * (input_size // 4)
        
        self.fc1 = nn.Linear(flat_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_cnn_model():
    print("--- ORBITAL AGRONOMY: CNN TRAINING PROTOCOL STARTED ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run image_to_tabular.py first to generate the dataset!")
        return
    
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Data Loaded: {df.shape[0]} samples | {df.shape[1]} features")
    
    # 2. Preprocess
    X = df.drop('Stress_Label', axis=1).values
    y = df['Stress_Label'].values
    
    # Encode labels
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Mapping: {label_mapping}")
    
    # 3. CRITICAL: Fit scaler on training data ONLY
    # The scaler must be saved and used during inference!
    print("Scaling data with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Split Data
    print("Splitting data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    
    # 5. Prepare PyTorch DataLoaders
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Reshape for 1D-CNN: (batch_size, 1, num_features)
    X_train_tensor = X_train_tensor.unsqueeze(1)
    X_test_tensor = X_test_tensor.unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 6. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = X_train.shape[1]
    num_classes = len(le.classes_)
    model = StressCNN(input_size, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 7. Training Loop
    print("Training CNN model...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, "
              f"Test Accuracy: {accuracy:.2f}%")
    
    # 8. Save Model AND Scaler (CRITICAL!)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model_path = os.path.join(MODEL_DIR, 'stress_cnn.pt')
    scaler_path = os.path.join(MODEL_DIR, 'cnn_scaler.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'cnn_label_encoder.pkl')
    
    # Save model state
    print(f"Saving CNN model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    
    # Save scaler (CRITICAL for inference!)
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    # Save label encoder
    print(f"Saving label encoder to {encoder_path}...")
    joblib.dump(le, encoder_path)
    
    print("--- CNN TRAINING COMPLETE ---")
    print(f"✅ Model, scaler, and encoder saved to {MODEL_DIR}")
    print("⚠️ REMEMBER: Always use the saved scaler during inference!")

if __name__ == "__main__":
    train_cnn_model()
