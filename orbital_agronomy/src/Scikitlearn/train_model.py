import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_stress_vision_model(csv_path, model_output_path):
    print("Loading the massive dataset... (this takes a moment)")
    df = pd.read_csv(csv_path)

    print(f"Total pixels loaded: {len(df)}")
    print("\nClass distribution before balancing:")
    print(df['Stress_Label'].value_counts())

    # HACKATHON OPTIMIZATION: 
    # 2.3 million rows takes too long to train. We sample 50,000 pixels 
    # per category to perfectly balance the data and train super fast.
    print("\nBalancing and sampling data to speed up training...")
    df_sampled = df.groupby('Stress_Label').sample(n=50000, random_state=42, replace=True)

    # Separate Features (Spectral Bands) and Target (Labels)
    X = df_sampled.drop('Stress_Label', axis=1)
    y = df_sampled['Stress_Label']

    print("Splitting into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining the Random Forest 'Stress-Vision' Model...")
    print("Using all CPU cores to train as fast as possible...")
    # n_jobs=-1 tells your laptop to use all its processing cores
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\nEvaluating Model Accuracy on unseen data...")
    y_pred = model.predict(X_test)
    
    print("\n================ MODEL PERFORMANCE ================")
    print(classification_report(y_test, y_pred))
    print("===================================================")

    print(f"\nSaving the trained brain to {model_output_path}...")
    joblib.dump(model, model_output_path)
    print("SUCCESS! Model is ready for the visualization phase.")

# --- RUN THE SCRIPT ---
# Go up 2 levels from Scikitlearn/ to reach orbital_agronomy/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'master_training_data.csv')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'stress_vision_model.pkl')

if __name__ == "__main__":
    train_stress_vision_model(CSV_DATA_PATH, MODEL_SAVE_PATH)