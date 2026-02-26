import rasterio
import pandas as pd
import numpy as np
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def process_folder_dataset(train_folder, output_csv):
    print(f"Scanning directories in {train_folder}...")
    
    all_pixels_list = []
    
    # The subfolders in 'train' are our labels ('Health', 'Other', 'Rust')
    labels = [d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]
    
    for label in labels:
        label_folder = os.path.join(train_folder, label)
        print(f"\nProcessing category: {label}...")
        
        # Loop through every .tif file in the category folder
        for image_name in os.listdir(label_folder):
            if not image_name.endswith('.tif'):
                continue
                
            image_path = os.path.join(label_folder, image_name)
            
            try:
                with rasterio.open(image_path) as src:
                    img_data = src.read()
                    num_bands = src.count
                    
                    # Flatten the 3D cube into a 2D table
                    flattened_data = img_data.reshape(num_bands, -1).transpose()
                    
                    column_names = [f"Band_{i+1}" for i in range(num_bands)]
                    df_temp = pd.DataFrame(flattened_data, columns=column_names)
                    
                    # Drop background/black pixels (0 values)
                    df_temp = df_temp.loc[(df_temp != 0).any(axis=1)]
                    
                    # Apply the folder name as the label
                    df_temp['Stress_Label'] = label
                    
                    all_pixels_list.append(df_temp)
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

    print("\nConcatenating all data (this might take a few minutes)...")
    master_df = pd.concat(all_pixels_list, ignore_index=True)
    
    print(f"Saving to {output_csv}...")
    master_df.to_csv(output_csv, index=False)
    print(f"SUCCESS! Saved {len(master_df)} pixel signatures.")
    
    return master_df

# --- RUN THE SCRIPT ---
TRAIN_DIRECTORY = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'train')
FINAL_ML_DATASET = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'master_training_data.csv')

if __name__ == "__main__":
    print(f"Train Directory: {TRAIN_DIRECTORY}")
    print(f"Output CSV: {FINAL_ML_DATASET}")
    df = process_folder_dataset(TRAIN_DIRECTORY, FINAL_ML_DATASET)