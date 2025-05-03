import torch
import numpy as np
import os
from ml_website.model import ANN, save_model

def main():
    """Generate sample model and feature files for testing"""
    print("Generating test model and feature files...")
    
    # Define paths
    model_dir = os.path.join('ml_website', 'models')
    data_dir = os.path.join('ml_website', 'data')
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Model parameters
    input_size = 17  # 13 MFCCs + 4 spectral features
    hidden_size = 128
    output_size = 10  # 10 speakers for test
    
    # Create a sample model
    model = ANN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    
    # Generate sample feature data
    num_samples = 100
    X_features = np.random.randn(num_samples, input_size).astype(np.float32)
    
    # Generate sample labels (0-9 for 10 speakers)
    y_labels = np.array([f"Speaker_{i+1}" for i in range(output_size)])
    
    # Save model
    model_path = os.path.join(model_dir, 'ann_model.pth')
    save_model(model, model_path)
    print(f"Sample model saved to {model_path}")
    
    # Save feature data for standardization
    features_path = os.path.join(data_dir, 'X_features.npy')
    np.save(features_path, X_features)
    print(f"Sample features saved to {features_path}")
    
    # Save labels
    labels_path = os.path.join(data_dir, 'y_labels.npy')
    np.save(labels_path, y_labels)
    print(f"Sample labels saved to {labels_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()