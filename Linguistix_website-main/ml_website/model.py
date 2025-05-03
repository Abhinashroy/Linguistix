import torch
import torch.nn as nn
import numpy as np

def standardize_data(X, mean=None, std=None):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Parameters:
    X (numpy.ndarray): The data to standardize
    mean (numpy.ndarray, optional): Mean values to use for standardization
    std (numpy.ndarray, optional): Standard deviation values to use for standardization
    
    Returns:
    numpy.ndarray: Standardized data
    """
    if mean is None or std is None:
        # Calculate mean and standard deviation
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Handle zero standard deviation
        std = np.where(std == 0, 1.0, std)
        
    # Standardize data
    X_scaled = (X - mean) / std
    return X_scaled

class ANN(nn.Module):
    """
    Artificial Neural Network for speaker identification.
    
    Architecture:
    - Input Layer: Feature vector size
    - Hidden Layer: Configurable size with ReLU activation
    - Output Layer: Number of speakers with softmax activation
    """
    def __init__(self, input_size=17, hidden_size=128, output_size=50):
        """
        Initialize ANN model.
        
        Parameters:
        input_size (int): Size of input feature vector
        hidden_size (int): Size of hidden layer
        output_size (int): Number of output classes (speakers)
        """
        super(ANN, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def predict(self, features):
        """
        Make prediction from features.
        
        Parameters:
        features (torch.Tensor): Input features
        
        Returns:
        tuple: (predicted_class, confidence)
        """
        self.eval()
        with torch.no_grad():
            outputs = self(features)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            return predicted.item(), confidence.item()

def save_model(model, path):
    """
    Save model weights to file.
    
    Parameters:
    model (torch.nn.Module): The model to save
    path (str): Path to save the model
    """
    torch.save(model.state_dict(), path)