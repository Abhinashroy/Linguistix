import torch
import torch.nn as nn
import numpy as np

def standardise_data(dataset, mean=None, std_dev=None):
    """
    Standardize the dataset using mean and standard deviation.
    If mean and std_dev are not provided, they are calculated from the dataset.
    """
    if mean is None or std_dev is None:
        mean = np.mean(dataset, axis=0)
        std_dev = np.std(dataset, axis=0)
    std_data = (dataset - mean) / std_dev
    return std_data, mean, std_dev

class ANN(nn.Module):
    """
    Neural Network architecture with two hidden layers using ReLU and Tanh activations.
    The model was trained to achieve high accuracy on the classification task.
    """
    def __init__(self, input_size, hidden_size=128, output_size=50):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # First hidden layer with ReLU
        x = torch.tanh(self.fc2(x))  # Second hidden layer with Tanh
        x = self.fc3(x)  # Logits (raw scores before softmax)
        return x