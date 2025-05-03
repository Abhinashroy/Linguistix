# Neural Network Model Showcase Website

An interactive web application that demonstrates a neural network classification model with ReLU and Tanh activation functions that achieves 100% validation accuracy.

## Features

- Interactive model prediction interface
- Detailed model architecture visualization
- Performance metrics and visualizations
- Technical documentation about the neural network
- Responsive design for all devices

## Project Structure

```
ml_website/
├── app.py              # Main Flask application
├── model.py            # Neural network model definition
├── run.py              # Entry point script
├── requirements.txt    # Project dependencies
├── static/             # Static assets
│   ├── css/            # Stylesheets
│   ├── js/             # JavaScript files
│   └── img/            # Images and SVGs
├── templates/          # HTML templates
│   ├── layout.html     # Base template
│   ├── index.html      # Home page
│   ├── model_info.html # Model information page
│   ├── visualization.html # Visualizations page
│   └── about.html      # About page
├── models/             # Stored model weights
│   └── ann_model.pth   # Trained PyTorch model
└── data/               # Sample data
    ├── X_features.npy  # Feature data
    └── y_labels.npy    # Target labels
```

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd ml_website
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Make sure you have the required model and data files:
   - Place your trained model file (`ann_model.pth`) in the `models/` directory
   - Place your feature data (`X_features.npy`) in the `data/` directory
   - Place your label data (`y_labels.npy`) in the `data/` directory

   Note: The application automatically calculates standardization parameters from your feature data.

## Usage

1. Run the application:
   ```
   python run.py
   ```

2. Open a web browser and go to:
   ```
   http://localhost:5000
   ```

3. Interact with the application:
   - Try model predictions by entering feature values or using random values
   - Explore the model architecture and technical details
   - View performance visualizations

## Model Architecture

The neural network uses a unique combination of activation functions:
- First hidden layer: ReLU activation
- Second hidden layer: Tanh activation
- Output layer: Softmax (for class probabilities)

This architecture achieved 100% validation accuracy on the test dataset.

## Requirements

- Python 3.8+
- PyTorch 2.0.0
- Flask 2.2.3
- NumPy 1.24.2
- Additional dependencies in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The model was trained on feature data
- Feature standardization is performed automatically using the mean and standard deviation calculated from the provided data
- The web interface was built with Flask, Bootstrap, and modern JavaScript