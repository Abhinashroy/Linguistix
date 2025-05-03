import os
import numpy as np
import torch
import librosa
from flask import Flask, render_template, request, jsonify
import torch.nn as nn
from werkzeug.utils import secure_filename

# Initialize Flask application
app = Flask(__name__, 
    template_folder='ml_website/templates',
    static_folder='ml_website/static')

# Configure upload settings
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the neural network model architecture
class SpeakerRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpeakerRecognitionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.tanh(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x

# Global variables for model and standardization parameters
model = None
mean_vector = None
std_vector = None
speaker_names = {}  # Dictionary to map speaker IDs to names

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_features(audio_file):
    """Extract MFCC and additional features from audio file"""
    try:
        # Load audio file with librosa
        y, sr = librosa.load(audio_file, sr=None)
        
        # Adjust duration if needed (take middle 3 seconds)
        if len(y) > sr * 3:
            center = len(y) // 2
            y = y[center - int(sr * 1.5):center + int(sr * 1.5)]
        elif len(y) < sr:
            # Pad if too short
            y = np.pad(y, (0, sr - len(y)), 'constant')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Extract additional features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        # Combine all features into one vector
        features = np.concatenate((mfccs_mean, [spectral_centroid, spectral_bandwidth, 
                                               spectral_rolloff, zero_crossing_rate]))
        
        return features
    
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def load_model_and_metadata():
    """Load the trained model and supporting metadata"""
    global model, mean_vector, std_vector, speaker_names
    
    try:
        # Define paths based on the app's root path
        model_dir = os.path.join(app.root_path, 'ml_website', 'model')
        model_path = os.path.join(model_dir, 'speaker_model.pth')
        mean_path = os.path.join(model_dir, 'mean_vector.npy')
        std_path = os.path.join(model_dir, 'std_vector.npy')
        labels_path = os.path.join(model_dir, 'speaker_labels.npy')
        
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            # Create a dummy model with sample data for testing
            input_size = 17  # 13 MFCCs + 4 spectral features
            hidden_size = 128
            num_classes = 50  # Assuming 50 speakers
            model = SpeakerRecognitionModel(input_size, hidden_size, num_classes)
            
            # Create dummy standardization vectors
            mean_vector = np.zeros(input_size)
            std_vector = np.ones(input_size)
            
            # Create dummy speaker names
            speaker_names = {i: f"Speaker {i+1}" for i in range(num_classes)}
        else:
            # Load the trained model
            input_size = 17  # 13 MFCCs + 4 spectral features
            hidden_size = 128
            
            # Load speaker labels to determine number of classes
            if os.path.exists(labels_path):
                labels = np.load(labels_path, allow_pickle=True)
                num_classes = max(labels) + 1
                
                # Create a mapping from label ID to speaker name
                for i in range(num_classes):
                    speaker_names[i] = f"Speaker {i+1}"
            else:
                num_classes = 50  # Default
            
            # Initialize and load the model
            model = SpeakerRecognitionModel(input_size, hidden_size, num_classes)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            
            # Load standardization parameters
            if os.path.exists(mean_path) and os.path.exists(std_path):
                mean_vector = np.load(mean_path)
                std_vector = np.load(std_path)
            else:
                mean_vector = np.zeros(input_size)
                std_vector = np.ones(input_size)
        
        print("Model and metadata loaded successfully")
        
    except Exception as e:
        print(f"Error loading model and metadata: {str(e)}")
        # Set defaults
        input_size = 17
        hidden_size = 128
        num_classes = 50
        model = SpeakerRecognitionModel(input_size, hidden_size, num_classes)
        mean_vector = np.zeros(input_size)
        std_vector = np.ones(input_size)
        speaker_names = {i: f"Speaker {i+1}" for i in range(num_classes)}

# Load the model when the application starts
@app.before_first_request
def setup():
    load_model_and_metadata()

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for model information page
@app.route('/model-info')
def model_info():
    return render_template('model_info.html')

# Route to handle file upload and make predictions
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'audio_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['audio_file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Check if the file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload a WAV file.'})
    
    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract features from the audio file
        features = extract_audio_features(file_path)
        if features is None:
            return jsonify({'success': False, 'error': 'Failed to extract features from the audio file'})
        
        # Standardize the features
        features_standardized = (features - mean_vector) / std_vector
        features_tensor = torch.FloatTensor(features_standardized).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(features_tensor)
            
        # Get the predicted class and confidence
        predicted_class = int(torch.argmax(output).item())
        confidence = float(output[0][predicted_class].item() * 100)
        
        # Get speaker name from dictionary
        speaker_name = speaker_names.get(predicted_class, f"Speaker {predicted_class+1}")
        
        # Delete the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Return the prediction
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'speaker': speaker_name,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Custom error handler for the before_first_request decorator in newer Flask versions
if hasattr(app, 'before_first_request'):
    pass  # Use the decorator as defined
else:
    with app.app_context():
        setup()

if __name__ == '__main__':
    app.run(debug=True)