import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
from werkzeug.utils import secure_filename
from facenet_pytorch import MTCNN
from avff.models import AVFFModel  # Import the correct model
from avff.preprocessing import VideoProcessor, AudioProcessor  # Import preprocessors

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize processors
video_processor = VideoProcessor()
audio_processor = AudioProcessor()

# Initialize face detector
face_detector = MTCNN(keep_all=True, device=device)

# Load the model
def load_model(checkpoint_path):
    try:
        # Initialize model with default parameters
        model = AVFFModel(
            num_classes=2,
            audio_backbone="facebook/wav2vec2-base-960h",
            visual_backbone="google/vit-base-patch16-224",
            fusion_dim=768,
            dropout=0.1
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model
model_path = r'C:\Users\mehdi\avff-deepfake-detector-1\outputs\dfdc\best_model.pth'
print(f"Loading model from: {model_path}")
model = load_model(model_path)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    media_type = request.form.get('mediaType', 'video')
    
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(file_path)
    
    try:
        if media_type == 'audio':
            # Process audio file
            result = process_audio(file_path)
        else:
            # Process video file
            result = process_video(file_path)
        
        # Clean up temporary file
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        return jsonify(result)
    
    except Exception as e:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

def process_video(video_path):
    """Process video file for deepfake detection"""
    # Extract video frames and audio
    video_frames = video_processor.extract_frames(video_path)
    audio_features = audio_processor.process_audio(video_path)
    
    # Add batch dimension
    video_frames = video_frames.unsqueeze(0).to(device)  # [1, num_frames, 3, H, W]
    audio_features = audio_features.unsqueeze(0).to(device)  # [1, sequence_length]
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(video_frames, audio_features)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1)
        
        # Get deepfake probability (class 1)
        deepfake_prob = float(probs[0, 1].item() * 100)
    
    return {
        'isDeepfake': deepfake_prob > 50,
        'confidence': deepfake_prob
    }

def process_audio(audio_path):
    """Process audio file for deepfake detection"""
    # Extract audio features
    audio_features = audio_processor.process_audio(audio_path)
    
    # Create dummy video frames (zeros) since our model expects both modalities
    video_frames = torch.zeros((1, 8, 3, 224, 224), device=device)
    
    # Add batch dimension to audio
    audio_features = audio_features.unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(video_frames, audio_features)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1)
        
        # Get deepfake probability (class 1)
        deepfake_prob = float(probs[0, 1].item() * 100)
    
    return {
        'isDeepfake': deepfake_prob > 50,
        'confidence': deepfake_prob
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000)