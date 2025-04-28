# Download script - save as download_model.py
import torch
from torchvision import models
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

def download_deepfake_model():
    # Option 1: Use a deepfake detection model from HuggingFace
    try:
        model_name = "selimsef/dfdc_deepfake_challenge"
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # Save the model
        torch.save(model.state_dict(), "deepfake_detector_huggingface.pth")
        print("Downloaded and saved HuggingFace deepfake detection model")
        return True
    except Exception as e:
        print(f"Error downloading HuggingFace model: {e}")
        
        # Fallback to Option 2
        return download_fallback_model()

def download_fallback_model():
    # Option 2: Use a pre-trained ResNet and save it (as a fallback)
    try:
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
        
        # Save the model
        torch.save(model.state_dict(), "deepfake_detector_resnet50.pth")
        print("Saved fallback ResNet model for deepfake detection")
        return True
    except Exception as e:
        print(f"Error creating fallback model: {e}")
        return False

if __name__ == "__main__":
    download_deepfake_model()