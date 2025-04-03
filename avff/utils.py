from pytube import YouTube
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

def download_youtube_video(url, output_path="."):
    """
    Download a YouTube video given its URL.
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the video
        
    Returns:
        str: Path to the downloaded video file
    """
    yt = YouTube(url)
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    video_path = video.download(output_path)
    return video_path

def visualize_results(video_path, audio_features, visual_features, prediction, save_path=None):
    """
    Visualize the input video frames and model predictions.
    
    Args:
        video_path (str): Path to the input video
        audio_features (torch.Tensor): Extracted audio features
        visual_features (torch.Tensor): Extracted visual features
        prediction (torch.Tensor): Model prediction
        save_path (str, optional): Path to save the visualization
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot audio features
    audio_features = audio_features.squeeze().cpu().numpy()
    axes[0, 0].imshow(audio_features, aspect='auto')
    axes[0, 0].set_title('Audio Features')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Frequency')
    
    # Plot visual features
    visual_features = visual_features.squeeze().cpu().numpy()
    axes[0, 1].imshow(visual_features, aspect='auto')
    axes[0, 1].set_title('Visual Features')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Feature Dimension')
    
    # Plot prediction probabilities
    probs = torch.softmax(prediction, dim=1).squeeze().cpu().numpy()
    axes[1, 0].bar(['Real', 'Fake'], probs)
    axes[1, 0].set_title('Prediction Probabilities')
    axes[1, 0].set_ylim(0, 1)
    
    # Add video frames
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[1, 1].imshow(frame)
        axes[1, 1].set_title('Sample Frame')
        axes[1, 1].axis('off')
    cap.release()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        device (str): Device to load the model on
        
    Returns:
        AVFFModel: Loaded model
    """
    model = AVFFModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model 