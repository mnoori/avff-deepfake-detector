import torch
from avff import AVFFModel, VideoProcessor, AudioProcessor, download_youtube_video, visualize_results

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize processors
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()
    
    # Download YouTube video
    video_url = "https://www.youtube.com/watch?v=your_video_id"
    video_path = download_youtube_video(video_url)
    
    # Process video and audio
    video_frames = video_processor.extract_frames(video_path)
    audio_features = audio_processor.extract_audio(video_path)
    
    # Move to device
    video_frames = video_frames.to(device)
    audio_features = audio_features.to(device)
    
    # Initialize and load model
    model = AVFFModel().to(device)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        prediction = model(audio_features, video_frames)
    
    # Visualize results
    visualize_results(
        video_path,
        audio_features,
        video_frames,
        prediction,
        save_path="results.png"
    )
    
    # Print prediction
    probs = torch.softmax(prediction, dim=1)
    print(f"Real probability: {probs[0][0]:.4f}")
    print(f"Fake probability: {probs[0][1]:.4f}")

if __name__ == "__main__":
    main() 