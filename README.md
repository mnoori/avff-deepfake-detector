# AVFF Deepfake Detector

An Audio-Visual Feature Fusion approach for deepfake video detection using both audio and visual features.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/avff-deepfake-detector.git
cd avff-deepfake-detector
```

2. Install the package:
```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from avff import AVFFModel, VideoProcessor, AudioProcessor

# Initialize processors
video_processor = VideoProcessor()
audio_processor = AudioProcessor()

# Process video and audio
video_frames = video_processor.extract_frames("path/to/video.mp4")
audio_features = audio_processor.extract_audio("path/to/video.mp4")

# Initialize model
model = AVFFModel()

# Make prediction
prediction = model(audio_features, video_frames)
```

### Downloading YouTube Videos

```python
from avff import download_youtube_video

# Download a YouTube video
video_path = download_youtube_video("https://www.youtube.com/watch?v=your_video_id")
```

### Visualization

```python
from avff import visualize_results

# Visualize results
visualize_results(
    video_path="path/to/video.mp4",
    audio_features=audio_features,
    visual_features=video_frames,
    prediction=prediction,
    save_path="results.png"
)
```

## Project Structure

```
avff-deepfake-detector/
├── avff/
│   ├── __init__.py
│   ├── models.py
│   ├── preprocessing.py
│   └── utils.py
├── example.py
├── setup.py
└── README.md
```

## Dependencies

- torch >= 1.9.0
- torchvision >= 0.10.0
- torchaudio >= 0.9.0
- transformers >= 4.11.0
- numpy >= 1.19.5
- opencv-python >= 4.5.3
- scipy >= 1.7.1
- matplotlib >= 3.4.3
- pytube >= 12.0.0
- moviepy >= 1.0.3
- librosa >= 0.8.1
- tqdm >= 4.62.3

## License

MIT License 