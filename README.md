# AVFF Deepfake Detector

An Audio-Visual Feature Fusion approach for deepfake video detection using both audio and visual features. This project implements a lightweight version of the state-of-the-art deepfake detection system described in [AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html) (CVPR 2024), leveraging both visual and audio cues to improve detection accuracy.

## Features

- **Multi-modal Analysis**: Combines visual and audio features for robust detection
- **Web Interface**: User-friendly interface for uploading and analyzing media
- **Real-time Processing**: Capable of processing video streams in real-time
- **YouTube Integration**: Direct support for YouTube video analysis
- **Visualization Tools**: Built-in tools for result visualization and analysis
- **Modular Architecture**: Easy to extend and customize
- **Lightweight Implementation**: Optimized for efficiency and ease of use

## Implementation Details

This implementation follows the key aspects of the original paper:
1. **Representation Learning Stage**: Self-supervised learning on real videos to capture audio-visual correspondences
2. **Classification Stage**: Supervised learning for deepfake classification
3. **Feature Fusion**: Novel audio-visual complementary masking and feature fusion strategy

## Key Components and Models

### Visual Processing
- **ViT (Vision Transformer)**: Used for visual feature extraction and encoding
- **OpenCV**: For video frame extraction and preprocessing
- **Contrastive Learning**: Implemented using PyTorch for visual feature learning

### Audio Processing
- **Wav2Vec2**: Pre-trained model for audio feature extraction
- **Librosa**: For audio preprocessing and mel spectrogram generation
- **Voice Activity Detection**: Custom implementation for audio segmentation

### Feature Fusion
- **Cross-Modal Attention**: Custom attention mechanism for audio-visual feature fusion
- **Transformer Encoder**: For encoding fused features

### Training and Optimization
- **PyTorch Lightning**: For training pipeline and experiment management
- **Contrastive Loss**: Custom implementation for self-supervised learning
- **Cross-Entropy Loss**: For supervised classification
- **AdamW Optimizer**: With learning rate scheduling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mnoori/avff-deepfake-detector.git
cd avff-deepfake-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Run the application to start the web interface:

```bash
python main.py
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface.

### API Usage

```python
from app.app import process_video

# Process a video file
result = process_video("path/to/video.mp4")
print(f"Is deepfake: {result['isDeepfake']}")
print(f"Confidence: {result['confidence']}%")
```

### Python Module Usage

```python
from avff.models import AVFFModel
from avff.preprocessing import VideoProcessor, AudioProcessor

# Initialize processors
video_processor = VideoProcessor()
audio_processor = AudioProcessor()

# Process video and audio
video_frames = video_processor.extract_frames("path/to/video.mp4")
audio_features = audio_processor.process_audio("path/to/video.mp4")

# Initialize model
model = AVFFModel()

# Make prediction
with torch.no_grad():
    outputs = model(video_frames, audio_features)
    logits = outputs['logits']
    probs = torch.softmax(logits, dim=1)
    deepfake_prob = float(probs[0, 1].item() * 100)
```

## Project Structure

```
avff-deepfake-detector/
├── app/                     # Backend application
│   ├── api/                # API endpoints
│   ├── models/             # Data models
│   ├── services/           # Business logic
│   ├── utils/              # Utility functions
│   └── app.py              # Main application
├── avff/                    # Core library
│   ├── models.py           # Model architectures
│   ├── preprocessing.py    # Data preprocessing
│   └── utils.py            # Utility functions
├── client/                  # Frontend application
│   ├── public/             # Static assets
│   └── index.html          # Main HTML interface
├── scripts/                 # Training and utility scripts
│   ├── train.py            # General training script
│   ├── train_dfdc.py       # DFDC dataset training
│   ├── evaluate.py         # Evaluation script
│   └── preprocess_dfdc.py  # DFDC preprocessing
├── docs/                    # Documentation
│   ├── assets/             # Documentation assets
│   └── training_report.md  # Training report
├── configs/                 # Configuration files
├── main.py                  # Application entry point
├── setup.py                 # Package setup
└── README.md                # Project documentation
```

## Dependencies

- torch >= 1.9.0
- torchvision >= 0.10.0
- torchaudio >= 0.9.0
- transformers >= 4.11.0
- flask >= 2.0.0
- flask-cors >= 3.0.0
- numpy >= 1.19.5
- opencv-python >= 4.5.3
- facenet-pytorch >= 2.5.2
- matplotlib >= 3.4.3
- librosa >= 0.8.1

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for more information.

## License

Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite both the original paper and this implementation:

```bibtex
@InProceedings{Oorloff_2024_CVPR,
    author = {Oorloff, Trevine and Koppisetti, Surya and Bonettini, Nicolò and Solanki, Divyaraj and Colman, Ben and Yacoob, Yaser and Shahriyari, Ali and Bharaj, Gaurav},
    title = {AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2024},
    pages = {27102-27112}
}

@software{AVFF_Deepfake_Detector,
    author = {M. Noori},
    title = {AVFF Deepfake Detector},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/mnoori/avff-deepfake-detector}}
}
``` 