# Lightweight AVFF Deepfake Detector

An Audio-Visual Feature Fusion approach for deepfake video detection using both audio and visual features. This project implements a lightweight version of the state-of-the-art deepfake detection system described in [AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html) (CVPR 2024), leveraging both visual and audio cues to improve detection accuracy.

## Features

- **Multi-modal Analysis**: Combines visual and audio features for robust detection
- **Real-time Processing**: Capable of processing video streams in real-time
- **YouTube Integration**: Direct support for YouTube video analysis
- **Visualization Tools**: Built-in tools for result visualization and analysis
- **Modular Architecture**: Easy to extend and customize
- **Lightweight Implementation**: Optimized for efficiency and ease of use
- **Two-Stage Learning**: Implements the paper's novel two-stage cross-modal learning approach
- **Self-Supervised Learning**: Captures intrinsic audio-visual correspondences
- **Contrastive Learning**: Utilizes contrastive learning and autoencoding objectives
- **Audio-Visual Masking**: Implements the paper's complementary masking strategy

## Implementation Details

This implementation follows the key aspects of the original paper:
1. **Representation Learning Stage**: Self-supervised learning on real videos to capture audio-visual correspondences
2. **Classification Stage**: Supervised learning for deepfake classification
3. **Feature Fusion**: Novel audio-visual complementary masking and feature fusion strategy

## Key Components and Models

### Visual Processing
- **ViT (Vision Transformer)**: Used for visual feature extraction and encoding
- **OpenCV**: For video frame extraction and preprocessing
- **FAISS**: Efficient similarity search for visual embeddings
- **Contrastive Learning**: Implemented using PyTorch for visual feature learning

### Audio Processing
- **Wav2Vec2**: Pre-trained model for audio feature extraction
- **Librosa**: For audio preprocessing and mel spectrogram generation
- **FAISS**: Efficient similarity search for audio embeddings
- **Voice Activity Detection**: Custom implementation for audio segmentation

### Feature Fusion
- **Cross-Modal Attention**: Custom attention mechanism for audio-visual feature fusion
- **Transformer Encoder**: For encoding fused features
- **FAISS**: For efficient similarity search of fused embeddings

### Training and Optimization
- **PyTorch Lightning**: For training pipeline and experiment management
- **Contrastive Loss**: Custom implementation for self-supervised learning
- **Cross-Entropy Loss**: For supervised classification
- **AdamW Optimizer**: With learning rate scheduling
- **Gradient Clipping**: For training stability

### Additional Tools
- **Pytube**: For YouTube video downloading
- **MoviePy**: For video processing and manipulation
- **Matplotlib**: For visualization and analysis
- **Scipy**: For signal processing and feature analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mnoori/lightweight-avff.git
cd lightweight-avff
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
lightweight-avff/
├── avff/                    # Main package
│   ├── __init__.py         # Package initialization
│   ├── models.py           # Model architectures
│   ├── preprocessing.py    # Data preprocessing
│   └── utils.py            # Utility functions
├── example.py              # Usage examples
├── setup.py                # Package setup
└── README.md               # Project documentation
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

@software{Lightweight_AVFF_Deepfake_Detector,
    author = {Your Name},
    title = {Lightweight AVFF Deepfake Detector},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/mnoori/lightweight-avff}}
}
``` 