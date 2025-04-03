"""
AVFF (Audio-Visual Feature Fusion) Deepfake Detector
A deep learning-based approach for detecting deepfake videos using both audio and visual features.
"""

__version__ = "0.1.0"

from .models import AVFFModel
from .preprocessing import VideoProcessor, AudioProcessor
from .utils import download_youtube_video, visualize_results 