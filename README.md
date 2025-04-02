# AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection

This project implements the Audio-Visual Feature Fusion (AVFF) approach for video deepfake detection as described in the CVPR 2024 paper "AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection".

## Overview

AVFF is a novel approach that combines audio and visual features to improve the robustness of deepfake detection. The method leverages the inherent relationship between audio and visual modalities to better identify manipulated content.

## Features

- Audio-visual feature extraction and fusion
- Multi-modal deepfake detection
- Support for various video formats
- Pre-trained model weights (coming soon)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/avff-deepfake-detector.git
cd avff-deepfake-detector
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
avff-deepfake-detector/
├── data/               # Dataset directory
├── models/            # Model implementations
├── utils/             # Utility functions
├── train.py           # Training script
├── evaluate.py        # Evaluation script
└── requirements.txt   # Project dependencies
```

## Usage

1. Prepare your dataset in the `data` directory
2. Train the model:
```bash
python train.py --config configs/default.yaml
```

3. Evaluate the model:
```bash
python evaluate.py --model_path checkpoints/model.pth
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{oorloff2024avff,
    title={AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection},
    author={Oorloff, et al.},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 