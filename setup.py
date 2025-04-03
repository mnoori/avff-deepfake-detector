from setuptools import setup, find_packages

setup(
    name="avff-deepfake-detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "torchaudio>=0.9.0",
        "transformers>=4.11.0",
        "numpy>=1.19.5",
        "opencv-python>=4.5.3",
        "scipy>=1.7.1",
        "matplotlib>=3.4.3",
        "pytube>=12.0.0",
        "moviepy>=1.0.3",
        "librosa>=0.8.1",
        "tqdm>=4.62.3",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Audio-Visual Feature Fusion for Deepfake Detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/avff-deepfake-detector",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 