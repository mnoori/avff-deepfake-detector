import os
import json
import yaml
import torch
import logging
from tqdm import tqdm
from pathlib import Path
from avff.preprocessing import VideoProcessor, AudioProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)

def process_video(args):
    video_path, output_dir, video_processor, audio_processor = args
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        logging.info(f"Starting to process video: {video_name}")
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        logging.info(f"Created output directory: {video_output_dir}")
        
        # Process video frames
        logging.info(f"Extracting frames from {video_name}")
        frames = video_processor.extract_frames(video_path)
        frames_path = os.path.join(video_output_dir, 'frames.pt')
        torch.save(frames, frames_path)
        logging.info(f"Saved frames to {frames_path} with shape {frames.shape}")
        
        # Process audio
        logging.info(f"Processing audio from {video_name}")
        audio = audio_processor.process_audio(video_path)
        audio_path = os.path.join(video_output_dir, 'audio.pt')
        torch.save(audio, audio_path)
        logging.info(f"Saved audio to {audio_path} with shape {audio.shape}")
        
        return True
    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}", exc_info=True)
        return False

def main():
    logging.info("Starting preprocessing pipeline")
    
    # Load configuration
    try:
        with open('configs/dfdc_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Loaded configuration successfully")
    except Exception as e:
        logging.error(f"Failed to load config: {str(e)}", exc_info=True)
        return
    
    # Setup directories
    data_root = config['dataset']['train_root']
    metadata_path = config['dataset']['train_metadata']
    output_dir = os.path.join(data_root, 'processed')
    
    logging.info(f"Data root: {data_root}")
    logging.info(f"Metadata path: {metadata_path}")
    logging.info(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    # Initialize processors
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()
    logging.info("Initialized video and audio processors")
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Loaded metadata with {len(metadata)} entries")
    except Exception as e:
        logging.error(f"Failed to load metadata: {str(e)}", exc_info=True)
        return
    
    # Process all videos
    logging.info(f"Processing {len(metadata)} videos")
    
    # Process videos with progress bar
    successful = 0
    failed = 0
    
    for filename, info in tqdm(metadata.items(), desc="Processing videos"):
        video_path = os.path.join(data_root, filename)
        logging.info(f"\nProcessing video: {filename}")
        
        if process_video((video_path, output_dir, video_processor, audio_processor)):
            successful += 1
        else:
            failed += 1
    
    logging.info(f"\nProcessing complete. Successful: {successful}, Failed: {failed}")
    
    # Save processing metadata
    processing_metadata = {
        'total_videos': len(metadata),
        'successful': successful,
        'failed': failed,
        'output_dir': output_dir,
        'data_root': data_root,
        'metadata_path': metadata_path
    }
    
    metadata_file = os.path.join(output_dir, 'processing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(processing_metadata, f, indent=2)
    logging.info(f"Saved processing metadata to {metadata_file}")

if __name__ == "__main__":
    main() 