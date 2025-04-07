import cv2
import numpy as np
import os

def create_dummy_video(output_path, num_frames=30, size=(224, 224)):
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, size)
    
    # Generate random frames
    for _ in range(num_frames):
        # Create a random frame
        frame = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        
        # Add some moving patterns
        t = _ / num_frames
        x = int(size[0] * (0.5 + 0.3 * np.sin(2 * np.pi * t)))
        y = int(size[1] * (0.5 + 0.3 * np.cos(2 * np.pi * t)))
        cv2.circle(frame, (x, y), 20, (255, 0, 0), -1)
        
        out.write(frame)
    
    out.release()

def main():
    # Create output directory if it doesn't exist
    output_dir = "data/dfdc/train"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dummy videos
    video_names = [
        "sample_real_001.mp4",
        "sample_fake_001.mp4",
        "sample_real_002.mp4",
        "sample_fake_002.mp4"
    ]
    
    for video_name in video_names:
        output_path = os.path.join(output_dir, video_name)
        print(f"Generating {video_name}...")
        create_dummy_video(output_path)
        print(f"Done generating {video_name}")

if __name__ == "__main__":
    main() 