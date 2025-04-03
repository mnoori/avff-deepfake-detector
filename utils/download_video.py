import os
import cv2
import numpy as np
import yt_dlp

def download_youtube_video(url, output_path='sample.mp4', duration=2):
    """
    Download a YouTube video and extract first few seconds using yt-dlp
    
    Args:
        url (str): YouTube video URL
        output_path (str): Path to save the video
        duration (int): Duration in seconds to extract
    """
    try:
        # Configure yt-dlp to download best quality with audio
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': 'temp_' + output_path,
            'quiet': True,
        }
        
        # Download video
        print("Downloading video...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Extract first few seconds with audio using ffmpeg
        temp_path = 'temp_' + output_path
        os.system(f'ffmpeg -y -i "{temp_path}" -t {duration} -c:v copy -c:a copy "{output_path}"')
        
        # Clean up temporary file
        os.remove(temp_path)
        
        print(f"Successfully downloaded and processed video to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Example usage with the new URL
    url = "https://www.youtube.com/watch?v=bDs8GbdBYog"
    video_path = download_youtube_video(url, output_path='sample_2.mp4')
    if video_path:
        print(f"Video saved to: {video_path}") 