import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional

def extract_frames(video_path: str,
                  num_frames: int = 16) -> List[np.ndarray]:
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

def extract_face(frame: np.ndarray,
                face_cascade: cv2.CascadeClassifier) -> Optional[np.ndarray]:
    """Extract face from frame using face detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        return face
    return None 