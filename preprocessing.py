# pip install 'git+https://github.com/facebookresearch/detectron2.git'
#pip install openface
import cv2
import numpy as np
# Replace with chosen CPU-friendly gaze estimation model implementation
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer

# Replace with chosen CPU-friendly emotion recognition model

import openface  # Example using OpenFace
import torch

import cv2
import torch
from tqdm import tqdm  # Install with `pip install tqdm` (optional for progress bar)
import numpy as np

def preprocess_video(video_path, target_resolution=(480, 320)):
    """
    Preprocesses a video for the pipeline, including resizing and frame extraction, returning tensors.

    Args:
        video_path (str): Path to the video file.
        target_resolution (tuple, optional): Desired resolution for resizing. Defaults to (480, 320).

    Returns:
        list: List of preprocessed video frames (PyTorch tensors).
    """

    cap = cv2.VideoCapture("/content/ABA Therapy_ Daniel - Communication.mp4")
    if not cap.isOpened():
        raise Exception("Error opening video file!")

    frames = []
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count

    # Create progress bar (optional)
    pbar = tqdm(total=num_frames) if num_frames > 0 else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, target_resolution)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)

        frames.append(frame_tensor)

        if pbar:
            pbar.update(1)  # Update progress bar (optional)

    cap.release()
    if pbar:
        pbar.close()  # Close progress bar (optional)

    return frames