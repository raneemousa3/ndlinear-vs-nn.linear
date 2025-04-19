import cv2
import os
from torch.utils.data import Dataset
from ndLinearVideo import NdVideoModel, BaselineVideoModel
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
print(torch.cuda.is_available())


class UCF101Dataset(Dataset):
    def __init__(self, root_dir, num_frames=8, frame_size=(64, 64), transform=None):
        """
        root_dir: Directory where UCF101 action class folders are stored.
        num_frames: Number of frames to extract per video.
        frame_size: The desired frame size as a tuple (width, height).
        transform: Optional transform to apply to the video tensor.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        
        # Get sorted list of class folders
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        # Create a mapping from class name to label (e.g., 'Archery' -> 0, 'BabyCrawling' -> 1, etc.)
        self.class_to_label = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
        # Create a list of (video_path, label) tuples
        self.video_paths = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.avi'):  # adjust extension if necessary
                    self.video_paths.append(os.path.join(class_dir, file))
                    self.labels.append(self.class_to_label[class_name])
                    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video_tensor = self.extract_frames(video_path)
        if self.transform:
            video_tensor = self.transform(video_tensor)
        return video_tensor, label
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Generate exactly num_frames indices (linspace repeats indices if total_frames < num_frames)
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        else:
            indices = np.zeros(self.num_frames, dtype=int)

        frames = []
        frame_idx = 0
        next_ptr = 0

        while next_ptr < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx == indices[next_ptr]:
                # Resize & convert
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                next_ptr += 1
            frame_idx += 1

        cap.release()

        # If too few frames, pad by repeating last or black frames
        if len(frames) < self.num_frames:
            if len(frames) == 0:
                black = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
                frames = [black.copy() for _ in range(self.num_frames)]
            else:
                last = frames[-1]
                while len(frames) < self.num_frames:
                    frames.append(last)

        video_array = np.array(frames, dtype=np.float32) / 255.0
        # Shape: (num_frames, H, W, C)
        return torch.from_numpy(video_array)

class CachedVideoDataset(Dataset):
    """
    Loads pre‑cached frame tensors instead of decoding video each time.
    Expects frame tensors saved as 000000.pt, 000001.pt, … plus a labels.pt file.
    """
    def __init__(self, cache_dir, transform=None):
        self.cache_dir = cache_dir
        self.labels    = torch.load(os.path.join(cache_dir, "labels.pt"))
        self.num_samples = len(self.labels)
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video_tensor = torch.load(os.path.join(self.cache_dir, f"{idx:06d}.pt"))
        label = self.labels[idx]
        if self.transform:
            video_tensor = self.transform(video_tensor)
        return video_tensor, label
