

import os
import torch
from videoAnalyser import UCF101Dataset  # import the class you already wrote
from videoAnalyser import CachedVideoDataset
from torch.utils.data import random_split, DataLoader

# 1. Create the cache directory
cache_dir = "/Users/raneemmousa/Desktop/NdLinear/ndlinear-vs-nn.linear/frame_cached" # choose a folder on disk
os.makedirs(cache_dir, exist_ok=True)

# 2. Instantiate the *old* dataset and loop through every video
ds = UCF101Dataset(root_dir="/Users/raneemmousa/Downloads/UCF-101",
                   num_frames=8,
                   frame_size=(64,64))
for idx in range(len(ds)):
    video_tensor, label = ds[idx]            # triggers extract_frames once
    torch.save(video_tensor, f"{cache_dir}/{idx:06d}.pt")

# 3. Save the labels list once too
torch.save(ds.labels, f"{cache_dir}/labels.pt")




