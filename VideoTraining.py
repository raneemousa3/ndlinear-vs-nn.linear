

# train_video.py

# Replace these two:
# from videoAnalyser import BaselineVideoModel, NdVideoModel
# with:
from ResNet18Model import ResNetBaselineVideoModel, ResNetNdVideoModel

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from videoAnalyser import CachedVideoDataset

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_samples = 0.0, 0
    for vids, labs in loader:
        vids, labs = vids.to(device), labs.to(device)
        optimizer.zero_grad()
        loss = criterion(model(vids), labs)
        loss.backward()
        optimizer.step()
        total_loss   += loss.item() * vids.size(0)
        total_samples+= vids.size(0)
    return total_loss / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for vids, labs in loader:
            vids, labs = vids.to(device), labs.to(device)
            loss = criterion(model(vids), labs)
            total_loss   += loss.item() * vids.size(0)
            total_samples+= labs.size(0)
    return total_loss / total_samples

def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for vids, labs in loader:
            vids, labs = vids.to(device), labs.to(device)
            preds = model(vids).argmax(dim=1)
            correct += (preds == labs).sum().item()
            total   += labs.size(0)
    return correct / total

def main():
    # — Setup —
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = "/Users/raneemmousa/Desktop/NdLinear/ndlinear-vs-nn.linear/frame_cached"

    # 1) Load cached dataset
    ucf_cached = CachedVideoDataset(cache_dir=cache_dir)

    # 2) Split into train/val
    total   = len(ucf_cached)
    train_n = int(0.8 * total)
    val_n   = total - train_n
    train_ds, val_ds = random_split(ucf_cached, [train_n, val_n])

    # 3) DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_ds,  batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,    batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # 4) Models
    num_classes = len(set(ucf_cached.labels))
    baseline = ResNetBaselineVideoModel(
    hidden_dim=32,
    num_classes=num_classes,
    freeze_backbone=True    # or False if you want to fine‑tune
    ).to(device)
    ndlin = ResNetNdVideoModel(
    hidden_dim=32,
    num_classes=num_classes,
    freeze_backbone=True
).to(device)

    
    # 5) Criterion & Opts
    criterion   = nn.CrossEntropyLoss()
    base_opt    = optim.Adam(baseline.parameters(), lr=1e-3)
    nd_opt      = optim.Adam(ndlin.parameters(),    lr=1e-3)

    # — Train & Evaluate —
    epochs = 10
    b_losses, b_accs = [], []
    n_losses, n_accs = [], []

    for epoch in range(1, epochs+1):
        bt = train_epoch(baseline, train_loader, base_opt, criterion, device)
        nt = train_epoch(ndlin,    train_loader, nd_opt,   criterion, device)
        bv = evaluate(baseline,  val_loader, criterion, device)
        nv = evaluate(ndlin,     val_loader, criterion, device)
        ba = accuracy(baseline,  val_loader, device)
        na = accuracy(ndlin,     val_loader, device)

        b_losses.append(bv); n_losses.append(nv)
        b_accs.append(ba);    n_accs.append(na)

        print(
          f"Epoch {epoch:2d} | "
          f"ResnetBaseline ▶ t-loss {bt:.4f}, v-loss {bv:.4f}, v-acc {ba:.3f} | "
          f"ResnetNdLinear ▶ t-loss {nt:.4f}, v-loss {nv:.4f}, v-acc {na:.3f}"
        )

    # — Plot —
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(1,epochs+1), b_losses, '-o', label='Baseline Val Loss using resnet18')
    plt.plot(range(1,epochs+1), n_losses, '-o', label='NdLinear Val Loss using resnet18')
    plt.title("Val Loss"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1,epochs+1), b_accs, '-o', label='Baseline Val Acc')
    plt.plot(range(1,epochs+1), n_accs, '-o', label='NdLinear Val Acc')
    plt.title("Val Acc"); plt.xlabel("Epoch"); plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
