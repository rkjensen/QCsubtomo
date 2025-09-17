import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from src.models import Encoder3D, Projector


def nt_xent_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    # mask diagonal
    mask = (~torch.eye(2*batch_size, dtype=torch.bool, device=z.device)).float()
    exp_sim = torch.exp(sim) * mask
    # positive pairs
    pos = torch.exp((z1 * z2).sum(dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    denom = exp_sim.sum(dim=1)
    loss = -torch.log(pos / denom)
    return loss.mean()

def make_augment():
    def aug(x):
        # x is numpy array
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=0).copy()
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1).copy()
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2).copy()
        # noise
        x = x + np.random.normal(0, np.random.uniform(0.01,0.2), size=x.shape).astype(x.dtype)
        # blur
        if np.random.rand() < 0.5:
            sigma = np.random.uniform(0,1.0)
            from scipy.ndimage import gaussian_filter
            x = gaussian_filter(x, sigma=sigma)
        # intensity
        x = x * np.random.uniform(0.8,1.2)
        return x
    return aug

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, items, out_shape=(64,64,64), transform=None):
        self.items = items
        self.out_shape = out_shape
        self.transform = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        info = self.items[idx]
        vol = np.array(torch.load(info['path']) if info['path'].endswith('.pt') else np.load(info['path'])) if False else None
        # we will assume items are mrc paths -> use dataset loader for consistency
        from src.dataset import load_mrc, center_crop_or_pad
        vol = load_mrc(info['path'])
        vol = center_crop_or_pad(vol, self.out_shape)
        vol = vol - vol.mean()
        std = vol.std()
        if std > 0:
            vol = vol / std
        # produce two augmentations
        a = self.transform(vol)
        b = self.transform(vol)
        return torch.from_numpy(a[None,...].astype(np.float32)), torch.from_numpy(b[None,...].astype(np.float32))


def train(args):
    # collect files labeled in subfolders 'usable' and 'junk' OR everything in data-dir
    usable = []
    junk = []
    for cls in ['usable','junk']:
        p = Path(args.data_dir) / cls
        if p.is_dir():
            for fn in p.iterdir():
                if fn.suffix == '.mrc':
                    usable.append({'path': os.path.join(p, fn)})
    # fallback: take all files in data-dir
    if len(usable) == 0:
        for fn in os.listdir(args.data_dir):
            if fn.lower().endswith('.mrc'):
                usable.append({'path': fn})

    transform = make_augment()
    ds = ContrastiveDataset(usable, out_shape=(args.size,args.size,args.size), transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder3D(in_ch=1, base_ch=args.base_ch, embed_dim=args.embed_dim).to(device)
    projector = Projector(args.embed_dim, proj_dim=args.proj_dim).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        encoder.train(); projector.train()
        losses = []
        for a,b in tqdm(loader, desc=f"Epoch {epoch}"):
            a = a.to(device); b = b.to(device)
            za = projector(encoder(a))
            zb = projector(encoder(b))
            loss = nt_xent_loss(za, zb, temperature=args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch} mean loss {np.mean(losses):.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'encoder': encoder.state_dict(), 'projector': projector.state_dict()}, args.out)
    print('Saved', args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--proj-dim', type=int, default=128)
    parser.add_argument('--base-ch', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.1)
    args = parser.parse_args()
    train(args)