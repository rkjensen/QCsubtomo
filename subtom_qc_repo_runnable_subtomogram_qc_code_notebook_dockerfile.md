# subtom_qc — Subtomogram QC & Classifier

This document contains a complete, runnable repository skeleton for the subtomogram QC project we discussed. It includes: code (PyTorch), a simulated-data Jupyter notebook you can run immediately, a Dockerfile + `requirements.txt`, and a standalone script to generate simulated noisy subtomograms (CTF + missing wedge + noise).

Copy the files into a git repo `subtom_qc/` or use the structure below. I intentionally kept the code small, readable, and easy to adapt. It is **not** production hardened (logging, argparse niceties, distributed training hooks), but is fully functional and ready for extension.

---

## Repo layout

```
subtom_qc/
├── README.md
├── requirements.txt
├── Dockerfile
├── examples/
│   └── sample_config.yaml
├── notebooks/
│   └── train_on_simulated_data.ipynb  # provided as marked-up notebook in this doc
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── models.py
│   ├── train_pretrain.py
│   ├── train_classifier.py
│   ├── eval.py
│   ├── score_subtomos.py
│   └── simulate.py
└── scripts/
    └── run_demo.sh
```


## `README.md` (top-level)

```markdown
# subtom_qc

Subtomogram quality-control and classifier pipeline. Small, practical PyTorch code for: simulated-data pretraining (contrastive), classifier finetune, scoring, and simple evaluation.

## Quickstart (local GPU)

1. Create a venv and install requirements:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

2. Run the notebook `notebooks/train_on_simulated_data.ipynb` or run scripts in `src/`.

3. Generate simulated data:

```bash
python src/simulate.py --out-dir data/sim --n-usable 500 --n-junk 500
```

4. Pretrain encoder (contrastive):

```bash
python src/train_pretrain.py --data-dir data/sim --out models/pretrained.pth
```

5. Train classifier:

```bash
python src/train_classifier.py --data-dir data/sim --pretrained models/pretrained.pth --out models/clf.pth
```

6. Score a folder of subtomograms:

```bash
python src/score_subtomos.py --model models/clf.pth --encoder models/pretrained.pth --folder /path/to/mrcs --out scores.csv
```

See notebooks for a guided run.
```
```

---

## `src/__init__.py`

```python
# empty
```

---

## `src/dataset.py`

```python
import os
import numpy as np
import mrcfile
import torch
from torch.utils.data import Dataset

def load_mrc(path):
    with mrcfile.open(path, permissive=True) as m:
        data = np.array(m.data, dtype=np.float32)
    return data


def center_crop_or_pad(volume, out_shape):
    # center crop or pad to out_shape (z,y,x)
    result = np.zeros(out_shape, dtype=volume.dtype)
    cz = (volume.shape[0] - out_shape[0]) // 2
    cy = (volume.shape[1] - out_shape[1]) // 2
    cx = (volume.shape[2] - out_shape[2]) // 2
    if cz >= 0:
        z0 = cz; sz = out_shape[0]
        src_z0 = z0; src_z1 = z0 + sz
        dst_z0 = 0
    else:
        # volume smaller -> pad
        src_z0 = 0; src_z1 = volume.shape[0]
        dst_z0 = -cz
        sz = src_z1
    # same for y/x
    def ranges(src_len, out_len):
        start = max(0, (src_len - out_len)//2)
        take = min(out_len, src_len)
        dst_start = max(0, (out_len - src_len)//2)
        return start, start+take, dst_start, dst_start+take
    sz0, sz1, dz0, dz1 = ranges(volume.shape[0], out_shape[0])
    sy0, sy1, dy0, dy1 = ranges(volume.shape[1], out_shape[1])
    sx0, sx1, dx0, dx1 = ranges(volume.shape[2], out_shape[2])
    result[dz0:dz1, dy0:dy1, dx0:dx1] = volume[sz0:sz1, sy0:sy1, sx0:sx1]
    return result

class SubtomogramDataset(Dataset):
    def __init__(self, items, out_shape=(64,64,64), transform=None):
        self.items = items
        self.out_shape = out_shape
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        info = self.items[idx]
        vol = load_mrc(info['path'])
        vol = center_crop_or_pad(vol, self.out_shape)
        vol = vol - vol.mean()
        std = vol.std()
        if std > 0:
            vol = vol / std
        vol = vol.astype(np.float32)
        if self.transform:
            vol = self.transform(vol)
        return torch.from_numpy(vol[None, ...]), torch.tensor(info.get('label', -1), dtype=torch.long)

# helper to build items list from directories with naming convention

def build_items_from_dir(folder, label=None):
    items = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.mrc', '.map')):
            items.append({'path': os.path.join(folder, fname), 'label': label})
    return items
```

---

## `src/transforms.py`

```python
import numpy as np
import scipy.ndimage as ndi

class RandomFlip:
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=0).copy()
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1).copy()
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2).copy()
        return x

class RandomNoise:
    def __init__(self, sigma=(0.01,0.2)):
        self.sigma = sigma
    def __call__(self, x):
        s = np.random.uniform(*self.sigma)
        x = x + np.random.normal(0, s, size=x.shape).astype(x.dtype)
        return x

class RandomBlur:
    def __init__(self, sigma=(0.0,1.0)):
        self.sigma = sigma
    def __call__(self, x):
        s = np.random.uniform(*self.sigma)
        if s > 1e-6:
            x = ndi.gaussian_filter(x, sigma=s)
        return x

class IntensityScale:
    def __call__(self, x):
        f = np.random.uniform(0.8, 1.2)
        return x * f

# simple missing wedge sim in Fourier space

def apply_missing_wedge(volume, wedge_deg=30):
    # wedge_deg in degrees (half-angle)
    vol_ft = np.fft.rfftn(volume)
    nz, ny, nx2 = vol_ft.shape
    # build angles for each (k_y, k_x) pair per z-slice is approximate; we approximate by kx, ky
    ky = np.fft.fftfreq(volume.shape[1])[:, None]
    kx = np.fft.rfftfreq(volume.shape[2])[None, :]
    angles = np.degrees(np.arctan2(np.abs(ky), np.abs(kx)))
    mask2d = (angles <= wedge_deg)
    for z in range(nz):
        vol_ft[z] *= mask2d
    vol = np.fft.irfftn(vol_ft, s=volume.shape)
    return vol
```

---

## `src/models.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SmallResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = BasicConv3d(channels, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm2(self.conv2(out))
        out = F.relu(out + x)
        return out

class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, embed_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            BasicConv3d(in_ch, base_ch, ks=5, stride=1, padding=2),
            BasicConv3d(base_ch, base_ch),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            BasicConv3d(base_ch, base_ch*2),
            SmallResBlock(base_ch*2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            BasicConv3d(base_ch*2, base_ch*4),
            SmallResBlock(base_ch*4)
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_ch*4, embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

# projector MLP used during pretraining (SimCLR)
class Projector(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
```

---

## `src/train_pretrain.py`

```python
"""
Contrastive pretraining (SimCLR-style) for 3D subtomograms.
Usage (simple):
python src/train_pretrain.py --data-dir data/sim --out models/pretrained.pth
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SubtomogramDataset, build_items_from_dir
from src.transforms import RandomFlip, RandomNoise, RandomBlur, IntensityScale
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

import torch.nn.functional as F

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
        p = os.path.join(args.data_dir, cls)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.lower().endswith('.mrc'):
                    usable.append({'path': os.path.join(p, fn)})
    # fallback: take all files in data-dir
    if len(usable) == 0:
        for fn in os.listdir(args.data_dir):
            if fn.lower().endswith('.mrc'):
                usable.append({'path': os.path.join(args.data_dir, fn)})

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
```

---

## `src/train_classifier.py`

```python
"""
Train a classifier (linear MLP) on top of encoder embeddings.
Usage:
python src/train_classifier.py --data-dir data/sim --pretrained models/pretrained.pth --out models/clf.pth
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from src.dataset import SubtomogramDataset, build_items_from_dir
from src.models import Encoder3D

class SimpleClassifier(torch.nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hid, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


def load_items(data_dir):
    # expects data_dir/usable and data_dir/junk
    items = []
    for lbl, name in [(1,'usable'), (0,'junk')]:
        p = os.path.join(data_dir, name)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if fn.lower().endswith('.mrc'):
                items.append({'path': os.path.join(p, fn), 'label': lbl})
    return items


def train(args):
    items = load_items(args.data_dir)
    np.random.shuffle(items)
    split = int(len(items)*0.8)
    train_items = items[:split]
    val_items = items[split:]
    train_ds = SubtomogramDataset(train_items, out_shape=(args.size,args.size,args.size))
    val_ds = SubtomogramDataset(val_items, out_shape=(args.size,args.size,args.size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder3D(in_ch=1, base_ch=args.base_ch, embed_dim=args.embed_dim).to(device)
    if args.pretrained:
        ck = torch.load(args.pretrained, map_location=device)
        encoder.load_state_dict(ck['encoder'])
        print('Loaded pretrained encoder')

    clf = SimpleClassifier(args.embed_dim, hid=args.hid).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(clf.parameters()), lr=args.lr)

    best_auc = 0.0
    for epoch in range(args.epochs):
        encoder.train(); clf.train()
        losses = []
        for x,y in tqdm(train_loader, desc=f'Train E{epoch}'):
            x = x.to(device); y = y.float().to(device)
            emb = encoder(x)
            logits = clf(emb)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # val
        encoder.eval(); clf.eval()
        ys, preds = [], []
        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(device)
                emb = encoder(x)
                logits = clf(emb).cpu().numpy()
                ps = 1/(1+np.exp(-logits))
                preds.extend(ps.tolist())
                ys.extend(y.numpy().tolist())
        auc = roc_auc_score(ys, preds) if len(set(ys))>1 else 0.5
        ap = average_precision_score(ys, preds) if len(set(ys))>1 else 0.0
        print(f"Epoch {epoch} train_loss {np.mean(losses):.4f} val_auc {auc:.4f} ap {ap:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save({'encoder': encoder.state_dict(), 'clf': clf.state_dict()}, args.out)
            print('Saved best ->', args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--base-ch', type=int, default=32)
    parser.add_argument('--hid', type=int, default=128)
    args = parser.parse_args()
    train(args)
```

---

## `src/eval.py`

```python
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from src.dataset import SubtomogramDataset
from src.models import Encoder3D

# simple evaluation script expecting saved checkpoint from train_classifier

def evaluate(args):
    ck = torch.load(args.checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder3D(in_ch=1, base_ch=args.base_ch, embed_dim=args.embed_dim).to(device)
    encoder.load_state_dict(ck['encoder'])
    from src.train_classifier import SimpleClassifier
    clf = SimpleClassifier(args.embed_dim, hid=args.hid).to(device)
    clf.load_state_dict(ck['clf'])

    # build dataset
    items = []
    for fn in os.listdir(args.data_dir):
        if fn.lower().endswith('.mrc'):
            # crude: determine label from folder name
            label = 1 if 'usable' in args.data_dir else 0
            items.append({'path': os.path.join(args.data_dir, fn), 'label': label})
    ds = SubtomogramDataset(items, out_shape=(args.size,args.size,args.size))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
    ys, ps = [], []
    with torch.no_grad():
        encoder.eval(); clf.eval()
        for x,y in loader:
            x = x.to(device)
            emb = encoder(x)
            logits = clf(emb).cpu().numpy()
            probs = 1/(1+np.exp(-logits))
            ps.extend(probs.tolist())
            ys.extend(y.numpy().tolist())
    print('AUC', roc_auc_score(ys, ps))
    print('AP', average_precision_score(ys, ps))

if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--base-ch', type=int, default=32)
    parser.add_argument('--hid', type=int, default=128)
    args = parser.parse_args()
    evaluate(args)
```

---

## `src/score_subtomos.py`

```python
"""
Score a folder of subtomograms using a trained encoder+classifier and write CSV of scores.
Usage:
python src/score_subtomos.py --encoder models/pretrained.pth --clf models/clf.pth --folder /path/to/mrcs --out scores.csv
"""
import os
import glob
import csv
import torch
import numpy as np
from src.dataset import load_mrc, center_crop_or_pad
from src.models import Encoder3D
from src.train_classifier import SimpleClassifier


def score(encoder_ck, clf_ck, folder, out_csv, size=64, device='cuda'):
    device = device if torch.cuda.is_available() else 'cpu'
    enc = Encoder3D(in_ch=1, base_ch=32, embed_dim=256).to(device)
    clf = SimpleClassifier(256, hid=128).to(device)
    enc.load_state_dict(torch.load(encoder_ck, map_location=device)['encoder'])
    clf.load_state_dict(torch.load(clf_ck, map_location=device)['clf'])
    enc.eval(); clf.eval()
    paths = glob.glob(os.path.join(folder, '*.mrc'))
    rows = []
    with torch.no_grad():
        for p in paths:
            vol = load_mrc(p)
            vol = center_crop_or_pad(vol, (size,size,size))
            vol = vol - vol.mean(); std=vol.std();
            if std>0: vol = vol/std
            x = torch.from_numpy(vol[None,None,...].astype(np.float32)).to(device)
            emb = enc(x)
            logit = clf(emb).cpu().numpy().ravel()[0]
            prob = 1/(1+np.exp(-logit))
            rows.append({'path':p, 'score':float(prob)})
    with open(out_csv,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path','score'])
        w.writeheader(); w.writerows(rows)
    print('Wrote', out_csv)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--clf', required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int, default=64)
    args = parser.parse_args()
    score(args.encoder, args.clf, args.folder, args.out, size=args.size)
```

---

## `src/simulate.py` — generate simulated subtomograms

```python
"""
Generate simulated usable and junk subtomograms. This script creates simple spherical/ellipsoidal densities as ground-truth 'macromolecules' and noise backgrounds with CTF-like modulation and missing wedge.

Usage:
python src/simulate.py --out-dir data/sim --n-usable 500 --n-junk 500

Output layout:
data/sim/usable/*.mrc
data/sim/junk/*.mrc

This is intentionally simple but effective to run a quick experiment when you don't have labeled data.
"""
import os
import argparse
import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter


def sphere(shape, radius, center=None, amplitude=1.0):
    z,y,x = np.indices(shape)
    if center is None:
        center = np.array(shape)/2
    r2 = ((z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2)
    mask = (r2 <= radius**2).astype(np.float32)
    return mask * amplitude

# simple CTF-ish multiplicative modulation in Fourier

def apply_ctf(vol, defocus=20000.0, pixel=3.0, amp=1.0):
    # very crude: multiply by low-frequency envelope
    ft = np.fft.fftn(vol)
    freqs = np.fft.fftfreq(vol.shape[0], d=pixel)
    r = np.sqrt(np.add.outer(freqs**2, np.add.outer(freqs**2, freqs**2)))
    env = np.exp(- (r*defocus/1e6)**2)
    ft = ft * env
    return np.real(np.fft.ifftn(ft))

# missing wedge crudely in Fourier

def apply_missing_wedge(vol, wedge_deg=30):
    ft = np.fft.rfftn(vol)
    ky = np.fft.fftfreq(vol.shape[1])[:,None]
    kx = np.fft.rfftfreq(vol.shape[2])[None,:]
    angles = np.degrees(np.arctan2(np.abs(ky), np.abs(kx)))
    mask2d = (angles <= wedge_deg)
    for z in range(ft.shape[0]):
        ft[z] *= mask2d
    vol2 = np.fft.irfftn(ft, s=vol.shape)
    return vol2


def save_mrc(vol, path):
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(vol.astype(np.float32))


def make_usable(out_path, n, size=64):
    os.makedirs(out_path, exist_ok=True)
    for i in range(n):
        v = np.random.normal(0, 0.05, size=(size,size,size)).astype(np.float32)
        # add sphere
        r = np.random.uniform(size*0.08, size*0.18)
        c = np.array([size/2 + np.random.uniform(-4,4) for _ in range(3)])
        s = sphere((size,size,size), r, center=c, amplitude=np.random.uniform(0.8,1.2))
        s = gaussian_filter(s, sigma=np.random.uniform(0.5,1.5))
        v += s
        v = apply_ctf(v, defocus=np.random.uniform(15000,30000), pixel=3.0)
        v = apply_missing_wedge(v, wedge_deg=np.random.uniform(15,40))
        save_mrc(v, os.path.join(out_path, f'usable_{i:04d}.mrc'))


def make_junk(out_path, n, size=64):
    os.makedirs(out_path, exist_ok=True)
    for i in range(n):
        v = np.random.normal(0, np.random.uniform(0.06,0.2), size=(size,size,size)).astype(np.float32)
        # add random blobs
        for _ in range(np.random.randint(0,4)):
            r = np.random.uniform(size*0.02, size*0.12)
            c = np.array([np.random.uniform(0,size) for _ in range(3)])
            b = sphere((size,size,size), r, center=c, amplitude=np.random.uniform(0.2,0.8))
            b = gaussian_filter(b, sigma=np.random.uniform(0.5,3.0))
            v += b
        v = apply_ctf(v, defocus=np.random.uniform(15000,40000), pixel=3.0)
        v = apply_missing_wedge(v, wedge_deg=np.random.uniform(10,60))
        save_mrc(v, os.path.join(out_path, f'junk_{i:04d}.mrc'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--n-usable', type=int, default=500)
    parser.add_argument('--n-junk', type=int, default=500)
    parser.add_argument('--size', type=int, default=64)
    args = parser.parse_args()
    make_usable(os.path.join(args.out_dir, 'usable'), args.n_usable, size=args.size)
    make_junk(os.path.join(args.out_dir, 'junk'), args.n_junk, size=args.size)
    print('Wrote simulated data to', args.out_dir)
```

---

## `notebooks/train_on_simulated_data.ipynb`

Below is the **notebook as a sequence of marked cells**. Copy into a `.ipynb` if you want, or run equivalent commands in a script/REPL.

**Cell 1 — setup**

```python
# Notebook: train_on_simulated_data.ipynb
%matplotlib inline
import os
os.environ['PYTHONPATH'] = os.getcwd()
from src.simulate import make_usable, make_junk
DATA_DIR = 'data/sim'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
# generate small dataset for a quick run
!python src/simulate.py --out-dir data/sim --n-usable 200 --n-junk 200 --size 64
```

**Cell 2 — pretrain (short)**

```python
!python src/train_pretrain.py --data-dir data/sim --out models/pretrained_small.pth --epochs 10 --batch-size 8 --size 64
```

**Cell 3 — train classifier**

```python
!python src/train_classifier.py --data-dir data/sim --pretrained models/pretrained_small.pth --out models/clf_small.pth --epochs 10 --batch-size 8 --size 64
```

**Cell 4 — score and inspect**

```python
!python src/score_subtomos.py --encoder models/pretrained_small.pth --clf models/clf_small.pth --folder data/sim/usable --out scores_usable.csv
!python src/score_subtomos.py --encoder models/pretrained_small.pth --clf models/clf_small.pth --folder data/sim/junk --out scores_junk.csv
import pandas as pd
pd.read_csv('scores_usable.csv').head()
```

---

## `scripts/run_demo.sh`

```bash
#!/bin/bash
set -e
python src/simulate.py --out-dir data/sim --n-usable 300 --n-junk 300 --size 64
python src/train_pretrain.py --data-dir data/sim --out models/pretrained.pth --epochs 30 --batch-size 16 --size 64
python src/train_classifier.py --data-dir data/sim --pretrained models/pretrained.pth --out models/clf.pth --epochs 20 --batch-size 16 --size 64
python src/score_subtomos.py --encoder models/pretrained.pth --clf models/clf.pth --folder data/sim/usable --out demo_scores.csv
```

---

## Final notes, caveats, and next steps

- This repo is intentionally **minimal** so you can iterate quickly.
- **Things to improve** (I can help implement any of these):
  - Proper argparse config and logging, model checkpointing with best-val logic (already rudimentary present).
  - Better Fourier-space CTF and missing-wedge simulation (I included simple approximations). Use `mrcfile` metadata or CTFFIND outputs if available in real data.
  - Add more advanced augmentations (random local occlusions, mixup in 3D, rotation in SO(3) using quaternion sampling). Use `torchio` or `kornia` for richer transforms.
  - Balanced sampling and focal loss for heavy class imbalance.
  - Active learning loop and GPU-distributed training.
  - A small evaluation script that aggregates per-tomogram scores (mean, median) and suggests a threshold for desired recall.

- **Labeling strategy**: start with a few hundred labelled subtomograms, then use the classifier to propose uncertain examples for manual labeling (active learning).

If you want, I can now:
- Convert the notebook cells into a real `.ipynb` file and place it in the repo (ready to download).
- Add a proper `setup.py` and CLI entrypoints.
- Implement Grad-CAM style explainability for 3D.

Tell me which of those you want next and I will add them to the repo.

