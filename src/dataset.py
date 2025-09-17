import numpy as np
import mrcfile
import torch
from torch.utils.data import Dataset

def load_mrc(path):
    with mrcfile.open(path, permissive=True) as m:
        data = m.data.astype(np.float32)
    return data

def center_crop_or_pad(volume, out_shape):
    z0 = max(0, (volume.shape[0] - out_shape[0])//2)
    y0 = max(0, (volume.shape[1] - out_shape[1])//2)
    x0 = max(0, (volume.shape[2] - out_shape[2])//2)
    # If smaller, pad
    result = np.zeros(out_shape, dtype=volume.dtype)
    # compute src ranges
    sz = min(out_shape[0], volume.shape[0])
    sy = min(out_shape[1], volume.shape[1])
    sx = min(out_shape[2], volume.shape[2])
    src = volume[z0:z0+sz, y0:y0+sy, x0:x0+sx]
    dz = (out_shape[0]-sz)//2
    dy = (out_shape[1]-sy)//2
    dx = (out_shape[2]-sx)//2
    result[dz:dz+sz, dy:dy+sy, dx:dx+sx] = src
    return result

class SubtomogramDataset(Dataset):
    """
    items: list of dicts: {'path':..., 'label':0/1, 'tomogram':...}
    """
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
        # Normalize — zero mean, unit var per volume
        vol = vol - vol.mean()
        std = vol.std()
        if std > 0:
            vol = vol / std
        vol = vol.astype(np.float32)
        if self.transform:
            vol = self.transform(vol)
        # return channel-first tensor
        return torch.from_numpy(vol[None, ...]), torch.tensor(info.get('label', -1), dtype=torch.long)