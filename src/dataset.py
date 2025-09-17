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