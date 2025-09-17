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