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