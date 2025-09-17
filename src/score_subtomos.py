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