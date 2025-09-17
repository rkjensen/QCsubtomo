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