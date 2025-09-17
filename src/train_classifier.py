import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from src.dataset import SubtomogramDataset
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