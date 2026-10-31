#!/usr/bin/env python
# coding: utf-8
"""
evaluate.py -- Inference-only evaluation for SGSCF-Net.

Loads a pretrained checkpoint and evaluates it on a dataset WITHOUT retraining.
The core training script (SGSCF_Net.py) is NOT imported or modified in any way;
the model definition below is an EXACT copy of the one in SGSCF_Net.py, so the
saved .pt weights load cleanly.

The architecture hyper-parameters (patch size, number of classes, channel width
K) are auto-detected from the checkpoint itself, so you only need to pass the
dataset flag and the .pt file.

Examples
--------
  python evaluate.py -d IN --ckpt checkpoints/SGSCFNet_IndianPines_OA98.72.pt
  python evaluate.py -d UP --ckpt checkpoints/SGSCFNet_PaviaUniversity_OA99.86.pt --data_dir ./data
  python evaluate.py -d SV --ckpt checkpoints/SGSCFNet_Salinas_OA100.00.pt --save_map

Note on the reported number
---------------------------
This script predicts EVERY labeled pixel in the scene and reports whole-scene
Overall Accuracy / Average Accuracy / Kappa. Because the exact random train/test
split used during the original training run is not stored, this whole-scene OA
includes the (~10%) training pixels and is therefore a close upper estimate of
the held-out test OA reported in the paper, not a bit-exact reproduction.
"""

import argparse, math, os
import numpy as np
import scipy.io as sio
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.data as Data
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report

# =========================================================================== #
# MODEL  --  EXACT COPY of the modules in SGSCF_Net.py
# (Only change: SGSCFNet takes `img_rows` as an argument instead of reading a
#  module-level global. Layer names are identical, so state_dict keys match.)
# =========================================================================== #

class SBAM(nn.Module):
    """Spectral Band Attention Module."""
    def __init__(self, C, r=8):
        super().__init__()
        mid = max(C//r, 8)
        self.pool = nn.AdaptiveAvgPool3d((1,1,C))
        self.fc   = nn.Sequential(nn.Linear(C,mid), nn.ReLU(inplace=True), nn.Linear(mid,C), nn.Sigmoid())
    def forward(self, x):
        z = self.pool(x).view(x.size(0), -1)
        w = self.fc(z).view(x.size(0),1,1,1,-1)
        return x * w


class CBSA(nn.Module):
    """Complementary Branch Spectral Attention."""
    def __init__(self, K, ks=3):
        super().__init__()
        self.gate = nn.Conv1d(K, K, ks, padding=ks//2, groups=K, bias=False)
        self.bn   = nn.BatchNorm3d(K)
    def forward(self, a, b):
        U = (a + b).mean(dim=(2,3))
        alpha = torch.sigmoid(self.gate(U))
        alpha = alpha[:,  :, None, None, :]
        return self.bn(a*alpha + b*(1-alpha))


class ECA3D(nn.Module):
    """Efficient Channel Attention 3D."""
    def __init__(self, K, ks=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1,1,ks, padding=(ks-1)//2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        y = self.pool(x).squeeze(-1).squeeze(-1).transpose(-1,-2)
        y = self.conv(y).transpose(-1,-2).unsqueeze(-1).unsqueeze(-1)
        return x * self.sig(y)


class ResBlock3D(nn.Module):
    """3D Residual Block + ECA."""
    def __init__(self, Ci, Co, ks, pad, start=False, end=False):
        super().__init__()
        self.start, self.end = start, end
        self.c1 = nn.Sequential(nn.Conv3d(Ci,Co,ks,padding=pad), nn.ReLU(inplace=True))
        self.c2 = nn.Conv3d(Co,Co,ks,padding=pad)
        if not start: self.bn0 = nn.BatchNorm3d(Ci)
        self.bn1 = nn.BatchNorm3d(Co)
        self.bn2 = nn.BatchNorm3d(Co)
        self.eca = ECA3D(Co)
    def forward(self, X):
        skip = X
        if self.start:
            out = self.c1(X)
        else:
            out = F.relu(self.bn0(X), inplace=True)
            out = self.c1(out)
        out = F.relu(self.bn1(out), inplace=True)
        out = self.c2(out)
        if self.start: out = self.bn2(out)
        out = self.eca(out) + skip
        if self.end: out = F.relu(self.bn2(out), inplace=True)
        return out


class SCFG(nn.Module):
    """Second-Order Covariance Fusion Gate."""
    def __init__(self, C, spatial_size):
        super().__init__()
        N = spatial_size * spatial_size
        self.gate = nn.Linear(N, 1)
        self.N = N
    def forward(self, x):
        B,C,H,W = x.shape
        f_ch = x.mean(dim=(2,3))
        f_sp = x.mean(dim=1).view(B,-1)
        G = torch.bmm(f_ch.unsqueeze(2), f_sp.unsqueeze(1))
        G = G / G.norm(dim=(1,2), keepdim=True).clamp(1e-8)
        gate = torch.sigmoid(self.gate(G).squeeze(-1))
        return x * gate.view(B,C,1,1)


class SGSCFNet(nn.Module):
    """SGSCF-Net (exact copy; img_rows passed in rather than read as a global)."""
    def __init__(self, band, classes, img_rows, K=16):
        super().__init__()
        self.sbam = SBAM(band)
        self.e1   = nn.Conv3d(1, K, (1,1,7), stride=(1,1,2))
        self.e3   = nn.Conv3d(1, K, (3,3,7), stride=(1,1,2), padding=(1,1,0))
        self.bn1  = nn.Sequential(nn.BatchNorm3d(K), nn.ReLU(inplace=True))
        self.bn3  = nn.Sequential(nn.BatchNorm3d(K), nn.ReLU(inplace=True))
        self.cbsa = CBSA(K)
        self.rs1  = ResBlock3D(K, K, (1,1,7), (0,0,3), start=True)
        self.rs2  = ResBlock3D(K, K, (1,1,7), (0,0,3))
        self.spool   = nn.AdaptiveAvgPool3d((None, None, 1))
        self.expand  = nn.Sequential(nn.Conv3d(K,128,(1,1,1)), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
        self.scfg = SCFG(128, img_rows)
        self.se   = nn.Sequential(nn.Conv3d(1,K,(3,3,128),padding=(1,1,0)), nn.BatchNorm3d(K), nn.ReLU(inplace=True))
        self.rs3  = ResBlock3D(K, K, (3,3,1), (1,1,0))
        self.rs4  = ResBlock3D(K, K, (3,3,1), (1,1,0), end=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc   = nn.Linear(K, classes)

    def forward(self, x):
        x   = self.sbam(x)
        a   = self.bn1(self.e1(x))
        b   = self.bn3(self.e3(x))
        v   = self.cbsa(a, b)
        v   = self.rs2(self.rs1(v))
        v   = self.expand(self.spool(v))
        v2d = self.scfg(v.squeeze(-1))
        v   = v2d.permute(0,2,3,1).unsqueeze(1)
        s   = self.se(v)
        s   = self.rs4(self.rs3(s))
        out = self.pool(s).view(s.size(0),-1)
        return self.fc(out)


# =========================================================================== #
# DATA  --  loading + patch extraction (mirrors SGSCF_Net.py)
# =========================================================================== #

CLASS_NAMES_ALL = {
    'IN': ['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture','Grass-trees',
           'Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill','Soybean-mintill',
           'Soybean-clean','Wheat','Woods','Buildings-Grass-Trees','Stone-Steel-Towers'],
    'UP': ['Asphalt','Meadows','Gravel','Trees','Painted-metal-sheets','Bare-Soil',
           'Bitumen','Self-Blocking-Bricks','Shadows'],
    'SV': ['Brocoli-green-weeds-1','Brocoli-green-weeds-2','Fallow','Fallow-rough-plow',
           'Fallow-smooth','Stubble','Celery','Grapes-untrained','Soil-vinyard-develop',
           'Corn-senesced-green-weeds','Lettuce-romaine-4wk','Lettuce-romaine-5wk',
           'Lettuce-romaine-6wk','Lettuce-romaine-7wk','Vinyard-untrained','Vinyard-vertical-trellis'],
}

COLOR_MAP = np.array([
    [0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],
    [192,192,192],[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],
    [0,128,128],[0,0,128],[255,165,0],[255,192,203]
])


def resolve_data_root(data_dir):
    if data_dir:
        root = data_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rel = os.path.join(script_dir, '..', 'data')
        root = rel if os.path.isdir(rel) else 'C:/Users/LENOVO/Pictures/MinorProject/data/'
    if root and not root.endswith(('/', '\\')):
        root += '/'
    return root


def load_dataset(name, root):
    if name == 'IN':
        data = sio.loadmat(root+'Indian_pines_corrected.mat')['indian_pines_corrected']
        gt   = sio.loadmat(root+'Indian_pines_gt.mat')['indian_pines_gt']
    elif name == 'UP':
        data = sio.loadmat(root+'PaviaU.mat')['paviaU']
        gt   = sio.loadmat(root+'PaviaU_gt.mat')['paviaU_gt']
    elif name == 'SV':
        data = sio.loadmat(root+'Salinas_corrected.mat')['salinas_corrected']
        gt   = sio.loadmat(root+'Salinas_gt.mat')['salinas_gt']
    else:
        raise ValueError(name)
    H,W,C = data.shape
    flat = preprocessing.scale(data.reshape(-1,C).astype(np.float32))
    return flat.reshape(H,W,C), gt


# Same seed list used by SGSCF_Net.py (run N -> seeds[N-1]).
SEEDS = [1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341]


def sampling(proportion, ground_truth):
    """EXACT copy of SGSCF_Net.py's split: (1-proportion) fraction per class is TRAIN."""
    tr, te = [], []
    for cls in range(1, int(ground_truth.max())+1):
        idx = np.where(ground_truth.ravel()==cls)[0].tolist()
        np.random.shuffle(idx)
        nb_train = max(int((1-proportion)*len(idx)), 3)
        tr += idx[:nb_train]
        te += idx[nb_train:]
    np.random.shuffle(tr); np.random.shuffle(te)
    return tr, te


def select_patch(mat, r, c, half):
    return mat[r-half:r+half+1, c-half:c+half+1]


def extract_patches(indices, cols, padded_data, half, dim):
    out = np.zeros((len(indices), 2*half+1, 2*half+1, dim), dtype=np.float32)
    for i,v in enumerate(indices):
        r = v // cols + half
        c = v %  cols + half
        out[i] = select_patch(padded_data, r, c, half)
    return out


def infer_arch(state_dict):
    """Read img_rows, classes, K straight from the checkpoint tensors."""
    scfg_n = next(v.shape[1] for k,v in state_dict.items() if k.endswith('scfg.gate.weight'))
    img_rows = int(round(math.sqrt(scfg_n)))
    classes, K = state_dict['fc.weight'].shape
    return img_rows, int(classes), int(K)


# =========================================================================== #
# MAIN
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description='Inference-only evaluation for SGSCF-Net.')
    ap.add_argument('-d','--dataset', required=True, choices=['IN','UP','SV'])
    ap.add_argument('--ckpt',         required=True, help='path to a .pt checkpoint')
    ap.add_argument('--data_dir',     default=None,  help='folder with the .mat files')
    ap.add_argument('--batch_size',   type=int, default=128)
    ap.add_argument('--split',        choices=['test','full'], default='test',
                    help="'test' = reproduce the exact held-out test split (paper number); "
                         "'full' = score every labeled pixel (upper estimate)")
    ap.add_argument('--run',          type=int, default=1,
                    help='run index used at training time -> seed (run 1 -> 1331). '
                         'Grid checkpoints used run 1; the bundled IN checkpoint used run 3.')
    ap.add_argument('--vs',           type=float, default=0.9,
                    help='valid_split used at training time (0.9 -> 10%% train / 90%% pool)')
    ap.add_argument('--save_map',     action='store_true', help='save prediction & GT maps as PNG')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')

    # --- checkpoint & architecture ---
    state_dict = torch.load(args.ckpt, map_location='cpu')
    img_rows, classes, K = infer_arch(state_dict)
    half = (img_rows - 1) // 2
    print(f'[INFO] Checkpoint: {os.path.basename(args.ckpt)}')
    print(f'[INFO] Auto-detected  patch={img_rows}x{img_rows}  classes={classes}  K={K}')

    # --- data ---
    root = resolve_data_root(args.data_dir)
    print(f'[INFO] Data root: {root}')
    data_hsi, gt_hsi = load_dataset(args.dataset, root)
    H, W, BAND = data_hsi.shape
    gt_flat = gt_hsi.reshape(-1)
    n_classes_data = int(gt_flat.max())
    print(f'[INFO] {args.dataset}: {H}x{W}x{BAND}  labeled-classes={n_classes_data}')
    if n_classes_data != classes:
        print(f'[WARN] checkpoint has {classes} classes but dataset has {n_classes_data} '
              f'-- make sure the checkpoint matches the dataset (-d {args.dataset}).')

    padded = np.pad(data_hsi, ((half,half),(half,half),(0,0)), mode='constant', constant_values=0)

    # --- model ---
    net = SGSCFNet(BAND, classes, img_rows, K).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    npar = sum(p.numel() for p in net.parameters())
    print(f'[INFO] Model loaded OK  ({npar:,} parameters)')

    # --- choose which pixels to score ---
    if args.split == 'test':
        # Reproduce the EXACT held-out test split from training.
        seed = SEEDS[args.run - 1]
        np.random.seed(seed)
        tr_idx, te_idx = sampling(args.vs, gt_flat)
        n_val = len(tr_idx)                 # SGSCF_Net.py uses |train| pixels for validation
        eval_idx = np.array(te_idx[n_val:]) # the rest is the held-out TEST set
        print(f'[INFO] Reproducing held-out test split  (run={args.run}, seed={seed}, vs={args.vs})')
        print(f'[INFO] train={len(tr_idx)}  val={n_val}  test={len(eval_idx)}')
        scope = 'HELD-OUT TEST'
    else:
        eval_idx = np.where(gt_flat > 0)[0]
        print(f'[INFO] Scoring every labeled pixel ({len(eval_idx)} px)')
        scope = 'WHOLE-SCENE'

    X = extract_patches(eval_idx.tolist(), W, padded, half, BAND)
    ds = Data.TensorDataset(torch.from_numpy(X).unsqueeze(1))
    ld = Data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for (xb,) in ld:
            preds.extend(net(xb.to(device)).argmax(1).cpu().numpy().tolist())
    preds = np.array(preds)
    y_true = (gt_flat[eval_idx] - 1).astype(np.int64)

    # --- metrics (mirrors SGSCF_Net.py exactly, incl. its confusion-matrix arg order) ---
    oa  = metrics.accuracy_score(preds, y_true)
    cm  = metrics.confusion_matrix(preds, y_true)
    rs  = cm.sum(1)
    per_cls = np.nan_to_num(np.diag(cm)/rs)
    aa  = per_cls.mean()
    kap = metrics.cohen_kappa_score(preds, y_true)

    names = CLASS_NAMES_ALL[args.dataset][:classes]
    print(f"\n{'='*58}\n  {scope} EVALUATION ({args.dataset})\n{'='*58}")
    print(f"  Overall Accuracy (OA): {oa*100:.4f}%")
    print(f"  Average Accuracy (AA): {aa*100:.4f}%")
    print(f"  Kappa:                 {kap*100:.4f}%")
    print(f"\n  Per-class accuracy:")
    for i, nm in enumerate(names):
        print(f"    {nm:<30s}: {per_cls[i]*100:6.2f}%")
    print(f"\n{classification_report(y_true, preds, digits=4)}")

    # --- optional maps ---
    if args.save_map:
        import matplotlib.pyplot as plt
        os.makedirs('eval_maps', exist_ok=True)
        pred_map = np.zeros(gt_hsi.shape, dtype=np.uint8)
        for idx, p_val in zip(eval_idx, preds):
            pred_map[idx // W, idx % W] = p_val + 1
        plt.imsave(f'eval_maps/{args.dataset}_ground_truth.png', COLOR_MAP[gt_hsi].astype(np.uint8))
        plt.imsave(f'eval_maps/{args.dataset}_prediction.png',   COLOR_MAP[pred_map].astype(np.uint8))
        print(f"[INFO] Maps saved to ./eval_maps/")


if __name__ == '__main__':
    main()
