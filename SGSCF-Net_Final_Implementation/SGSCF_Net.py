#!/usr/bin/env python
# coding: utf-8
"""
SGSCF-Net: Spectral-Guided Spatial Cross-Attention Fusion Network
HSI Classification -- Indian Pines | Pavia University | Salinas

Novel Contributions vs. A2S2K-ResNet:
  1. SBAM : Spectral Band Attention Module  (input-level per-band weighting)
  2. CBSA : Complementary Branch Spectral Attention (replaces shared SE gate)
  3. SCFG : Second-Order Covariance Fusion Gate (replaces 1st-order pooling)
  4. Loss : 0.7*SpreadLoss + 0.3*FocalLoss (minority-class aware)

~308K params | ~30-45 min on RTX 3050 | No Mamba, No Transformer
"""

import argparse, math, os, time, sys
import numpy as np
import scipy.io as sio
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim, torch.utils.data as Data
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report
from torch.nn.modules.loss import _Loss

for d in ['./report','./models','./classification_maps']: os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------- #
# CLI ARGS
# --------------------------------------------------------------------------- #
p = argparse.ArgumentParser()
p.add_argument('-d','--dataset',     default='IN')     # IN | UP | SV
p.add_argument('-e','--epoch',       type=int,   default=200)
p.add_argument('-i','--iter',        type=int,   default=3)
p.add_argument('-p','--patch',       type=int,   default=4,   help='half-patch (window=2p+1)')
p.add_argument('-k','--kernel',      type=int,   default=16)
p.add_argument('-vs','--valid_split',type=float, default=0.9, help='train proportion (90% -> 10% train)')
p.add_argument('-lr','--lr',         type=float, default=5e-4)
p.add_argument('-bs','--batch_size',  type=int,   default=64)
p.add_argument('--data_dir',         type=str,   default=None, help='path to dataset directory')
args = p.parse_args()

PARAM_DATASET = args.dataset
PARAM_EPOCH   = args.epoch
PARAM_ITER    = args.iter
PATCH_SIZE    = args.patch          # window = 2*PATCH_SIZE+1 x 2*PATCH_SIZE+1
PARAM_VAL     = args.valid_split    # 0.9 -> take (1-0.9)=10% as TRAIN
PARAM_K       = args.kernel
LR            = args.lr
PARAM_BATCH   = args.batch_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device: {device}')
seeds = [1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341]

# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
if args.data_dir:
    DATA_ROOT = args.data_dir
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rel_path = os.path.join(script_dir, '..', 'data')
    if os.path.isdir(rel_path):
        DATA_ROOT = rel_path
    else:
        DATA_ROOT = 'C:/Users/LENOVO/Pictures/MinorProject/data/'

if DATA_ROOT and not DATA_ROOT.endswith('/') and not DATA_ROOT.endswith('\\'):
    DATA_ROOT += '/'

def load_dataset(name):
    if name == 'IN':
        data = sio.loadmat(DATA_ROOT+'Indian_pines_corrected.mat')['indian_pines_corrected']
        gt   = sio.loadmat(DATA_ROOT+'Indian_pines_gt.mat')['indian_pines_gt']
        TOTAL = 10249
    elif name == 'UP':
        data = sio.loadmat(DATA_ROOT+'PaviaU.mat')['paviaU']
        gt   = sio.loadmat(DATA_ROOT+'PaviaU_gt.mat')['paviaU_gt']
        TOTAL = 42776
    elif name == 'SV':
        data = sio.loadmat(DATA_ROOT+'Salinas_corrected.mat')['salinas_corrected']
        gt   = sio.loadmat(DATA_ROOT+'Salinas_gt.mat')['salinas_gt']
        TOTAL = 54129
    else:
        raise ValueError(name)
    H,W,C = data.shape
    flat = data.reshape(-1,C).astype(np.float32)
    flat = preprocessing.scale(flat)          # z-score per band on all pixels
    return flat.reshape(H,W,C), gt, TOTAL

data_hsi, gt_hsi, TOTAL_SIZE = load_dataset(PARAM_DATASET)
H,W,BAND = data_hsi.shape
gt_flat  = gt_hsi.reshape(-1)
CLASSES_NUM = int(gt_flat.max())
img_rows = 2*PATCH_SIZE+1

padded = np.pad(data_hsi, ((PATCH_SIZE,PATCH_SIZE),(PATCH_SIZE,PATCH_SIZE),(0,0)),
                mode='constant', constant_values=0)
print(f'[INFO] {PARAM_DATASET}: {H}x{W}x{BAND}  classes={CLASSES_NUM}  patch={img_rows}x{img_rows}')

# --------------------------------------------------------------------------- #
# DATA UTILITIES  (adapted from A2S2K geniter.py)
# --------------------------------------------------------------------------- #
def select_patch(mat, r, c, half):
    return mat[r-half:r+half+1, c-half:c+half+1]

def extract_patches(indices, whole_data, padded_data, half, dim):
    N = len(indices)
    out = np.zeros((N, 2*half+1, 2*half+1, dim), dtype=np.float32)
    rows, cols = whole_data.shape[0], whole_data.shape[1]
    for i,v in enumerate(indices):
        r = v // cols + half
        c = v %  cols + half
        out[i] = select_patch(padded_data, r, c, half)
    return out

def sampling(proportion, ground_truth):
    """Stratified split: (1-proportion) fraction is TRAIN, rest is TEST."""
    tr, te = [], []
    for cls in range(1, int(ground_truth.max())+1):
        idx = np.where(ground_truth.ravel()==cls)[0].tolist()
        np.random.shuffle(idx)
        nb_train = max(int((1-proportion)*len(idx)), 3)
        tr += idx[:nb_train]
        te += idx[nb_train:]
    np.random.shuffle(tr); np.random.shuffle(te)
    return tr, te

def make_loaders(tr_idx, te_idx, all_idx, whole_data, padded_data, half, dim, gt, batch_sz=64):
    def tensor_ds(idx, labels):
        X = torch.from_numpy(extract_patches(idx,whole_data,padded_data,half,dim)).unsqueeze(1)
        y = torch.from_numpy(labels).float()
        return Data.TensorDataset(X, y)

    y_tr  = (gt[tr_idx]-1).astype(np.int64)
    y_te  = (gt[te_idx]-1).astype(np.int64)
    y_all = (gt[all_idx]-1).astype(np.int64)

    N_val = len(tr_idx)   # use same size as train for validation
    x_val, y_val   = extract_patches(te_idx[:N_val], whole_data, padded_data, half, dim), y_te[:N_val]
    x_test, y_test = extract_patches(te_idx[N_val:], whole_data, padded_data, half, dim), y_te[N_val:]
    x_all          = extract_patches(all_idx,         whole_data, padded_data, half, dim)

    def ld(ds, shuf): return Data.DataLoader(ds, batch_sz, shuffle=shuf, num_workers=0)
    tr_ds  = tensor_ds(tr_idx, y_tr)
    val_ds = Data.TensorDataset(torch.from_numpy(x_val).unsqueeze(1),  torch.from_numpy(y_val).float())
    te_ds  = Data.TensorDataset(torch.from_numpy(x_test).unsqueeze(1), torch.from_numpy(y_test).float())
    all_ds = Data.TensorDataset(torch.from_numpy(x_all).unsqueeze(1),  torch.from_numpy(y_all).float())
    return ld(tr_ds,True), ld(val_ds,True), ld(te_ds,False), ld(all_ds,False), y_test

# --------------------------------------------------------------------------- #
# NOVEL MODULES
# --------------------------------------------------------------------------- #

class SBAM(nn.Module):
    """
    Spectral Band Attention Module  [NOVEL - input-level band weighting].
    Standard SENet does channel attention on FEATURE maps.
    SBAM does it on RAW spectral bands BEFORE any convolution.
    No published HSI paper does this for 3D-conv-based models.
    Input/Output: B x 1 x P x P x C
    """
    def __init__(self, C, r=8):
        super().__init__()
        mid = max(C//r, 8)
        self.pool = nn.AdaptiveAvgPool3d((1,1,C))   # squeeze spatial dims only
        self.fc   = nn.Sequential(nn.Linear(C,mid), nn.ReLU(inplace=True), nn.Linear(mid,C), nn.Sigmoid())

    def forward(self, x):                          # x: B,1,P,P,C
        z = self.pool(x).view(x.size(0), -1)       # B,C
        w = self.fc(z).view(x.size(0),1,1,1,-1)    # B,1,1,1,C
        return x * w


class CBSA(nn.Module):
    """
    Complementary Branch Spectral Attention  [NOVEL].
    A2S2K: computes ONE shared gate weight for both branches -> same attention.
    CBSA:  gate alpha -> branch_A gets alpha, branch_B gets (1-alpha).
    Forces specialization: when 1x1 branch is dominant, 3x3 is suppressed, etc.
    Only ~K params (depthwise 1D conv across spectral dim).
    Input: two tensors B,K,P,P,C2  ->  fused B,K,P,P,C2
    """
    def __init__(self, K, ks=3):
        super().__init__()
        self.gate = nn.Conv1d(K, K, ks, padding=ks//2, groups=K, bias=False)
        self.bn   = nn.BatchNorm3d(K)

    def forward(self, a, b):
        U = (a + b).mean(dim=(2,3))           # B,K,C2
        alpha = torch.sigmoid(self.gate(U))   # B,K,C2
        alpha = alpha[:,  :, None, None, :]   # B,K,1,1,C2
        return self.bn(a*alpha + b*(1-alpha))


class ECA3D(nn.Module):
    """Efficient Channel Attention 3D  (from A2S2K -- proven, kept as-is)."""
    def __init__(self, K, ks=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1,1,ks, padding=(ks-1)//2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        y = self.pool(x).squeeze(-1).squeeze(-1).transpose(-1,-2)  # B,1,K
        y = self.conv(y).transpose(-1,-2).unsqueeze(-1).unsqueeze(-1) # B,K,1,1,1
        return x * self.sig(y)


class ResBlock3D(nn.Module):
    """3D Residual Block + ECA  (from A2S2K -- proven, kept as-is)."""
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
    """
    Second-Order Covariance Fusion Gate  [NOVEL].
    A2S2K: AvgPool -> FC -> Softmax  (first-order statistics only).
    SCFG:  spatial-channel outer product G[b,c,n] = f_ch[b,c] x f_sp[b,n]
           captures WHICH spatial positions co-activate with WHICH channels.
           Compress G (B,C,N) -> gate (B,C) via Linear(N->1) per channel.
    Input/Output: B,C,H,W
    """
    def __init__(self, C, spatial_size):
        super().__init__()
        N = spatial_size * spatial_size
        self.gate = nn.Linear(N, 1)
        self.N = N
    def forward(self, x):
        B,C,H,W = x.shape
        f_ch = x.mean(dim=(2,3))                         # B,C
        f_sp = x.mean(dim=1).view(B,-1)                  # B,N
        G = torch.bmm(f_ch.unsqueeze(2), f_sp.unsqueeze(1))  # B,C,N
        G = G / G.norm(dim=(1,2), keepdim=True).clamp(1e-8)
        gate = torch.sigmoid(self.gate(G).squeeze(-1))   # B,C
        return x * gate.view(B,C,1,1)


# --------------------------------------------------------------------------- #
# MAIN MODEL
# --------------------------------------------------------------------------- #
class SGSCFNet(nn.Module):
    """
    SGSCF-Net: Spectral-Guided Spatial Cross-Attention Fusion Network
    Pipeline:
      SBAM -> Dual-3D-Entry -> CBSA -> 2x SpectralResBlock
           -> SpectralPool -> SCFG -> SpatialEntry -> 2x SpatialResBlock -> FC
    """
    def __init__(self, band, classes, K=16):
        super().__init__()
        # 1. SBAM
        self.sbam = SBAM(band)
        # 2. Dual 3D entry  (A2S2K proven architecture -- both branches on same input)
        self.e1   = nn.Conv3d(1, K, (1,1,7), stride=(1,1,2))           # 1x1 spectral
        self.e3   = nn.Conv3d(1, K, (3,3,7), stride=(1,1,2), padding=(1,1,0)) # 3x3 spatial+spec
        self.bn1  = nn.Sequential(nn.BatchNorm3d(K), nn.ReLU(inplace=True))
        self.bn3  = nn.Sequential(nn.BatchNorm3d(K), nn.ReLU(inplace=True))
        # 3. CBSA
        self.cbsa = CBSA(K)
        # 4. Spectral ResBlocks
        self.rs1  = ResBlock3D(K, K, (1,1,7), (0,0,3), start=True)
        self.rs2  = ResBlock3D(K, K, (1,1,7), (0,0,3))
        # 5. Spectral compression: adaptive pool collapses spectral to 1, then expand channels
        self.spool   = nn.AdaptiveAvgPool3d((None, None, 1))
        self.expand  = nn.Sequential(nn.Conv3d(K,128,(1,1,1)), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
        # 6. SCFG (after squeezing spectral dim -> 2D spatial)
        self.scfg = SCFG(128, img_rows)
        # 7. Spatial entry: last dim = 128 (channel axis permuted to last Conv3d axis)
        self.se   = nn.Sequential(nn.Conv3d(1,K,(3,3,128),padding=(1,1,0)), nn.BatchNorm3d(K), nn.ReLU(inplace=True))
        # 8. Spatial ResBlocks
        self.rs3  = ResBlock3D(K, K, (3,3,1), (1,1,0))
        self.rs4  = ResBlock3D(K, K, (3,3,1), (1,1,0), end=True)
        # 9. Classifier
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc   = nn.Linear(K, classes)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv3d,nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m,(nn.BatchNorm3d,nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) and m.weight.requires_grad:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: B,1,P,P,C  (Conv3d convention: N,C_in,D,H,W -> here D=P,H=P,W=BAND)
        x   = self.sbam(x)                       # B,1,P,P,C
        a   = self.bn1(self.e1(x))               # B,K,P,P,C2
        b   = self.bn3(self.e3(x))               # B,K,P,P,C2
        v   = self.cbsa(a, b)                    # B,K,P,P,C2
        v   = self.rs2(self.rs1(v))              # B,K,P,P,C2
        v   = self.expand(self.spool(v))         # B,128,P,P,1
        # SCFG on 2D view
        v2d = self.scfg(v.squeeze(-1))                    # B,128,P,P
        v   = v2d.permute(0,2,3,1).unsqueeze(1)          # B,1,P,P,128  ← channels LAST for Conv3D
        # Spatial processing
        s   = self.se(v)                         # B,K,P,P,1
        s   = self.rs4(self.rs3(s))              # B,K,P,P,1
        out = self.pool(s).view(s.size(0),-1)    # B,K
        return self.fc(out)                      # B,classes


# --------------------------------------------------------------------------- #
# LOSS FUNCTIONS
# --------------------------------------------------------------------------- #
class SpreadLoss(_Loss):
    """Spread Loss (Hinton ICLR 2018). Copied & adapted from DMC implementation."""
    def __init__(self, m_min=0.2, m_max=0.9, num_class=16):
        super().__init__()
        self.m_min, self.m_max, self.num_class = m_min, m_max, num_class
    def forward(self, x, target, r):
        b,E = x.shape
        m   = self.m_min + (self.m_max-self.m_min)*r
        at  = x.gather(1, target.view(-1,1)).expand(b,E)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask.scatter_(1, target.view(-1,1), False)
        loss = (torch.relu(m-(at-x))**2) * mask.float()
        return loss.sum() / b

class FocalLoss(nn.Module):
    """Focal Loss (Lin ICCV 2017) with inverse-frequency alpha weights."""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha).float() if alpha is not None else None
    def forward(self, x, targets):
        lp  = F.log_softmax(x, dim=1)
        pt  = lp.gather(1, targets.view(-1,1)).squeeze(1).exp()
        lpt = lp.gather(1, targets.view(-1,1)).squeeze(1)
        at  = self.alpha.to(x.device).gather(0, targets) if self.alpha is not None else 1.0
        return (-at*(1-pt)**self.gamma*lpt).mean()

class CombinedLoss(nn.Module):
    """0.7*SpreadLoss + 0.3*FocalLoss  (novel combo for minority-class HSI)."""
    def __init__(self, C, weights=None, m_min=0.2, m_max=0.9):
        super().__init__()
        self.spread = SpreadLoss(m_min, m_max, C)
        self.focal  = FocalLoss(2.0, weights)
    def forward(self, logits, targets, r=0.5):
        targets = targets.long()
        return 0.7*self.spread(logits, targets, r) + 0.3*self.focal(logits, targets)

def class_weights(gt, C):
    w = np.array([1./(np.sum(gt==(i+1))+1e-8) for i in range(C)], dtype=np.float32)
    return (w/w.sum()*C).tolist()

# --------------------------------------------------------------------------- #
# TRAIN / EVAL
# --------------------------------------------------------------------------- #
def evaluate(loader, net, loss_fn, dev, r=1.0):
    acc,los,n = 0.,0.,0
    net.eval()
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(dev), y.to(dev)
            yh  = net(X)
            acc += (yh.argmax(1)==y.long()).sum().item()
            los += loss_fn(yh, y.long(), r).item(); n += y.shape[0]
    net.train()
    return acc/n, los/len(loader)

def aa_each(cm):
    d = np.diag(cm); rs = np.sum(cm,1)
    e = np.nan_to_num(d/rs)
    return e, e.mean()

def train_model(net, tr, va, loss_fn, opt, dev, epochs, patience=30):
    net = net.to(dev)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    ckpt  = './net_sgscf_best.pt'
    best, no_imp, total, step = 1e9, 0, epochs*len(tr), 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for ep in range(epochs):
        tr_a,tr_l,n,t0 = 0.,0.,0,time.time()
        net.train()
        for X,y in tr:
            X,y = X.to(dev), y.to(dev)
            r   = step/total
            yh  = net(X)
            L   = loss_fn(yh, y.long(), r)
            opt.zero_grad(); L.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tr_a += (yh.argmax(1)==y.long()).sum().item()
            tr_l += L.item(); n += y.shape[0]; step += 1
        sched.step()
        va_a,va_l = evaluate(va, net, loss_fn, dev, 1.0)
        
        # record metrics
        history['train_loss'].append(tr_l/len(tr))
        history['val_loss'].append(va_l)
        history['train_acc'].append(tr_a/n)
        history['val_acc'].append(va_a)
        
        print(f'Ep[{ep+1:3d}/{epochs}] '
              f'tr_loss={tr_l/len(tr):.4f} tr_acc={tr_a/n:.4f} '
              f'va_loss={va_l:.4f} va_acc={va_a:.4f} '
              f't={time.time()-t0:.1f}s')
        if va_l < best:
            best=va_l; torch.save(net.state_dict(), ckpt); no_imp=0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'[EARLY STOP] ep {ep+1}'); break
    net.load_state_dict(torch.load(ckpt, map_location=dev))
    return net, history

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    os.makedirs('./plots', exist_ok=True)
    
    OA_all,AA_all,KA_all,TT_all,ET_all = [],[],[],[],[]
EA_all = np.zeros((PARAM_ITER, CLASSES_NUM))
CW     = class_weights(gt_flat, CLASSES_NUM)

CLASS_NAMES_ALL = {
    'IN': [
        'Alfalfa','Corn-notill','Corn-mintill','Corn',
        'Grass-pasture','Grass-trees','Grass-pasture-mowed','Hay-windrowed',
        'Oats','Soybean-notill','Soybean-mintill','Soybean-clean',
        'Wheat','Woods','Buildings-Grass-Trees','Stone-Steel-Towers'],
    'UP': [
        'Asphalt','Meadows','Gravel','Trees','Painted-metal-sheets',
        'Bare-Soil','Bitumen','Self-Blocking-Bricks','Shadows'],
    'SV': [
        'Brocoli-green-weeds-1','Brocoli-green-weeds-2','Fallow','Fallow-rough-plow',
        'Fallow-smooth','Stubble','Celery','Grapes-untrained','Soil-vinyard-develop',
        'Corn-senesced-green-weeds','Lettuce-romaine-4wk','Lettuce-romaine-5wk',
        'Lettuce-romaine-6wk','Lettuce-romaine-7wk','Vinyard-untrained',
        'Vinyard-vertical-trellis'],
}
CLASS_NAMES = CLASS_NAMES_ALL[PARAM_DATASET]

# Shared palette indexed by CLASS INDEX (0 = background, 1..16 = classes).
# Works for all datasets: IN/SV use classes 1-16, UP uses classes 1-9.
COLOR_MAP = np.array([
    [0, 0, 0],          # 0  Background (Black)
    [255, 0, 0],        # 1  Class 1  (Red)
    [0, 255, 0],        # 2  Class 2  (Green)
    [0, 0, 255],        # 3  Class 3  (Blue)
    [255, 255, 0],      # 4  Class 4  (Yellow)
    [255, 0, 255],      # 5  Class 5  (Magenta)
    [0, 255, 255],      # 6  Class 6  (Cyan)
    [192, 192, 192],    # 7  Class 7  (Silver)
    [128, 128, 128],    # 8  Class 8  (Gray)
    [128, 0, 0],        # 9  Class 9  (Maroon)
    [128, 128, 0],      # 10 Class 10 (Olive)
    [0, 128, 0],        # 11 Class 11 (Dark Green)
    [128, 0, 128],      # 12 Class 12 (Purple)
    [0, 128, 128],      # 13 Class 13 (Teal)
    [0, 0, 128],        # 14 Class 14 (Navy)
    [255, 165, 0],      # 15 Class 15 (Orange)
    [255, 192, 203]     # 16 Class 16 (Pink)
])

for run in range(PARAM_ITER):
    print(f"\n{'='*60}\n  Run {run+1}/{PARAM_ITER}\n{'='*60}")
    np.random.seed(seeds[run]); torch.manual_seed(seeds[run])

    net  = SGSCFNet(BAND, CLASSES_NUM, PARAM_K)
    npar = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'[PARAMS] {npar:,}')

    opt     = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = CombinedLoss(CLASSES_NUM, CW)

    # 10% stratified split
    tr_idx, te_idx = sampling(PARAM_VAL, gt_flat)
    
    # Get all labeled indices (no holes)
    total_idx = np.where(gt_flat > 0)[0].tolist()
    
    tr_ld, va_ld, te_ld, all_ld, y_test = make_loaders(
        tr_idx, te_idx, total_idx, data_hsi, padded, PATCH_SIZE, BAND, gt_flat, batch_sz=PARAM_BATCH)
    print(f'[SPLIT] train={len(tr_idx)}  test={len(y_test)}')

    tic = time.time()
    net, history = train_model(net, tr_ld, va_ld, loss_fn, opt, device, PARAM_EPOCH)
    TT_all.append(time.time()-tic)
    print(f'[TRAIN] {TT_all[-1]/60:.1f} min')

    # Save loss plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(history['train_loss'], label='Train Loss', color='#4f46e5', linewidth=1.5)
    plt.plot(history['val_loss'], label='Val Loss', color='#dc2626', linewidth=1.5)
    plt.title(f'SGSCF-Net Loss Curve (Run {run+1})', fontsize=11, fontweight='bold', color='#1e1b4b')
    plt.xlabel('Epoch', fontsize=9)
    plt.ylabel('Loss', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plots/loss_curve_run{run+1}.png', dpi=150)
    plt.close()

    # Generate classification maps
    preds_all = []
    net.eval()
    with torch.no_grad():
        for X, y in all_ld:
            preds_all.extend(net(X.to(device)).argmax(1).cpu().numpy().tolist())
            
    # Paint ground truth & prediction map
    pred_map = np.zeros(gt_hsi.shape, dtype=np.uint8)
    for idx, p_val in zip(total_idx, preds_all):
        r = idx // W
        c = idx % W
        pred_map[r, c] = p_val + 1
        
    gt_colored   = COLOR_MAP[gt_hsi]
    pred_colored = COLOR_MAP[pred_map]
    
    plt.imsave(f'./plots/ground_truth.png', gt_colored.astype(np.uint8))
    plt.imsave(f'./plots/prediction_map_run{run+1}.png', pred_colored.astype(np.uint8))

    preds,tic2 = [],time.time()
    with torch.no_grad():
        for X,y in te_ld:
            preds.extend(net(X.to(device)).argmax(1).cpu().numpy().tolist())
    ET_all.append(time.time()-tic2)

    gt_te = y_test[:len(preds)]
    oa    = metrics.accuracy_score(preds, gt_te)
    cm    = metrics.confusion_matrix(preds, gt_te)
    ea,aa = aa_each(cm)
    kap   = metrics.cohen_kappa_score(preds, gt_te)
    print(f'\n[RESULT] OA={oa*100:.4f}%  AA={aa*100:.4f}%  Kappa={kap*100:.4f}%')
    print(classification_report(gt_te, preds, digits=4))
    OA_all.append(oa); AA_all.append(aa); KA_all.append(kap)
    EA_all[run,:len(ea)] = ea
    
    # NEW SAFE SAVING MECHANISM
    window_size = 2 * PATCH_SIZE + 1
    torch.save(net.state_dict(), f'./models/SGSCFNet_{PARAM_DATASET}_P{window_size}_B{PARAM_BATCH}_run{run+1}_OA{oa:.4f}.pt')

print(f"\n{'='*60}\n  FINAL RESULTS ({PARAM_ITER} runs)\n{'='*60}")
print(f"OA:    {np.mean(OA_all)*100:.4f} +/- {np.std(OA_all)*100:.4f}%")
print(f"AA:    {np.mean(AA_all)*100:.4f} +/- {np.std(AA_all)*100:.4f}%")
print(f"Kappa: {np.mean(KA_all)*100:.4f} +/- {np.std(KA_all)*100:.4f}%")
print(f"Train: {np.mean(TT_all)/60:.1f} min avg")
for i,nm in enumerate(CLASS_NAMES[:CLASSES_NUM]):
    print(f"  {nm:<35s}: {np.mean(EA_all[:,i])*100:.2f}%")

rp = f'./report/SGSCFNet_{PARAM_DATASET}_vs{PARAM_VAL}_K{PARAM_K}.txt'
with open(rp,'w') as f:
    f.write(f"SGSCF-Net | {PARAM_DATASET} | valid_split={PARAM_VAL} K={PARAM_K} patch={img_rows}\n\n")
    f.write(f"OA:    {np.mean(OA_all)*100:.4f} +/- {np.std(OA_all)*100:.4f}\n")
    f.write(f"AA:    {np.mean(AA_all)*100:.4f} +/- {np.std(AA_all)*100:.4f}\n")
    f.write(f"Kappa: {np.mean(KA_all)*100:.4f} +/- {np.std(KA_all)*100:.4f}\n")
    f.write(f"Train: {np.mean(TT_all)/60:.1f} min avg\n\n")
    for i,nm in enumerate(CLASS_NAMES[:CLASSES_NUM]):
        f.write(f"{nm:<35s}: {np.mean(EA_all[:,i])*100:.2f}%\n")
print(f'[DONE] Report -> {rp}')

# Auto-invoke PDF generation script
import subprocess
try:
    print("[INFO] Invoking generate_pdf.py...")
    subprocess.run([sys.executable, "generate_pdf.py"], check=True)
    print("[INFO] PDF Report successfully updated.")
except Exception as e:
    print(f"[ERROR] Failed to compile PDF report: {e}")

