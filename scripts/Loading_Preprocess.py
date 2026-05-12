import os
import glob
import math
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 2) I/O + preprocessing helpers
# =========================
def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine

def pad_to_min_size(x, ps):
    D,H,W = x.shape
    pd = max(0, ps - D)
    ph = max(0, ps - H)
    pw = max(0, ps - W)
    if pd==0 and ph==0 and pw==0:
        return x
    return np.pad(x, ((0,pd),(0,ph),(0,pw)), mode='constant', constant_values=0).astype(np.float32)

def brain_mask_robust_raw(x3d):
    """Simple brain-ish mask from raw T1; used only for normalization and patch sampling."""
    x3d = x3d.astype(np.float32)
    bm = np.zeros_like(x3d, dtype=np.uint8)
    for z in range(x3d.shape[0]):
        sl = x3d[z]
        v = sl[np.isfinite(sl)]
        if v.size < 100:
            continue
        p10 = np.percentile(v, 10)
        p99 = np.percentile(v, 99)
        thr = p10 + 0.15 * (p99 - p10)
        bm[z] = (sl > thr).astype(np.uint8)

    t = torch.from_numpy(bm[None,None].astype(np.float32))
    # close/open-ish cleanup (no scipy)
    t = F.max_pool3d(t, 3, 1, 1)               # dilate
    t = -F.max_pool3d(-t, 3, 1, 1)             # erode
    t = -F.max_pool3d(-t, 3, 1, 1)             # erode
    t = F.max_pool3d(t, 3, 1, 1)               # dilate
    bm2 = (t[0,0].numpy() > 0.5).astype(np.uint8)
    return bm2

def zscore_within_mask(x, mask, eps=1e-6, clip=5.0):
    v = x[mask > 0]
    if v.size == 0:
        mu, sd = 0.0, 1.0
    else:
        mu = float(v.mean())
        sd = float(v.std()) + eps
    z = (x - mu) / sd
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32)

def dice_coeff(pred_bin, gt_bin, eps=1e-6):
    # pred_bin, gt_bin: numpy arrays {0,1}
    inter = float((pred_bin * gt_bin).sum())
    return (2*inter + eps) / (float(pred_bin.sum() + gt_bin.sum()) + eps)
# =========================
# 3) Discover ATLAS pairs (T1, lesion mask)
# =========================
t1_list = sorted(glob.glob(os.path.join(ATLAS_ROOT, '**', '*_T1w.nii.gz'), recursive=True))
pairs = []
for t1 in t1_list:
    base = t1.replace('_T1w.nii.gz', '')
    mask = base + '_label-L_desc-T1lesion_mask.nii.gz'
    if os.path.exists(mask):
        pairs.append((t1, mask))

print("T1 files:", len(t1_list))
print("Paired (T1,mask):", len(pairs))
assert len(pairs) > 0, "No paired files found. Check ATLAS_ROOT and filename patterns."
print("Example:\n", pairs[0][0], "\n", pairs[0][1])


# =========================
# 4) Split train/val/test
# =========================
random.shuffle(pairs)
n = len(pairs)
n_test = max(1, int(0.1 * n))
n_val  = max(1, int(0.1 * n))

test_pairs = pairs[:n_test]
val_pairs  = pairs[n_test:n_test+n_val]
train_pairs= pairs[n_test+n_val:]

print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}  Test: {len(test_pairs)}")
