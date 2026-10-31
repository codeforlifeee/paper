# -*- coding: utf-8 -*-
"""
std_viz.py  --  Standardized classification-map coloring + loss curves
               for ALL 8 HSI baseline models (Indian Pines benchmark).

This module is PURELY a visualization / output helper.  It does NOT touch any
model's architecture, training loop, hyper-parameters, or reported metrics.
Every baseline calls it so that the Ground-Truth map, the Prediction map and
the training Loss curve share one identical colour scheme, making the
comparative study visually consistent.

Canonical palette = the COLOR_MAP used by our own SGSCF-Net
(ActualImplementation/SGSCF-Net_Final_Implementation/SGSCF_Net.py):
    index 0  -> background (black)
    index 1..16 -> the 16 Indian-Pines classes (fixed RGB)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Canonical shared palette (identical to SGSCF-Net COLOR_MAP) ------------
COLOR_MAP = np.array([
    [0,   0,   0],      # 0  Background (Black)
    [255, 0,   0],      # 1  Red
    [0,   255, 0],      # 2  Green
    [0,   0,   255],    # 3  Blue
    [255, 255, 0],      # 4  Yellow
    [255, 0,   255],    # 5  Magenta
    [0,   255, 255],    # 6  Cyan
    [192, 192, 192],    # 7  Silver
    [128, 128, 128],    # 8  Gray
    [128, 0,   0],      # 9  Maroon
    [128, 128, 0],      # 10 Olive
    [0,   128, 0],      # 11 Dark Green
    [128, 0,   128],    # 12 Purple
    [0,   128, 128],    # 13 Teal
    [0,   0,   128],    # 14 Navy
    [255, 165, 0],      # 15 Orange
    [255, 192, 203],    # 16 Pink
], dtype=np.uint8)

OUT_DIRS = [".", "/kaggle/working/outputs"]


def _dirs():
    ds = []
    for d in OUT_DIRS:
        try:
            os.makedirs(d, exist_ok=True)
            ds.append(d)
        except Exception:
            pass
    return ds


def colorize(int_map):
    """int label map (H,W), bg=0, classes 1..16 -> RGB uint8 image."""
    a = np.clip(np.asarray(int_map).astype(int), 0, 16)
    return COLOR_MAP[a]


def save_maps(model, gt_int, pred_int):
    """Save standardized ground_truth.png + pred_<model>.png (identical palette).

    gt_int / pred_int : (H,W) integer arrays, background=0, classes 1..16.
    Also dumps the raw integer maps as .npy for reproducibility.
    """
    gt_int = np.asarray(gt_int).astype(int)
    pred_int = np.asarray(pred_int).astype(int)
    try:
        np.save("/kaggle/working/%s_gt.npy" % model, gt_int)
        np.save("/kaggle/working/%s_pred.npy" % model, pred_int)
    except Exception:
        pass
    for d in _dirs():
        try:
            plt.imsave(os.path.join(d, "ground_truth.png"), colorize(gt_int))
            plt.imsave(os.path.join(d, "pred_%s.png" % model), colorize(pred_int))
        except Exception as e:
            print("[std_viz] map save skipped for", d, ":", e)
    print("[std_viz] saved standardized GT + prediction maps for", model)


def save_loss(model, train_losses, val_losses=None):
    """Save standardized loss_<model>.png from in-memory loss lists."""
    try:
        train_losses = [float(x) for x in train_losses]
    except Exception:
        return
    _vl = None
    try:
        if val_losses is not None:
            _vl = [float(x) for x in val_losses]
    except Exception:
        _vl = None
    try:
        np.save("/kaggle/working/%s_loss.npy" % model, np.asarray(train_losses, dtype=float))
        if _vl is not None and len(_vl) == len(train_losses) and len(_vl) > 0:
            np.save("/kaggle/working/%s_vloss.npy" % model, np.asarray(_vl, dtype=float))
    except Exception:
        pass
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses,
                 label="Train Loss", color="#2563eb", lw=2)
        if val_losses is not None and len(val_losses) == len(train_losses) and len(val_losses) > 0:
            plt.plot(range(1, len(val_losses) + 1), val_losses,
                     label="Val Loss", color="#dc2626", lw=2)
        plt.title("%s - Loss Curve (Indian Pines)" % model, fontsize=13, fontweight="bold")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(frameon=True); plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        for d in _dirs():
            try:
                plt.savefig(os.path.join(d, "loss_%s.png" % model), bbox_inches="tight", dpi=200)
            except Exception as e:
                print("[std_viz] loss save skipped for", d, ":", e)
        plt.close()
        print("[std_viz] saved loss curve for", model)
    except Exception as e:
        print("[std_viz] save_loss error:", e)


def log_loss_csv(model, epoch, train_loss, val_loss=None):
    """Append-only, scope-safe per-epoch loss logging (call inside train loop)."""
    try:
        p = "/kaggle/working/%s_loss.csv" % model
        new = not os.path.exists(p)
        with open(p, "a") as f:
            if new:
                f.write("epoch,train_loss,val_loss\n")
            f.write("%s,%s,%s\n" % (epoch, float(train_loss),
                                    "" if val_loss is None else float(val_loss)))
    except Exception:
        pass


def save_loss_from_csv(model):
    """Plot loss_<model>.png from the CSV produced by log_loss_csv()."""
    p = "/kaggle/working/%s_loss.csv" % model
    if not os.path.exists(p):
        print("[std_viz] no loss csv for", model)
        return
    import csv
    tr, va = [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                tr.append(float(row["train_loss"]))
            except Exception:
                pass
            try:
                if row.get("val_loss", "") not in ("", None):
                    va.append(float(row["val_loss"]))
            except Exception:
                pass
    save_loss(model, tr, va if len(va) == len(tr) and len(va) > 0 else None)
