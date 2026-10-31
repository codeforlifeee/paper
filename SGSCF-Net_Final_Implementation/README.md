# SGSCF-Net: Spectral-Guided Spatial Cross-Attention Fusion Network

A lightweight 3D-CNN for **hyperspectral image (HSI) classification**, evaluated on
Indian Pines, Pavia University, and Salinas. SGSCF-Net introduces three novel
attention/fusion modules on top of a proven dual-branch 3D residual backbone, and
is trained with a joint Spread–Focal loss designed for extreme class imbalance.

The entire model is lightweight (≈50K trainable parameters; the exact count is
printed at startup), and trains in minutes on a single laptop GPU (RTX 3050).

## Novel Contributions

| Module | What it does | Replaces / adds over baseline |
| --- | --- | --- |
| **SBAM** — Spectral Band Attention Module | Learns per-band weights on the **raw** input spectrum, before any convolution, to suppress noisy bands early. | Input-level band gating (standard SE acts on feature maps, not raw bands). |
| **CBSA** — Complementary Branch Spectral Attention | Fuses the dual 3D entry branches with complementary gates (α and 1−α), forcing the branches to specialize. | Replaces a single shared SE gate shared across both branches. |
| **SCFG** — Second-Order Covariance Fusion Gate | Uses a spatial–channel outer product (covariance) to model second-order co-activation, then gates channels. | Replaces first-order average-pool → FC gating. |
| **Combined Loss** | `0.7 × SpreadLoss + 0.3 × FocalLoss` with inverse-frequency class weights. | Targets extreme minority classes (e.g. *Oats*, *Grass-pasture-mowed*). |

## Repository Structure

```
SGSCF-Net_Final_Implementation/
├── SGSCF_Net.py           # complete model, training & evaluation (single-file, self-contained)
├── evaluate.py            # load a checkpoint and score it — no retraining
├── requirements.txt
├── data/                  # place dataset .mat files here (see data/README.md)
├── checkpoints/           # best pretrained weights, one per dataset
│   ├── SGSCFNet_IndianPines_OA98.72.pt
│   ├── SGSCFNet_PaviaUniversity_OA99.86.pt
│   └── SGSCFNet_Salinas_OA100.00.pt
└── results/               # grid-search reports, metrics & figures per dataset
    ├── Indian_Pines/
    ├── Pavia_University/
    └── Salinas_Valley/
```

## Setup

```bash
conda create -n hsi-pytorch python=3.9 -y
conda activate hsi-pytorch
pip install -r requirements.txt   # or torch==1.13.1+cu117 for CUDA 11.7
```

Then download the datasets into `data/` as described in [data/README.md](data/README.md).

## Usage

From inside the repo folder:

```bash
cd "C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/SGSCF-Net_Final_Implementation"
```

**Quick smoke test first** (1 run, 50 epochs, ~10 min on RTX 3050):

```bash
python SGSCF_Net.py -d IN -e 50 -i 1
```

**Full experiment** (3 runs, 200 epochs) — one command per dataset:

```bash
# Indian Pines
python SGSCF_Net.py -d IN -e 200 -i 3 -p 4 -k 16 -vs 0.9 -lr 5e-4

# Pavia University
python SGSCF_Net.py -d UP -e 200 -i 3 -p 4 -k 16 -vs 0.9 -lr 5e-4

# Salinas
python SGSCF_Net.py -d SV -e 200 -i 3 -p 4 -k 16 -vs 0.9 -lr 5e-4
```

### Where it finds the data

The script auto-looks for `../data`, and if that's missing it falls back to the
hardcoded path `C:/Users/LENOVO/Pictures/MinorProject/data/`. On the original
author's machine the `.mat` files already live there, so the commands above work
as-is.

To use the repo's own `data/` folder (after copying the `.mat` files into it) or
any other location, point to it explicitly:

```bash
python SGSCF_Net.py -d UP --data_dir ./data
```

### Evaluate a pretrained checkpoint (no retraining)

`evaluate.py` loads one of the bundled checkpoints and scores it directly — no
training. It does **not** import or alter `SGSCF_Net.py`; it carries an exact
copy of the model so the saved weights load identically, and it **auto-detects**
the patch size, class count, and width `K` from the checkpoint (you don't need to
remember that IN=15×15, UP=13×13, SV=17×17).

It has two modes via `--split`:

- **`--split test` (default)** — reproduces the **exact held-out test split** used
  during training (by replaying the same seed + stratified sampling), so it
  returns the real paper number. It needs the training `--run` index (grid
  checkpoints used run 1; the bundled Indian Pines checkpoint used run 3) and the
  `--vs` value (0.9 for all bundled checkpoints).
- **`--split full`** — scores every labeled pixel (whole-scene). Fast sanity check;
  reads as a slight upper estimate because it also includes training pixels.

Exact commands for the three bundled checkpoints:

```bash
# Indian Pines  -> reproduces OA ≈ 98.72%
python evaluate.py -d IN --ckpt checkpoints/SGSCFNet_IndianPines_OA98.72.pt --run 3

# Pavia University -> reproduces OA ≈ 99.86%
python evaluate.py -d UP --ckpt checkpoints/SGSCFNet_PaviaUniversity_OA99.86.pt --run 1

# Salinas -> reproduces OA ≈ 100.00%
python evaluate.py -d SV --ckpt checkpoints/SGSCFNet_Salinas_OA100.00.pt --run 1

# Whole-scene instead, and save prediction maps:
python evaluate.py -d SV --ckpt checkpoints/SGSCFNet_Salinas_OA100.00.pt --split full --save_map
```

Other flags: `--data_dir <path>` (same auto-detect/fallback as training),
`--batch_size <n>`, `--save_map` (writes GT + prediction PNGs to `eval_maps/`).

> **Reproducibility note:** OA matches the reported value to within ~1 pixel
> (≈0.01%). The residual difference is floating-point / cuDNN non-determinism,
> because the weights were originally trained on different GPU hardware (Kaggle).
> The train/test *split* itself is reproduced exactly.

### Key arguments

| Flag | Meaning | Default |
| --- | --- | --- |
| `-d`  | Dataset: `IN` / `UP` / `SV` | `IN` |
| `-e`  | Epochs | `200` |
| `-i`  | Number of runs (for mean ± std) | `3` |
| `-p`  | Half-patch size (window = 2p+1) | `4` → 9×9 |
| `-k`  | Base channel width K | `16` |
| `-vs` | Validation split (0.9 → 10% used for training) | `0.9` |
| `-lr` | Learning rate | `5e-4` |
| `--data_dir` | Explicit path to dataset folder | auto-detect `../data` |

## Results (best overall accuracy)

| Dataset | OA | Pretrained weights |
| --- | --- | --- |
| Indian Pines | **98.72%** | `checkpoints/SGSCFNet_IndianPines_OA98.72.pt` |
| Pavia University | **99.86%** | `checkpoints/SGSCFNet_PaviaUniversity_OA99.86.pt` |
| Salinas | **100.00%** | `checkpoints/SGSCFNet_Salinas_OA100.00.pt` |

Full patch/batch grid-search sweeps, per-class accuracies, prediction maps, and
loss curves for each dataset are in [`results/`](results/).

## Class Labels & Visualization Color Map

**Class names are dataset-aware.** The script selects the correct label set based
on the `-d` flag, so per-class accuracy printouts and reports always match the
chosen dataset:

- `-d IN` → 16 Indian Pines classes (Alfalfa, Corn-notill, …)
- `-d UP` → 9 Pavia University classes (Asphalt, Meadows, Gravel, …)
- `-d SV` → 16 Salinas classes (Brocoli-green-weeds-1, Fallow, Celery, …)

**One shared color palette** is used for all ground-truth and prediction maps.
Colors are assigned by **class index** (not by class name), so the same palette
serves every dataset — Indian Pines and Salinas use indices 1–16, Pavia
University uses indices 1–9. Index `0` is always the unlabeled background.

| Index | Color | RGB | | Index | Color | RGB |
| :---: | --- | --- | --- | :---: | --- | --- |
| 0 | Black (background) | `0,0,0`       | | 9  | Maroon     | `128,0,0`     |
| 1 | Red               | `255,0,0`     | | 10 | Olive      | `128,128,0`   |
| 2 | Green             | `0,255,0`     | | 11 | Dark Green | `0,128,0`     |
| 3 | Blue              | `0,0,255`     | | 12 | Purple     | `128,0,128`   |
| 4 | Yellow            | `255,255,0`   | | 13 | Teal       | `0,128,128`   |
| 5 | Magenta           | `255,0,255`   | | 14 | Navy       | `0,0,128`     |
| 6 | Cyan              | `0,255,255`   | | 15 | Orange     | `255,165,0`   |
| 7 | Silver            | `192,192,192` | | 16 | Pink       | `255,192,203` |
| 8 | Gray              | `128,128,128` | |    |            |               |

## Notes

- `SGSCF_Net.py` is self-contained: it loads data, builds the model, trains over
  multiple seeded runs, evaluates (OA / AA / Kappa / per-class), and writes
  checkpoints, classification maps, and a text report at runtime.
- Running the script creates `models/`, `report/`, `plots/`, and
  `classification_maps/` folders in the working directory for its outputs.
