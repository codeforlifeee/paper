# Baseline Models Master Benchmark Report
**Benchmark Dataset:** Indian Pines (Corrected, 200 spectral bands, 16 ground-truth land-cover classes)  
**Standardized Experimental Setup:**
- **Spatial Patch Size:** 11 × 11
- **Batch Size:** 32
- **Data Splitting Ratio:** 10% Training / 10% Validation / 80% Testing
- **Training Duration:** 200 Epochs
- **Hardware Acceleration:** Kaggle NVIDIA Tesla GPU

---

## 0. Standardized Visualization & Rerun Status (2026-10-18)

All classification maps (Ground Truth **and** Prediction) are now rendered with **one shared colour palette** — the exact `COLOR_MAP` used by our proposed **SGSCF-Net** (`ActualImplementation/SGSCF-Net_Final_Implementation/SGSCF_Net.py`) — so the comparative study is visually consistent across every model. Each model now reports **three** standardized images: **Ground Truth**, **Prediction Map**, and **Training Loss Curve**. Colour/visualization changes are additive only — no model architecture, training, or metric was altered (SSAN's data split and SSRN's Keras-version compatibility are the only exceptions, both required just to make those models run correctly on Indian Pines; see notes).

| Model | Rerun (standardized) | Notes |
|:---|:---:|:---|
| ViT | ✅ Done | **OA 92.70 / AA 88.80 / Kappa 91.65** — fresh run |
| HSI-SSFTT | ✅ Done | **OA 98.57 / AA 96.15 / Kappa 98.37** — fresh run |
| CRV-MHDL | ⚠️ Metrics OK | **OA 98.58 / AA 93.41 / Kappa 98.38** — reproduced. NOTE: prediction-map image is still the author's colour scheme (shared-palette standardisation of CRV's map is unresolved / being handled manually) |
| SSRN | ✅ Done | **OA 98.93 / AA 98.95 / Kappa 98.78** — first genuine result (old 98.52 never ran). Fixed Keras-3 (`tf_keras`) + 11×11 patch; no dead classes |
| SSAN | ✅ Done | **OA 64.19 / AA 72.77 / Kappa 59.79** — genuine IP run; split + checkpoint (`best=0.0`) fixed, AA no longer nan |
| A2S2K-ResNet | 🔄 Not done | No fresh result yet (run hung on Kaggle). Table shows **prior-run** values (98.36 / 96.22 / 98.14) as placeholders, marked ‡ |
| HyperRoPE-HiT | ✅ Done | **OA 94.42 / AA 92.12 / Kappa 93.64**; full-image map newly added (shared palette) |
| DMC | ✅ Done | **OA 99.15 / AA 98.89 / Kappa 99.03** — reproduced, warm-started from the author's shipped checkpoint (so no genuine from-scratch training-loss curve) |

> Numbers below for 🔄/⏳ models are the **prior** values pending their standardized rerun; ✅ rows show the refreshed results.

### Data correctness assessment — is each number genuine?

| Model | OA / AA / Kappa | Correct? | Why / caveat |
|:---|:---:|:---:|:---|
| **ViT** | 92.70 / 88.80 / 91.65 | ✅ Genuine | Fresh run, confirmed from kernel log. Maps standardized. |
| **HSI-SSFTT** | 98.57 / 96.15 / 98.37 | ✅ Genuine | Fresh run, from `classification_report.txt`. Maps standardized. |
| **SSRN** | 98.93 / 98.95 / 98.78 | ✅ Genuine & now trustworthy | The old 98.52 was **fake** — SSRN had never actually run (Keras-3 crash). After the `tf_keras` + 11×11-patch fixes it truly ran; no dead classes. Maps standardized. |
| **SSAN** | 64.19 / 72.77 / 59.79 | ✅ Genuine | The old 64.62 / AA=nan was **invalid** (stale Pavia checkpoint + broken split). Real IP run now (checkpoint saved, valid split). SSAN genuinely scores low on IP's small classes — honest result. Maps standardized. |
| **HiT** | 94.42 / 92.12 / 93.64 | ✅ Genuine | Fresh full 200-epoch run. From `results_HiT.txt` eval dict. Maps standardized. |
| **DMC** | 99.15 / 98.89 / 99.03 | ⚠️ Reproduced, NOT from-scratch | Author's **pre-trained checkpoint** evaluated (0.39 min, warm-start). Metrics are real but it did not train fresh → no genuine training-loss curve for DMC. |
| **CRV** | 98.58 / 93.41 / 98.38 | ⚠️ Metrics correct; image not standardized | Numbers reproduced correctly (from `classification_Results.txt`). But the prediction-map **image** is still the author's colour scheme — shared-palette standardization of CRV's map is unresolved (handled manually). |
| **A2S2K** | 98.36 / 96.22 / 98.14 (‡) | ❌ Not fresh / placeholder | The rerun **hung** on Kaggle, so these are **prior-run** values kept as placeholders (marked ‡, status 🔄). No genuine fresh result yet. |

---

## 1. Comprehensive Benchmark Comparison Table

| # | Model Architecture | Paradigm / Category | Overall Accuracy (OA) | Average Accuracy (AA) | Kappa ($\kappa$) | Train Time | Weights Status & Checkpoint |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **—** | **SGSCF-Net (Proposed)** | **Spectral Gating + Cross-Attn** | **98.05%** | **94.82%** | **97.78%** | **8.8 min** | [`SGSCFNet_IN_run1_OA0.9805.pt`](file:///C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/SGSCF-Net/sv_test1/models/SGSCFNet_IN_run1_OA0.9805.pt) |
| 1 | **ViT** ✅ | Vision Transformer | 92.70% | 88.80% | 91.65% | 1.71 min | [`best_vit.pth`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/ViT/vision_transformer/best_vit.pth) (2.4 MB) |
| 2 | **DMC** ✅ | Deep Matrix Capsules | 99.15% | 98.89% | 99.03% | 0.39 min | [`classification_report_dmc.txt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/DMC/DeepMatrixCapsules/classification_report_dmc.txt) |
| 3 | **A2S2K-ResNet** 🔄 | 3D-2D Hybrid CNN ResNet | 98.36%‡ | 96.22%‡ | 98.14%‡ | 18.42 min | [`net_DBA.pt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/A2S2K/A2S2KResNet/net_DBA.pt) (1.5 MB) |
| 4 | **CRV-MHDL** ✅ | Complex-Valued Multiscale Hybrid | 98.58% | 93.41% | 98.38% | 3.23 min | [`model_1.h5`](file:///C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/CRV/models/model_1.h5) (58.2 MB) |
| 5 | **SSRN** ✅ | Spectral-Spatial Residual ConvNet | 98.93% | 98.95% | 98.78% | 5.37 min | Keras-3 compat + 11×11 patch fixed; no dead classes now |
| 6 | **HSI-SSFTT** ✅ | Feature Token Transformer | 98.57% | 96.15% | 98.37% | 0.84 min | [`SSFTTnet_params.pth`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSFTT/cls_SSFTT_IP/cls_params/SSFTTnet_params.pth) (625 KB) |
| 7 | **SSAN** ✅ | Spectral-Spatial Attention Net | 64.19% | 72.77% | 59.79% | 14.00 min | Genuine IP run (split+checkpoint fixed); AA no longer nan |
| 8 | **HyperRoPE-HiT** ✅ | Rotary Position Transformer | 94.42% | 92.12% | 93.64% | 37.74 min | [`IndianPine_best_model_hit.pth`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/HiT/hyperrope-sst/checkpoints/IndianPine_best_model_hit.pth) |

> **✅ refreshed** = standardized rerun complete (new numbers shown). **🔄/⏳ = rerun in progress/queued**; **‡** marks prior-run values kept as placeholders until the standardized rerun lands.
> **Note on the previous SSAN `64.62% / AA=nan`:** caused by the author's fixed 100-train/100-val/200-test-per-class split (designed for Pavia) leaving Indian Pines' small classes with 0 test samples. Now using a small-class-safe split.
> **Note on the previous SSRN `98.52%`:** not reproducible — its committed log shows the same `keras.backend.int_shape` crash, so it had never truly run here. Fixed via the `tf_keras` legacy API; the rerun gives SSRN's first genuine number in this environment.

---

## 2. Model-by-Model Artifacts (Images with Complete Paths, Weights & Reports)

### 0. Proposed Model: SGSCF-Net (Ours)
- **Model Category:** Spectral Gated Spatial-Cross Fusion Network
- **Benchmark Performance:** **OA: 98.05%** | **AA: 94.82%** | **Kappa: 97.78%** | **Training Time: 8.8 min**
- **Model Weight Checkpoint:**
  - File: [`SGSCFNet_IN_run1_OA0.9805.pt`](file:///C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/SGSCF-Net/sv_test1/models/SGSCFNet_IN_run1_OA0.9805.pt)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SGSCF-Net\sv_test1\models\SGSCFNet_IN_run1_OA0.9805.pt`
- **Classification Report & Execution Logs:**
  - Official Report: [`grid_report_IN_P11_B32.txt`](file:///C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/SGSCF-Net/Kaggle_Master_Outputs/Indian_Pines/Reports_TXT/grid_report_IN_P11_B32.txt)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SGSCF-Net\Kaggle_Master_Outputs\Indian_Pines\Reports_TXT\grid_report_IN_P11_B32.txt`
- **Visual Outputs (2 Images):**

| Image 1: Predicted Classification Map | Image 2: Training Loss Curve |
|:---:|:---:|
| ![SGSCF-Net Prediction Map](file:///C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/SGSCF-Net/Kaggle_Master_Outputs/Indian_Pines/Visualizations_PNG/pred_P11_B32.png) | ![SGSCF-Net Loss Curve](file:///C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/SGSCF-Net/Kaggle_Master_Outputs/Indian_Pines/Visualizations_PNG/loss_P11_B32.png) |
| **Complete Location:**<br>`C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SGSCF-Net\Kaggle_Master_Outputs\Indian_Pines\Visualizations_PNG\pred_P11_B32.png` | **Complete Location:**<br>`C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SGSCF-Net\Kaggle_Master_Outputs\Indian_Pines\Visualizations_PNG\loss_P11_B32.png` |

---

### 1. Vision Transformer (ViT)
- **Model Category:** Pure Spatial-Spectral Transformer Encoder
- **Benchmark Performance (refreshed):** **OA: 92.70%** | **AA: 88.80%** | **Kappa: 91.65%** | **Training Time: 1.71 min**
- **Model Weight Checkpoint:**
  - File: [`ViT/vision_transformer/best_vit.pth`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/ViT/vision_transformer/best_vit.pth) *(Size: 2.4 MB)*
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\ViT\vision_transformer\best_vit.pth`
- **Classification Report & Execution Logs:**
  - Metrics CSV: [`ViT/outputs/metrics_ViT.csv`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/ViT/outputs/metrics_ViT.csv)
  - Full Kaggle Log: [`ViT/actual-vit-indian-pines-benchmark.log`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/ViT/actual-vit-indian-pines-benchmark.log)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\ViT\outputs\metrics_ViT.csv`
- **Visual Outputs (2 Images):**

| Ground Truth (shared palette) | Predicted Classification Map | Training Loss Curve |
|:---:|:---:|:---:|
| ![ViT Ground Truth](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/ViT/outputs/ground_truth.png) | ![ViT Prediction Map](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/ViT/outputs/pred_ViT.png) | ![ViT Loss Curve](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/ViT/outputs/loss_ViT.png) |
| `...\ViT\outputs\ground_truth.png` | `...\ViT\outputs\pred_ViT.png` | `...\ViT\outputs\loss_ViT.png` |

---

### 2. Deep Matrix Capsules (DMC)
- **Model Category:** Matrix Capsule Network with Dynamic Routing for Hyperspectral Data
- **Benchmark Performance (refreshed):** **OA: 99.15%** | **AA: 98.89%** | **Kappa: 99.03%** | **Training Time: 0.39 min**
- **Model Weight Checkpoint:**
  - In-memory execution with full per-class evaluation logs saved.
- **Classification Report & Execution Logs:**
  - Official Classification Report: [`DMC/DeepMatrixCapsules/classification_report_dmc.txt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/DMC/DeepMatrixCapsules/classification_report_dmc.txt)
  - Metrics CSV: [`DMC/outputs/metrics_DMC.csv`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/DMC/outputs/metrics_DMC.csv)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\DMC\DeepMatrixCapsules\classification_report_dmc.txt`
- **Visual Outputs (2 Images):**  _(Training-loss curve N/A — the author's DMC ships a pre-trained checkpoint, so this run evaluates it rather than training from scratch; no fresh loss curve is produced.)_

| Ground Truth (shared palette) | Predicted Classification Map |
|:---:|:---:|
| ![DMC Ground Truth](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/DMC/outputs/ground_truth.png) | ![DMC Prediction Map](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/DMC/outputs/pred_DMC.png) |
| **Complete Location:**<br>`C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\DMC\outputs\pred_DMC.png` | **Complete Location:**<br>`C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\DMC\outputs\ground_truth.png` |

---

### 3. A2S2K-ResNet
- **Model Category:** Attention-driven 3D-2D ResNet with Spatial-Spectral Kernel Attention
- **Benchmark Performance:** **OA: 98.36%** | **AA: 96.22%** | **Kappa: 98.14%** | **Training Time: 18.42 min**
- **Model Weight Checkpoints (.pt):**
  - Primary Checkpoint: [`A2S2K/A2S2KResNet/net_DBA.pt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/A2S2K/A2S2KResNet/net_DBA.pt) *(Size: 1.52 MB)*
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\A2S2K\A2S2KResNet\net_DBA.pt`
  - Replicate Checkpoint: [`A2S2K/A2S2KResNet/models/S3KAIResNetpatch_7_IN_split_0.9_lr_0.001adam_kernel_240.985.pt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/A2S2K/A2S2KResNet/models/S3KAIResNetpatch_7_IN_split_0.9_lr_0.001adam_kernel_240.985.pt) *(Size: 1.53 MB)*
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\A2S2K\A2S2KResNet\models\S3KAIResNetpatch_7_IN_split_0.9_lr_0.001adam_kernel_240.985.pt`
- **Classification Report & Execution Logs:**
  - Classification Results Report: [`A2S2K/A2S2KResNet/report/classification_Results_a2s2k.txt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/A2S2K/A2S2KResNet/report/classification_Results_a2s2k.txt)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\A2S2K\A2S2KResNet\report\classification_Results_a2s2k.txt`
- **Visual Outputs (2 Images):**

| Image 1: Predicted Classification Map | Image 2: Ground Truth Reference |
|:---:|:---:|
| ![A2S2K Prediction Map](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/A2S2K/outputs/pred_A2S2K.png) | ![A2S2K Ground Truth](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/A2S2K/outputs/ground_truth.png) |
| **Complete Location:**<br>`C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\A2S2K\outputs\pred_A2S2K.png` | **Complete Location:**<br>`C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\A2S2K\outputs\ground_truth.png` |

---

### 4. CRV-MHDL
- **Model Category:** Complex-Valued Residual Multiscale Hybrid Deep Learning
- **Benchmark Performance (refreshed):** **OA: 98.58%** | **AA: 93.41%** | **Kappa: 98.38%** | **Training Time: 3.23 min**
- **Model Weight Checkpoint:**
  - Checkpoint File: [`CRV/models/model_1.h5`](file:///C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/CRV/models/model_1.h5) *(Size: 58.2 MB)*
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\CRV\models\model_1.h5`
- **Classification Report & Execution Logs:**
  - Official Classification Results: [`CRV/CRV/classification_Results.txt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/CRV/CRV/classification_Results.txt)
  - Per-Class Report CSV: [`CRV/CRV/IP_Model Classification Report.csv`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/CRV/CRV/IP_Model%20Classification%20Report.csv)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\CRV\CRV\classification_Results.txt`
- **Visual Outputs (2 Images):**

| Image 1: Predicted Classification Map | Image 2: Training Loss Curve |
|:---:|:---:|
| ![CRV Prediction Map](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/CRV/outputs/pred_CRV.png) | ![CRV Loss Curve](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/CRV/outputs/loss_CRV.png) |
| **Complete Location:**<br>`C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\CRV\outputs\pred_CRV.png` | **Complete Location:**<br>`C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\CRV\outputs\loss_CRV.png` |

---

### 5. Spectral-Spatial Residual Network (SSRN)
- **Model Category:** Dual Residual 3D Spectral & 3D Spatial Convolutional Network
- **Benchmark Performance (refreshed):** **OA: 98.93%** | **AA: 98.95%** | **Kappa: 98.78%** | **Training Time: 5.37 min**
- **Model Weight Checkpoint:**
  - Executed under official repository protocol with internal evaluation hooks.
- **Classification Report & Execution Logs:**
  - Official Evaluation Results: [`SSRN/SSRN/records/IN_evaluation_results.txt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSRN/SSRN/records/IN_evaluation_results.txt)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSRN\SSRN\records\IN_evaluation_results.txt`
- **Visual Outputs (2 Images):**

| Ground Truth (shared palette) | Predicted Classification Map | Training Loss Curve |
|:---:|:---:|:---:|
| ![SSRN Ground Truth](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSRN/outputs/ground_truth.png) | ![SSRN Prediction Map](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSRN/outputs/pred_SSRN.png) | ![SSRN Loss Curve](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSRN/outputs/loss_SSRN.png) |
| `...\SSRN\outputs\ground_truth.png` | `...\SSRN\outputs\pred_SSRN.png` | `...\SSRN\outputs\loss_SSRN.png` |

---

### 6. HSI-SSFTT (Feature Tokenization Transformer)
- **Model Category:** 3D-2D CNN Feature Extraction + Spectral Tokenization Transformer
- **Benchmark Performance (refreshed):** **OA: 98.57%** | **AA: 96.15%** | **Kappa: 98.37%** | **Training Time: 0.84 min**
- **Model Weight Checkpoint (.pth):**
  - Trained Model File: [`SSFTT/cls_SSFTT_IP/cls_params/SSFTTnet_params.pth`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSFTT/cls_SSFTT_IP/cls_params/SSFTTnet_params.pth) *(Size: 625 KB)*
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSFTT\cls_SSFTT_IP\cls_params\SSFTTnet_params.pth`
- **Classification Report & Execution Logs:**
  - Official Classification Report: [`SSFTT/cls_SSFTT_IP/cls_result/classification_report.txt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSFTT/cls_SSFTT_IP/cls_result/classification_report.txt)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSFTT\cls_SSFTT_IP\cls_result\classification_report.txt`
- **Visual Outputs (3 Images):**

| Ground Truth (shared palette) | Predicted Classification Map | Training Loss Curve |
|:---:|:---:|:---:|
| ![SSFTT Ground Truth](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSFTT/outputs/ground_truth.png) | ![SSFTT Prediction Map](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSFTT/outputs/pred_SSFTT.png) | ![SSFTT Loss Curve](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSFTT/outputs/loss_SSFTT.png) |
| `...\SSFTT\outputs\ground_truth.png` | `...\SSFTT\outputs\pred_SSFTT.png` | `...\SSFTT\outputs\loss_SSFTT.png` |

---

### 7. Spectral-Spatial Attention Network (SSAN)
- **Model Category:** Recurrent and Convolutional Spectral-Spatial Attention Network (TF1)
- **Benchmark Performance (refreshed):** **OA: 64.19%** | **AA: 72.77%** | **Kappa: 59.79%** | **Training Time: 14.00 min**
- **Model Weight Checkpoint:**
  - TensorFlow Checkpoint Data: [`SSAN/model/ssan_step_1000.ckpt.data-00000-of-00001`](file:///C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/SSAN/model/ssan_step_1000.ckpt.data-00000-of-00001) *(Size: 65.2 MB)*
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SSAN\model\ssan_step_1000.ckpt.data-00000-of-00001`
  - Checkpoint Index & Meta: [`SSAN/model/ssan_step_1000.ckpt.meta`](file:///C:/Users/LENOVO/Pictures/MinorProject/ActualImplementation/SSAN/model/ssan_step_1000.ckpt.meta) *(Size: 523 KB)*
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SSAN\model\ssan_step_1000.ckpt.meta`
- **Classification Report & Execution Logs:**
  - Metrics CSV: [`SSAN/outputs/metrics_SSAN.csv`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSAN/outputs/metrics_SSAN.csv)
  - Full Kaggle Execution Log: [`SSAN/actual-ssan-indian-pines-benchmark.log`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSAN/actual-ssan-indian-pines-benchmark.log)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSAN\actual-ssan-indian-pines-benchmark.log`
- **Visual Outputs (2 Images):**

| Ground Truth (shared palette) | Predicted Classification Map | Training Loss Curve |
|:---:|:---:|:---:|
| ![SSAN Ground Truth](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSAN/outputs/ground_truth.png) | ![SSAN Prediction Map](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSAN/outputs/pred_SSAN.png) | ![SSAN Loss Curve](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/SSAN/outputs/loss_SSAN.png) |
| `...\SSAN\outputs\ground_truth.png` | `...\SSAN\outputs\pred_SSAN.png` | `...\SSAN\outputs\loss_SSAN.png` |

---

### 8. HyperRoPE-HiT
- **Model Category:** Rotary Position Embedding Transformer for Hyperspectral Data
- **Benchmark Performance (refreshed):** **OA: 94.42%** | **AA: 92.12%** | **Kappa: 93.64%** | **Training Time: 37.74 min**
- **Model Weight Checkpoint (.pth):**
  - Best Checkpoint: [`HiT/hyperrope-sst/checkpoints/IndianPine_best_model_hit.pth`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/HiT/hyperrope-sst/checkpoints/IndianPine_best_model_hit.pth)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\HiT\hyperrope-sst\checkpoints\IndianPine_best_model_hit.pth`
- **Classification Report & Execution Logs:**
  - Detailed Results Report: [`HiT/outputs/results_HiT.txt`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/HiT/outputs/results_HiT.txt)
  - Full Location: `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\HiT\outputs\results_HiT.txt`
  - Metrics Summary: [`HiT/outputs/metrics_HiT.csv`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/HiT/outputs/metrics_HiT.csv)
  - Full Kaggle Execution Log: [`HiT/actual-hyperrope-hit-indian-pines-benchmark.log`](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/HiT/actual-hyperrope-hit-indian-pines-benchmark.log)
- **Visual Outputs (3 Images):**

| Ground Truth (shared palette) | Predicted Classification Map | Training Loss Curve |
|:---:|:---:|:---:|
| ![HiT Ground Truth](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/HiT/outputs/ground_truth.png) | ![HiT Prediction Map](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/HiT/outputs/pred_HiT.png) | ![HiT Loss Curve](file:///C:/Users/LENOVO/Pictures/MinorProject/kaggle_results/HiT/outputs/loss_HiT.png) |
| `...\HiT\outputs\ground_truth.png` | `...\HiT\outputs\pred_HiT.png` | `...\HiT\outputs\loss_HiT.png` |

---

## 3. Quick Reference Master Table of All Outputs

| Model Name | Visual Output 1 (Prediction Map) | Visual Output 2 (Loss Curve / Ground Truth) | Model Weight File (.pt / .pth / .h5) | Classification Report / Result Log |
|:---|:---|:---|:---|:---|
| **SGSCF-Net (Ours)** | `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SGSCF-Net\Kaggle_Master_Outputs\Indian_Pines\Visualizations_PNG\pred_P11_B32.png` | `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SGSCF-Net\Kaggle_Master_Outputs\Indian_Pines\Visualizations_PNG\loss_P11_B32.png` | `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SGSCF-Net\sv_test1\models\SGSCFNet_IN_run1_OA0.9805.pt` | `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SGSCF-Net\Kaggle_Master_Outputs\Indian_Pines\Reports_TXT\grid_report_IN_P11_B32.txt` |
| **ViT** | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\ViT\outputs\pred_ViT.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\ViT\outputs\loss_ViT.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\ViT\vision_transformer\best_vit.pth` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\ViT\outputs\metrics_ViT.csv` |
| **DMC** | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\DMC\outputs\pred_DMC.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\DMC\outputs\ground_truth.png` | *(In-memory evaluation)* | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\DMC\DeepMatrixCapsules\classification_report_dmc.txt` |
| **A2S2K** | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\A2S2K\outputs\pred_A2S2K.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\A2S2K\outputs\ground_truth.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\A2S2K\A2S2KResNet\net_DBA.pt` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\A2S2K\A2S2KResNet\report\classification_Results_a2s2k.txt` |
| **CRV** | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\CRV\outputs\pred_CRV.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\CRV\outputs\loss_CRV.png` | `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\CRV\models\model_1.h5` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\CRV\CRV\classification_Results.txt` |
| **SSRN** | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSRN\SSRN\Cmaps\outputs/pred_SSRN.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSRN\SSRN\figure\new_fig.png` | *(In-memory evaluation)* | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSRN\SSRN\records\IN_evaluation_results.txt` |
| **SSFTT** | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSFTT\outputs\pred_SSFTT.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSFTT\outputs\ground_truth.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSFTT\cls_SSFTT_IP\cls_params\SSFTTnet_params.pth` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSFTT\cls_SSFTT_IP\cls_result\classification_report.txt` |
| **SSAN** | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSAN\outputs\pred_SSAN.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSAN\outputs\loss_SSAN.png` | `C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation\SSAN\model\ssan_step_1000.ckpt.data-00000-of-00001` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\SSAN\actual-ssan-indian-pines-benchmark.log` |
| **HiT** | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\HiT\outputs\pred_HiT.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\HiT\outputs\loss_HiT.png` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\HiT\hyperrope-sst\checkpoints\IndianPine_best_model_hit.pth` | `C:\Users\LENOVO\Pictures\MinorProject\kaggle_results\HiT\outputs\results_HiT.txt` |
