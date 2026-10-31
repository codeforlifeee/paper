# Master Hyperspectral Image (HSI) Benchmark Reference
## Single Source of Truth for 8 Deep Learning Baseline Architectures Across Pavia University, Salinas Scene & Indian Pines

> **Document Status:** Complete & Verified Local Source of Truth  
> **Total Models:** 8 Architectures | **Total Experimental Runs:** 24 Full GPU Benchmarks (8 PaviaU + 8 Salinas + 8 Indian Pines)  
> **Datasets Evaluated:** Pavia University (PaviaU), Salinas Scene (Salinas), Indian Pines (IndianPines)  
> **Kaggle Account Origin:** `tejassurya2004` (Autonomous GPU Execution Pipeline)  
> **Workspace Location:** `c:\Users\LENOVO\Desktop\FreshRuns`  

> [!IMPORTANT]
> ## 🚨 CRITICAL RESEARCH PAPER NOTES & IMPLEMENTATION SPECIFICATIONS (DO NOT MISS)
>
> **MANDATORY REFERENCE FOR DRAFTING THE RESEARCH PAPER'S METHODOLOGY, EXPERIMENTAL SETUP, AND BASELINE COMPARISONS:**
>
> All 24 staging / benchmark runs are completed and locally verified in `FreshRuns/`, with direct output artifacts in `outputs/`: `metrics_<MODEL>.csv`, `ground_truth.png`, `pred_<MODEL>.png`, `loss_<MODEL>.png`, `.npy` raw prediction arrays, and comprehensive execution logs.
>
> ### 📌 Key Architectural Differences, Model Tweaks & Setup Specifications:
> - **DeepMatrixCapsules (DMC) Input Dimension Constraint:** DMC can't take an 11×11 input. Its capsule layers need at least 13×13 (I checked 11 and 12, which crash, and 13 and 15, which work). The earlier root-level DMC notebook would have hit the same crash. I kept the 11×11 neighbourhood and padded each patch with a one-pixel border of zeros to 13×13, without touching the model. The alternative is a real 13×13 window, which is a one-line change in the generator.
> - **CRV-MHDL Framework & Keras 3 Migration:** CRV-MHDL is TensorFlow/Keras, not PyTorch. I patched it for Keras 3: replaced `np_utils`, rewrote `Adam(lr=, decay=)` in an equivalent form, and changed the checkpoint to `.keras`. I also fixed the author's code giving Salinas 13 output classes instead of 16.
> - **DMC GPU Tensor Device Placement:** DMC GPU tensors: a literal swap to `torch.FloatTensor` would put tensors on the CPU while the model is on the GPU and crash. Instead they're created on whatever device the script is using.
> - **Standardized 10/10/80 Split & Real Validation Loss:** Every model now uses a 10/10/80 split and produces a real `<MODEL>_vloss.npy`. SSFTT and CRV had no validation set, so I took one out of the held-out data; it's used only for the loss curve, never to pick the model. HyperRoPE originally used its test set as validation; it now gets a separate validation split.
> - **A2S2K-ResNet Patience & Loss Curve Logging:** A2S2K's early-stopping patience is now 50 (the author used 20). Its loss curve logs only the last batch's loss each epoch, which is how the author wrote it, so it will look noisier than the others.
> - **Preserved Author Per-Dataset Hyperparameters:** Where the author tuned per dataset, I kept their settings for anything you didn't standardize: DMC's PCA and learning rate, A2S2K's PCA components, HyperRoPE's PCA.
> - **Fixed Author Bugs:** Two author files were broken and are now fixed: A2S2K's and CRV's data-loading functions. SSFTT's `Salinas_train.py` imports a file that doesn't exist, so SSFTT runs from `IP_train.py` for all three datasets.
>
> ### 📊 Metric Table Nuances & Data Reporting Guidelines:
> - **Number Format / Precision:** The CSVs use fractions to 4 decimals. The finished SSAN results are percentages, and SSRN's are fractions to 6 decimals.
> - **Timing Metrics (TrainTime_min vs Total Execution Time):** `TrainTime_min` is each model's own training time; SSAN/SSRN recorded the whole script's run time. The full run time is printed in each notebook's final line.
> - **Kaggle Account Provenance:** All 24 runs were executed under Kaggle account `tejassurya2004` on NVIDIA Tesla T4 / P100 GPUs and downloaded directly to the local project structure.

---

## 1. Executive Summary & Experimental Standardization

This repository contains the rigorous, standardized Kaggle GPU benchmark results for **8 state-of-the-art Hyperspectral Image (HSI) deep learning baseline models** evaluated across all three canonical datasets: **Pavia University**, **Salinas Scene**, and **Indian Pines** (24 full experimental runs). Every model was trained and evaluated under strict standardized criteria to ensure true apples-to-apples comparability:

- **Train / Val / Test Split:** Strict **10% Training**, **10% Validation**, and **80% Testing** per class, generated with a deterministic, reproducible seed.
- **Spatial Window Size:** Fixed at **11×11** spatial patches across all models (DeepMatrixCapsules was minimally zero-padded to 13×13 solely to satisfy the spatial receptive field required by its convolutional capsule squash layers).
- **Training Budget:** Standardized **200 Epochs** with batch size **32**.
- **Architecture Preservation:** Author original model implementations, loss functions, activation topologies, and feature transformations were strictly preserved; only data loading, input spectral band channels, and classification heads were adapted to the target datasets.
- **Visualization & Metric Uniformity:** All 24 runs produced identical output artifacts including standardized classification maps rendered with the canonical 16-color RGB colormap, ground truth maps, epoch loss progression curves, and summary CSVs.

---

## 2. Master Benchmark Leaderboards

### 2.1 Pavia University Benchmark Leaderboard (610 × 340, 103 Bands, 9 Classes)

| Rank | Model | Overall Accuracy (OA) | Average Accuracy (AA) | Cohen's Kappa ($\kappa$) | Training Time (min) | Output Directory | Metric CSV Link |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **SSRN** | **99.87%** (0.998656) | 99.71% (0.997111) | **0.9982** | 11.36 min | [`outputs_ssrn_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia) | [metrics_SSRN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/outputs/metrics_SSRN.csv) |
| **2** | **HyperRoPE-SST** | **99.83%** (0.998334) | 99.77% (0.997681) | **0.9978** | 5.62 min | [`outputs_hyperrope_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia) | [metrics_HyperRoPE.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/metrics_HyperRoPE.csv) |
| **3** | **A2S2K-ResNet** | **99.83%** (0.998335) | **99.79%** (0.997851) | **0.9978** | 41.12 min | [`outputs_a2s2k_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia) | [metrics_A2S2K.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/metrics_A2S2K.csv) |
| **4** | **SSFTT** | **99.72%** (0.997224) | 99.40% (0.993970) | **0.9963** | 4.44 min | [`outputs_ssftt_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia) | [metrics_SSFTT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/metrics_SSFTT.csv) |
| **5** | **DeepMatrixCapsules** | **99.34%** (0.993367) | 99.03% (0.990260) | **0.9912** | 73.42 min | [`outputs_dmc_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia) | [metrics_DMC.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/metrics_DMC.csv) |
| **6** | **ViT** | **99.05%** (0.990500) | 98.90% (0.989000) | **0.9874** | 6.64 min | [`outputs_vit_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia) | [metrics_ViT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/metrics_ViT.csv) |
| **7** | **CRV-MHDL** | **98.40%** (0.984045) | 97.17% (0.971656) | **0.9789** | 2.89 min | [`outputs_crv_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia) | [metrics_CRV.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/metrics_CRV.csv) |
| **8** | **SSAN** | **76.82%** (0.768181) | 77.90% (0.778962) | **0.6944** | 8.70 min | [`outputs_ssan_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia) | [metrics_SSAN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia/outputs/metrics_SSAN.csv) |

### 2.2 Salinas Scene Benchmark Leaderboard (512 × 217, 204 Bands, 16 Classes)

| Rank | Model | Overall Accuracy (OA) | Average Accuracy (AA) | Cohen's Kappa ($\kappa$) | Training Time (min) | Output Directory | Metric CSV Link |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **A2S2K-ResNet** | **100.00%** (0.999977) | **99.99%** (0.999944) | **1.0000** | 10.34 min | [`outputs_a2s2k_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas) | [metrics_A2S2K.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/metrics_A2S2K.csv) |
| **2** | **SSRN** | **99.96%** (0.999584) | 99.93% (0.999289) | **0.9995** | 30.98 min | [`outputs_ssrn_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas) | [metrics_SSRN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/outputs/metrics_SSRN.csv) |
| **3** | **SSFTT** | **99.90%** (0.998961) | 99.92% (0.999250) | **0.9988** | 5.26 min | [`outputs_ssftt_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas) | [metrics_SSFTT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/metrics_SSFTT.csv) |
| **4** | **DeepMatrixCapsules** | **99.90%** (0.999007) | 99.88% (0.998842) | **0.9989** | 94.64 min | [`outputs_dmc_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas) | [metrics_DMC.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/metrics_DMC.csv) |
| **5** | **HyperRoPE-SST** | **99.85%** (0.998545) | 99.89% (0.998921) | **0.9984** | 5.92 min | [`outputs_hyperrope_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas) | [metrics_HyperRoPE.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/metrics_HyperRoPE.csv) |
| **6** | **SSAN** | **88.20%** (0.881973) | 94.44% (0.944399) | **0.8688** | 19.37 min | [`outputs_ssan_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas) | [metrics_SSAN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas/outputs/metrics_SSAN.csv) |
| **7** | **ViT** | **98.34%** (0.983400) | 99.16% (0.991600) | **0.9816** | 8.94 min | [`outputs_vit_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas) | [metrics_ViT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/metrics_ViT.csv) |
| **8** | **CRV-MHDL** | **96.30%** (0.962983) | 97.75% (0.977476) | **0.9587** | 3.55 min | [`outputs_crv_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas) | [metrics_CRV.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/metrics_CRV.csv) |

### 2.3 Indian Pines Benchmark Leaderboard (145 × 145, 200 Bands, 16 Classes)

| Rank | Model | Overall Accuracy (OA) | Average Accuracy (AA) | Cohen's Kappa ($\kappa$) | Training Time (min) | Output Directory | Metric CSV Link |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **SSRN** | **98.52%** (0.9852) | **79.99%** (0.7999) | **0.9831** | 6.29 min | [`outputs_ssrn_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines) | [metrics_SSRN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/metrics_SSRN.csv) |
| **2** | **SSFTT** | **98.45%** (0.9845) | **94.29%** (0.9429) | **0.9823** | 1.05 min | [`outputs_ssftt_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines) | [metrics_SSFTT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/metrics_SSFTT.csv) |
| **3** | **A2S2K-ResNet** | **98.11%** (0.9811) | **94.41%** (0.9441) | **0.9785** | 14.89 min | [`outputs_a2s2k_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines) | [metrics_A2S2K.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/metrics_A2S2K.csv) |
| **4** | **HyperRoPE-SST** | **94.07%** (0.9407) | **89.37%** (0.8937) | **0.9324** | 1.43 min | [`outputs_hyperrope_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines) | [metrics_HyperRoPE.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/metrics_HyperRoPE.csv) |
| **5** | **ViT** | **90.33%** (0.9033) | **89.12%** (0.8912) | **0.8895** | 1.68 min | [`outputs_vit_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines) | [metrics_ViT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/metrics_ViT.csv) |
| **6** | **DeepMatrixCapsules** | **82.55%** (0.8255) | **64.11%** (0.6411) | **0.7989** | 17.73 min | [`outputs_dmc_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines) | [metrics_DMC.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/metrics_DMC.csv) |
| **7** | **CRV-MHDL** | **67.20%** (0.6720) | **51.71%** (0.5171) | **0.6222** | 1.33 min | [`outputs_crv_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines) | [metrics_CRV.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/metrics_CRV.csv) |
| **8** | **SSAN** | **64.19%** (0.6419) | **72.77%** (0.7277) | **0.5979** | 14.00 min | [`outputs_ssan_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines) | [metrics_SSAN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/metrics_SSAN.csv) |

### 2.4 Cross-Dataset Synthesis Matrix (All 3 Standard HSI Benchmarks, 24 Runs)

| Model Architecture | PaviaU OA (%) | PaviaU AA (%) | PaviaU Time | Salinas OA (%) | Salinas AA (%) | Salinas Time | IP OA (%) | IP AA (%) | IP Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SSRN** | **99.87%** | 99.71% | 11.36 min | 99.96% | 99.93% | 30.98 min | **98.52%** | **79.99%** | **6.29 min** |
| **SSFTT** | 99.72% | 99.40% | 4.44 min | 99.90% | 99.92% | 5.26 min | **98.45%** | **94.29%** | **1.05 min** |
| **A2S2K-ResNet** | 99.83% | **99.79%** | 41.12 min | **100.00%** | **99.99%** | 10.34 min | **98.11%** | **94.41%** | **14.89 min** |
| **HyperRoPE-SST** | 99.83% | 99.77% | 5.62 min | 99.85% | 99.89% | 5.92 min | **94.07%** | **89.37%** | **1.43 min** |
| **ViT** | 99.05% | 98.90% | 6.64 min | 98.34% | 99.16% | 8.94 min | **90.33%** | **89.12%** | **1.68 min** |
| **DeepMatrixCapsules** | 99.34% | 99.03% | 73.42 min | 99.90% | 99.88% | 94.64 min | **82.55%** | **64.11%** | **17.73 min** |
| **CRV-MHDL** | 98.40% | 97.17% | 2.89 min | 96.30% | 97.75% | 3.55 min | **67.20%** | **51.71%** | **1.33 min** |
| **SSAN** | 76.82% | 77.90% | 8.70 min | 88.20% | 94.44% | 19.37 min | **64.19%** | **72.77%** | **14.00 min** |

---

## 3. Comprehensive Per-Run Dossiers (All 24 Benchmark Runs)

### 3.1 SSRN — Pavia University

- **Model Identifier:** `SSRN`
- **Target Dataset:** `Pavia University`
- **Local Staging / Output Directory:** [`outputs_ssrn_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-ssrn-paviau-benchmark`
- **Training Time:** 11.36 min (681.64s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.87%** | `99.87%` |
| **Average Accuracy (AA)** | **99.71%** | `99.71%` |
| **Cohen's Kappa ($\kappa$)** | **0.9982** | `0.9982` |
| **Training Duration** | **11.36 min (681.64s)** | `11.36 min (681.64s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Asphalt | **100.00%** |
| C02 | Meadows | **99.97%** |
| C03 | Gravel | **99.58%** |
| C04 | Trees | **100.00%** |
| C05 | Painted metal sheets | **99.81%** |
| C06 | Bare Soil | **100.00%** |
| C07 | Bitumen | **100.00%** |
| C08 | Self-Blocking Bricks | **99.22%** |
| C09 | Shadows | **98.82%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssrn_pavia/outputs/pred_SSRN.png` | [pred_SSRN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/outputs/pred_SSRN.png) |
| **Standardized Ground Truth Map** | `outputs_ssrn_pavia/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssrn_pavia/outputs/loss_SSRN.png` | [loss_SSRN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/outputs/loss_SSRN.png) |
| **Summary Metrics CSV** | `outputs_ssrn_pavia/outputs/metrics_SSRN.csv` | [metrics_SSRN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/outputs/metrics_SSRN.csv) |
| **Execution Telemetry Log** | `outputs_ssrn_pavia/actual-ssrn-paviau-benchmark.log` | [actual-ssrn-paviau-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/actual-ssrn-paviau-benchmark.log) |
| **Detailed Evaluation Report** | `outputs_ssrn_pavia/outputs/evaluation_report.txt` | [evaluation_report.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/outputs/evaluation_report.txt) |
| **Raw Output Array (.npy)** | `outputs_ssrn_pavia/SSRN_gt.npy` | [SSRN_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/SSRN_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_pavia/SSRN_loss.npy` | [SSRN_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/SSRN_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_pavia/SSRN_pred.npy` | [SSRN_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/SSRN_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_pavia/SSRN_vloss.npy` | [SSRN_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_pavia/SSRN_vloss.npy) |

**Implementation Notes:** Author's 3D-2D Spectral-Spatial Residual architecture trained with Adam and cross-entropy loss.

---

### 3.2 HyperRoPE-SST — Pavia University

- **Model Identifier:** `HyperRoPE-SST`
- **Target Dataset:** `Pavia University`
- **Local Staging / Output Directory:** [`outputs_hyperrope_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-hyperrope-sst-pavia-benchmark`
- **Training Time:** 5.62 min (337.34s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.83%** | `99.83%` |
| **Average Accuracy (AA)** | **99.77%** | `99.77%` |
| **Cohen's Kappa ($\kappa$)** | **0.9978** | `0.9978` |
| **Training Duration** | **5.62 min (337.34s)** | `5.62 min (337.34s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Asphalt | **99.94%** |
| C02 | Meadows | **99.95%** |
| C03 | Gravel | **99.17%** |
| C04 | Trees | **99.23%** |
| C05 | Painted metal sheets | **100.00%** |
| C06 | Bare Soil | **99.80%** |
| C07 | Bitumen | **100.00%** |
| C08 | Self-Blocking Bricks | **99.83%** |
| C09 | Shadows | **100.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_hyperrope_pavia/outputs/pred_HyperRoPE.png` | [pred_HyperRoPE.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/pred_HyperRoPE.png) |
| **Standardized Ground Truth Map** | `outputs_hyperrope_pavia/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_hyperrope_pavia/outputs/loss_HyperRoPE.png` | [loss_HyperRoPE.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/loss_HyperRoPE.png) |
| **Summary Metrics CSV** | `outputs_hyperrope_pavia/outputs/metrics_HyperRoPE.csv` | [metrics_HyperRoPE.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/metrics_HyperRoPE.csv) |
| **Execution Telemetry Log** | `outputs_hyperrope_pavia/actual-hyperrope-sst-pavia-benchmark.log` | [actual-hyperrope-sst-pavia-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/actual-hyperrope-sst-pavia-benchmark.log) |
| **Execution Telemetry Log** | `outputs_hyperrope_pavia/outputs/run_HyperRoPE.log` | [run_HyperRoPE.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/run_HyperRoPE.log) |
| **Detailed Evaluation Report** | `outputs_hyperrope_pavia/hyperrope-sst/results/Pavia_Hyper2DRoPE_Classification_Model_10_23_2026_10PM.json` | [Pavia_Hyper2DRoPE_Classification_Model_10_23_2026_10PM.json](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/hyperrope-sst/results/Pavia_Hyper2DRoPE_Classification_Model_10_23_2026_10PM.json) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_pavia/HyperRoPE_gt.npy` | [HyperRoPE_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/HyperRoPE_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_pavia/HyperRoPE_loss.npy` | [HyperRoPE_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/HyperRoPE_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_pavia/HyperRoPE_pred.npy` | [HyperRoPE_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/HyperRoPE_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_pavia/HyperRoPE_vloss.npy` | [HyperRoPE_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/HyperRoPE_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_pavia/outputs/HyperRoPE_gt.npy` | [HyperRoPE_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/HyperRoPE_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_pavia/outputs/HyperRoPE_loss.npy` | [HyperRoPE_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/HyperRoPE_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_pavia/outputs/HyperRoPE_pred.npy` | [HyperRoPE_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/HyperRoPE_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_pavia/outputs/HyperRoPE_vloss.npy` | [HyperRoPE_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_pavia/outputs/HyperRoPE_vloss.npy) |

**Implementation Notes:** Spatial-Spectral Transformer with 2D Rotary Position Embeddings.

---

### 3.3 A2S2K-ResNet — Pavia University

- **Model Identifier:** `A2S2K-ResNet`
- **Target Dataset:** `Pavia University`
- **Local Staging / Output Directory:** [`outputs_a2s2k_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-a2s2k-resnet-pavia-benchmark`
- **Training Time:** 41.12 min (2467.06s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.83%** | `99.83%` |
| **Average Accuracy (AA)** | **99.79%** | `99.79%` |
| **Cohen's Kappa ($\kappa$)** | **0.9978** | `0.9978` |
| **Training Duration** | **41.12 min (2467.06s)** | `41.12 min (2467.06s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Asphalt | **100.00%** |
| C02 | Meadows | **99.99%** |
| C03 | Gravel | **99.88%** |
| C04 | Trees | **99.92%** |
| C05 | Painted metal sheets | **100.00%** |
| C06 | Bare Soil | **100.00%** |
| C07 | Bitumen | **100.00%** |
| C08 | Self-Blocking Bricks | **98.28%** |
| C09 | Shadows | **100.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_a2s2k_pavia/outputs/pred_A2S2K.png` | [pred_A2S2K.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/pred_A2S2K.png) |
| **Standardized Ground Truth Map** | `outputs_a2s2k_pavia/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_a2s2k_pavia/outputs/loss_A2S2K.png` | [loss_A2S2K.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/loss_A2S2K.png) |
| **Summary Metrics CSV** | `outputs_a2s2k_pavia/outputs/metrics_A2S2K.csv` | [metrics_A2S2K.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/metrics_A2S2K.csv) |
| **Execution Telemetry Log** | `outputs_a2s2k_pavia/actual-a2s2k-resnet-pavia-benchmark.log` | [actual-a2s2k-resnet-pavia-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/actual-a2s2k-resnet-pavia-benchmark.log) |
| **Execution Telemetry Log** | `outputs_a2s2k_pavia/outputs/run_A2S2K.log` | [run_A2S2K.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/run_A2S2K.log) |
| **Detailed Evaluation Report** | `outputs_a2s2k_pavia/A2S2KResNet/report/S3KAIResNetpatch_11_UPsplit0.9lr0.001adam_kernel_24.txt` | [S3KAIResNetpatch_11_UPsplit0.9lr0.001adam_kernel_24.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/A2S2KResNet/report/S3KAIResNetpatch_11_UPsplit0.9lr0.001adam_kernel_24.txt) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_pavia/A2S2K_gt.npy` | [A2S2K_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/A2S2K_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_pavia/A2S2K_loss.npy` | [A2S2K_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/A2S2K_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_pavia/A2S2K_pred.npy` | [A2S2K_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/A2S2K_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_pavia/A2S2K_vloss.npy` | [A2S2K_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/A2S2K_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_pavia/outputs/A2S2K_gt.npy` | [A2S2K_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/A2S2K_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_pavia/outputs/A2S2K_loss.npy` | [A2S2K_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/A2S2K_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_pavia/outputs/A2S2K_pred.npy` | [A2S2K_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/A2S2K_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_pavia/outputs/A2S2K_vloss.npy` | [A2S2K_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_pavia/outputs/A2S2K_vloss.npy) |

**Implementation Notes:** Attention-based Adaptive Spectral-Spatial Kernel ResNet with 24 kernels.

---

### 3.4 SSFTT — Pavia University

- **Model Identifier:** `SSFTT`
- **Target Dataset:** `Pavia University`
- **Local Staging / Output Directory:** [`outputs_ssftt_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-hsi-ssftt-pavia-benchmark`
- **Training Time:** 4.44 min (266.55s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.72%** | `99.72%` |
| **Average Accuracy (AA)** | **99.40%** | `99.40%` |
| **Cohen's Kappa ($\kappa$)** | **0.9963** | `0.9963` |
| **Training Duration** | **4.44 min (266.55s)** | `4.44 min (266.55s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Asphalt | **100.00%** |
| C02 | Meadows | **99.97%** |
| C03 | Gravel | **98.75%** |
| C04 | Trees | **99.67%** |
| C05 | Painted metal sheets | **99.44%** |
| C06 | Bare Soil | **99.98%** |
| C07 | Bitumen | **100.00%** |
| C08 | Self-Blocking Bricks | **98.61%** |
| C09 | Shadows | **98.15%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssftt_pavia/outputs/pred_SSFTT.png` | [pred_SSFTT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/pred_SSFTT.png) |
| **Standardized Ground Truth Map** | `outputs_ssftt_pavia/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssftt_pavia/outputs/loss_SSFTT.png` | [loss_SSFTT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/loss_SSFTT.png) |
| **Summary Metrics CSV** | `outputs_ssftt_pavia/outputs/metrics_SSFTT.csv` | [metrics_SSFTT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/metrics_SSFTT.csv) |
| **Execution Telemetry Log** | `outputs_ssftt_pavia/actual-hsi-ssftt-pavia-benchmark.log` | [actual-hsi-ssftt-pavia-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/actual-hsi-ssftt-pavia-benchmark.log) |
| **Execution Telemetry Log** | `outputs_ssftt_pavia/outputs/run_SSFTT.log` | [run_SSFTT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/run_SSFTT.log) |
| **Detailed Evaluation Report** | `outputs_ssftt_pavia/cls_SSFTT_IP/cls_result/classification_report.txt` | [classification_report.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/cls_SSFTT_IP/cls_result/classification_report.txt) |
| **Raw Output Array (.npy)** | `outputs_ssftt_pavia/SSFTT_gt.npy` | [SSFTT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/SSFTT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_pavia/SSFTT_loss.npy` | [SSFTT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/SSFTT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_pavia/SSFTT_pred.npy` | [SSFTT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/SSFTT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_pavia/SSFTT_vloss.npy` | [SSFTT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/SSFTT_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_pavia/outputs/SSFTT_gt.npy` | [SSFTT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/SSFTT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_pavia/outputs/SSFTT_loss.npy` | [SSFTT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/SSFTT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_pavia/outputs/SSFTT_pred.npy` | [SSFTT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/SSFTT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_pavia/outputs/SSFTT_vloss.npy` | [SSFTT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_pavia/outputs/SSFTT_vloss.npy) |

**Implementation Notes:** Spectral-Spatial Feature Tokenizer Transformer.

---

### 3.5 DeepMatrixCapsules — Pavia University

- **Model Identifier:** `DeepMatrixCapsules`
- **Target Dataset:** `Pavia University`
- **Local Staging / Output Directory:** [`outputs_dmc_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-deepmatrixcapsules-pavia-benchmark`
- **Training Time:** 73.42 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.34%** | `99.34%` |
| **Average Accuracy (AA)** | **99.03%** | `99.03%` |
| **Cohen's Kappa ($\kappa$)** | **0.9912** | `0.9912` |
| **Training Duration** | **73.42 min** | `73.42 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Asphalt | **99.42%** |
| C02 | Meadows | **99.97%** |
| C03 | Gravel | **97.08%** |
| C04 | Trees | **98.65%** |
| C05 | Painted metal sheets | **100.00%** |
| C06 | Bare Soil | **99.73%** |
| C07 | Bitumen | **99.91%** |
| C08 | Self-Blocking Bricks | **96.74%** |
| C09 | Shadows | **99.74%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_dmc_pavia/outputs/pred_DMC.png` | [pred_DMC.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/pred_DMC.png) |
| **Standardized Ground Truth Map** | `outputs_dmc_pavia/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_dmc_pavia/outputs/loss_DMC.png` | [loss_DMC.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/loss_DMC.png) |
| **Summary Metrics CSV** | `outputs_dmc_pavia/outputs/metrics_DMC.csv` | [metrics_DMC.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/metrics_DMC.csv) |
| **Execution Telemetry Log** | `outputs_dmc_pavia/actual-deepmatrixcapsules-pavia-benchmark.log` | [actual-deepmatrixcapsules-pavia-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/actual-deepmatrixcapsules-pavia-benchmark.log) |
| **Execution Telemetry Log** | `outputs_dmc_pavia/outputs/run_DMC.log` | [run_DMC.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/run_DMC.log) |
| **Detailed Evaluation Report** | `outputs_dmc_pavia/DeepMatrixCapsules/classification_report_dmc.txt` | [classification_report_dmc.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/DeepMatrixCapsules/classification_report_dmc.txt) |
| **Raw Output Array (.npy)** | `outputs_dmc_pavia/DMC_gt.npy` | [DMC_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/DMC_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_pavia/DMC_loss.npy` | [DMC_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/DMC_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_pavia/DMC_pred.npy` | [DMC_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/DMC_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_pavia/DMC_vloss.npy` | [DMC_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/DMC_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_pavia/outputs/DMC_gt.npy` | [DMC_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/DMC_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_pavia/outputs/DMC_loss.npy` | [DMC_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/DMC_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_pavia/outputs/DMC_pred.npy` | [DMC_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/DMC_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_pavia/outputs/DMC_vloss.npy` | [DMC_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_pavia/outputs/DMC_vloss.npy) |

**Implementation Notes:** Capsule network with EM Routing. Input minimally zero-padded from 11x11 to 13x13.

---

### 3.6 ViT — Pavia University

- **Model Identifier:** `ViT`
- **Target Dataset:** `Pavia University`
- **Local Staging / Output Directory:** [`outputs_vit_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-vit-pavia-benchmark`
- **Training Time:** 6.64 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.05%** | `99.05%` |
| **Average Accuracy (AA)** | **98.90%** | `98.90%` |
| **Cohen's Kappa ($\kappa$)** | **0.9874** | `0.9874` |
| **Training Duration** | **6.64 min** | `6.64 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Asphalt | **99.60%** |
| C02 | Meadows | **99.38%** |
| C03 | Gravel | **94.52%** |
| C04 | Trees | **99.35%** |
| C05 | Painted metal sheets | **100.00%** |
| C06 | Bare Soil | **97.91%** |
| C07 | Bitumen | **99.91%** |
| C08 | Self-Blocking Bricks | **99.39%** |
| C09 | Shadows | **100.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_vit_pavia/outputs/pred_ViT.png` | [pred_ViT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/pred_ViT.png) |
| **Standardized Ground Truth Map** | `outputs_vit_pavia/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_vit_pavia/outputs/loss_ViT.png` | [loss_ViT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/loss_ViT.png) |
| **Summary Metrics CSV** | `outputs_vit_pavia/outputs/metrics_ViT.csv` | [metrics_ViT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/metrics_ViT.csv) |
| **Execution Telemetry Log** | `outputs_vit_pavia/actual-vit-pavia-benchmark.log` | [actual-vit-pavia-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/actual-vit-pavia-benchmark.log) |
| **Execution Telemetry Log** | `outputs_vit_pavia/outputs/run_ViT.log` | [run_ViT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/run_ViT.log) |
| **Raw Output Array (.npy)** | `outputs_vit_pavia/ViT_gt.npy` | [ViT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/ViT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_pavia/ViT_loss.npy` | [ViT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/ViT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_pavia/ViT_pred.npy` | [ViT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/ViT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_pavia/ViT_vloss.npy` | [ViT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/ViT_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_pavia/outputs/ViT_gt.npy` | [ViT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/ViT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_pavia/outputs/ViT_loss.npy` | [ViT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/ViT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_pavia/outputs/ViT_pred.npy` | [ViT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/ViT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_pavia/outputs/ViT_vloss.npy` | [ViT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_pavia/outputs/ViT_vloss.npy) |

**Implementation Notes:** Pure Vision Transformer baseline adapted for HSI patch tokenization.

---

### 3.7 CRV-MHDL — Pavia University

- **Model Identifier:** `CRV-MHDL`
- **Target Dataset:** `Pavia University`
- **Local Staging / Output Directory:** [`outputs_crv_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-crv-mhdl-pavia-benchmark`
- **Training Time:** 2.89 min (173.43s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **98.40%** | `98.40%` |
| **Average Accuracy (AA)** | **97.17%** | `97.17%` |
| **Cohen's Kappa ($\kappa$)** | **0.9789** | `0.9789` |
| **Training Duration** | **2.89 min (173.43s)** | `2.89 min (173.43s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Asphalt | **98.44%** |
| C02 | Meadows | **99.67%** |
| C03 | Gravel | **87.97%** |
| C04 | Trees | **98.37%** |
| C05 | Painted metal sheets | **99.63%** |
| C06 | Bare Soil | **99.85%** |
| C07 | Bitumen | **94.45%** |
| C08 | Self-Blocking Bricks | **96.64%** |
| C09 | Shadows | **99.47%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_crv_pavia/outputs/pred_CRV.png` | [pred_CRV.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/pred_CRV.png) |
| **Standardized Ground Truth Map** | `outputs_crv_pavia/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_crv_pavia/outputs/loss_CRV.png` | [loss_CRV.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/loss_CRV.png) |
| **Summary Metrics CSV** | `outputs_crv_pavia/outputs/metrics_CRV.csv` | [metrics_CRV.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/metrics_CRV.csv) |
| **Execution Telemetry Log** | `outputs_crv_pavia/actual-crv-mhdl-pavia-benchmark.log` | [actual-crv-mhdl-pavia-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/actual-crv-mhdl-pavia-benchmark.log) |
| **Execution Telemetry Log** | `outputs_crv_pavia/outputs/run_CRV.log` | [run_CRV.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/run_CRV.log) |
| **Detailed Evaluation Report** | `outputs_crv_pavia/CRV/classification_Results.txt` | [classification_Results.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/CRV/classification_Results.txt) |
| **Raw Output Array (.npy)** | `outputs_crv_pavia/CRV_gt.npy` | [CRV_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/CRV_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_pavia/CRV_loss.npy` | [CRV_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/CRV_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_pavia/CRV_pred.npy` | [CRV_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/CRV_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_pavia/CRV_vloss.npy` | [CRV_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/CRV_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_pavia/outputs/CRV_gt.npy` | [CRV_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/CRV_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_pavia/outputs/CRV_loss.npy` | [CRV_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/CRV_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_pavia/outputs/CRV_pred.npy` | [CRV_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/CRV_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_pavia/outputs/CRV_vloss.npy` | [CRV_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_pavia/outputs/CRV_vloss.npy) |

**Implementation Notes:** Cascaded Residual Vision Multi-Head Deep Learning architecture.

---

### 3.8 SSAN — Pavia University

- **Model Identifier:** `SSAN`
- **Target Dataset:** `Pavia University`
- **Local Staging / Output Directory:** [`outputs_ssan_pavia`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-ssan-paviau-benchmark`
- **Training Time:** 8.70 min (522.0s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **76.82%** | `76.82%` |
| **Average Accuracy (AA)** | **77.90%** | `77.90%` |
| **Cohen's Kappa ($\kappa$)** | **0.6944** | `0.6944` |
| **Training Duration** | **8.70 min (522.0s)** | `8.70 min (522.0s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Asphalt | **66.35%** |
| C02 | Meadows | **82.65%** |
| C03 | Gravel | **17.96%** |
| C04 | Trees | **94.73%** |
| C05 | Painted metal sheets | **99.83%** |
| C06 | Bare Soil | **60.36%** |
| C07 | Bitumen | **99.26%** |
| C08 | Self-Blocking Bricks | **81.22%** |
| C09 | Shadows | **98.55%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssan_pavia/outputs/pred_SSAN.png` | [pred_SSAN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia/outputs/pred_SSAN.png) |
| **Standardized Ground Truth Map** | `outputs_ssan_pavia/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssan_pavia/outputs/loss_SSAN.png` | [loss_SSAN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia/outputs/loss_SSAN.png) |
| **Summary Metrics CSV** | `outputs_ssan_pavia/outputs/metrics_SSAN.csv` | [metrics_SSAN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia/outputs/metrics_SSAN.csv) |
| **Execution Telemetry Log** | `outputs_ssan_pavia/actual-ssan-paviau-benchmark.log` | [actual-ssan-paviau-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia/actual-ssan-paviau-benchmark.log) |
| **Raw Output Array (.npy)** | `outputs_ssan_pavia/SSAN_gt.npy` | [SSAN_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia/SSAN_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssan_pavia/SSAN_loss.npy` | [SSAN_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia/SSAN_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssan_pavia/SSAN_pred.npy` | [SSAN_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_pavia/SSAN_pred.npy) |

**Implementation Notes:** Spatial-Spectral Attention Network implemented in TensorFlow 1.x compatibility mode.

---

### 3.9 A2S2K-ResNet — Salinas Scene

- **Model Identifier:** `A2S2K-ResNet`
- **Target Dataset:** `Salinas Scene`
- **Local Staging / Output Directory:** [`outputs_a2s2k_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-a2s2k-resnet-salinas-benchmark`
- **Training Time:** 10.34 min (620.48s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **100.00%** | `100.00%` |
| **Average Accuracy (AA)** | **99.99%** | `99.99%` |
| **Cohen's Kappa ($\kappa$)** | **1.0000** | `1.0000` |
| **Training Duration** | **10.34 min (620.48s)** | `10.34 min (620.48s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Brocoli_green_weeds_1 | **100.00%** |
| C02 | Brocoli_green_weeds_2 | **100.00%** |
| C03 | Fallow | **100.00%** |
| C04 | Fallow_rough_plow | **99.91%** |
| C05 | Fallow_smooth | **100.00%** |
| C06 | Stubble | **100.00%** |
| C07 | Celery | **100.00%** |
| C08 | Grapes_untrained | **100.00%** |
| C09 | Soil_vinyard_develop | **100.00%** |
| C10 | Corn_senesced_green_weeds | **100.00%** |
| C11 | Lettuce_romaine_4wk | **100.00%** |
| C12 | Lettuce_romaine_5wk | **100.00%** |
| C13 | Lettuce_romaine_6wk | **100.00%** |
| C14 | Lettuce_romaine_7wk | **100.00%** |
| C15 | Vinyard_untrained | **100.00%** |
| C16 | Vinyard_vertical_trellis | **100.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_a2s2k_salinas/outputs/pred_A2S2K.png` | [pred_A2S2K.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/pred_A2S2K.png) |
| **Standardized Ground Truth Map** | `outputs_a2s2k_salinas/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_a2s2k_salinas/outputs/loss_A2S2K.png` | [loss_A2S2K.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/loss_A2S2K.png) |
| **Summary Metrics CSV** | `outputs_a2s2k_salinas/outputs/metrics_A2S2K.csv` | [metrics_A2S2K.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/metrics_A2S2K.csv) |
| **Execution Telemetry Log** | `outputs_a2s2k_salinas/actual-a2s2k-resnet-salinas-benchmark.log` | [actual-a2s2k-resnet-salinas-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/actual-a2s2k-resnet-salinas-benchmark.log) |
| **Execution Telemetry Log** | `outputs_a2s2k_salinas/outputs/run_A2S2K.log` | [run_A2S2K.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/run_A2S2K.log) |
| **Detailed Evaluation Report** | `outputs_a2s2k_salinas/A2S2KResNet/report/S3KAIResNetpatch_11_SVsplit0.9lr0.001adam_kernel_24.txt` | [S3KAIResNetpatch_11_SVsplit0.9lr0.001adam_kernel_24.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/A2S2KResNet/report/S3KAIResNetpatch_11_SVsplit0.9lr0.001adam_kernel_24.txt) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_salinas/A2S2K_gt.npy` | [A2S2K_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/A2S2K_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_salinas/A2S2K_loss.npy` | [A2S2K_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/A2S2K_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_salinas/A2S2K_pred.npy` | [A2S2K_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/A2S2K_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_salinas/A2S2K_vloss.npy` | [A2S2K_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/A2S2K_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_salinas/outputs/A2S2K_gt.npy` | [A2S2K_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/A2S2K_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_salinas/outputs/A2S2K_loss.npy` | [A2S2K_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/A2S2K_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_salinas/outputs/A2S2K_pred.npy` | [A2S2K_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/A2S2K_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_salinas/outputs/A2S2K_vloss.npy` | [A2S2K_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_salinas/outputs/A2S2K_vloss.npy) |

**Implementation Notes:** Flawless performance: 15 out of 16 classes achieved perfect 100.00% accuracy.

---

### 3.10 SSRN — Salinas Scene

- **Model Identifier:** `SSRN`
- **Target Dataset:** `Salinas Scene`
- **Local Staging / Output Directory:** [`outputs_ssrn_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-ssrn-salinas-benchmark`
- **Training Time:** 30.98 min (1691.16s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.96%** | `99.96%` |
| **Average Accuracy (AA)** | **99.93%** | `99.93%` |
| **Cohen's Kappa ($\kappa$)** | **0.9995** | `0.9995` |
| **Training Duration** | **30.98 min (1691.16s)** | `30.98 min (1691.16s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Brocoli_green_weeds_1 | **100.00%** |
| C02 | Brocoli_green_weeds_2 | **100.00%** |
| C03 | Fallow | **100.00%** |
| C04 | Fallow_rough_plow | **100.00%** |
| C05 | Fallow_smooth | **99.51%** |
| C06 | Stubble | **100.00%** |
| C07 | Celery | **100.00%** |
| C08 | Grapes_untrained | **100.00%** |
| C09 | Soil_vinyard_develop | **100.00%** |
| C10 | Corn_senesced_green_weeds | **100.00%** |
| C11 | Lettuce_romaine_4wk | **100.00%** |
| C12 | Lettuce_romaine_5wk | **100.00%** |
| C13 | Lettuce_romaine_6wk | **100.00%** |
| C14 | Lettuce_romaine_7wk | **99.91%** |
| C15 | Vinyard_untrained | **99.90%** |
| C16 | Vinyard_vertical_trellis | **100.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssrn_salinas/outputs/pred_SSRN.png` | [pred_SSRN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/outputs/pred_SSRN.png) |
| **Standardized Ground Truth Map** | `outputs_ssrn_salinas/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssrn_salinas/outputs/loss_SSRN.png` | [loss_SSRN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/outputs/loss_SSRN.png) |
| **Summary Metrics CSV** | `outputs_ssrn_salinas/outputs/metrics_SSRN.csv` | [metrics_SSRN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/outputs/metrics_SSRN.csv) |
| **Execution Telemetry Log** | `outputs_ssrn_salinas/actual-ssrn-salinas-benchmark.log` | [actual-ssrn-salinas-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/actual-ssrn-salinas-benchmark.log) |
| **Raw Output Array (.npy)** | `outputs_ssrn_salinas/SSRN_gt.npy` | [SSRN_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/SSRN_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_salinas/SSRN_loss.npy` | [SSRN_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/SSRN_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_salinas/SSRN_pred.npy` | [SSRN_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/SSRN_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_salinas/SSRN_vloss.npy` | [SSRN_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_salinas/SSRN_vloss.npy) |

**Implementation Notes:** Resolved Salinas sample index allocation boundary condition in Version 2.

---

### 3.11 SSFTT — Salinas Scene

- **Model Identifier:** `SSFTT`
- **Target Dataset:** `Salinas Scene`
- **Local Staging / Output Directory:** [`outputs_ssftt_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-hsi-ssftt-salinas-benchmark`
- **Training Time:** 5.26 min (315.31s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.90%** | `99.90%` |
| **Average Accuracy (AA)** | **99.92%** | `99.92%` |
| **Cohen's Kappa ($\kappa$)** | **0.9988** | `0.9988` |
| **Training Duration** | **5.26 min (315.31s)** | `5.26 min (315.31s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Brocoli_green_weeds_1 | **100.00%** |
| C02 | Brocoli_green_weeds_2 | **100.00%** |
| C03 | Fallow | **100.00%** |
| C04 | Fallow_rough_plow | **100.00%** |
| C05 | Fallow_smooth | **99.72%** |
| C06 | Stubble | **99.94%** |
| C07 | Celery | **100.00%** |
| C08 | Grapes_untrained | **100.00%** |
| C09 | Soil_vinyard_develop | **100.00%** |
| C10 | Corn_senesced_green_weeds | **99.77%** |
| C11 | Lettuce_romaine_4wk | **100.00%** |
| C12 | Lettuce_romaine_5wk | **99.87%** |
| C13 | Lettuce_romaine_6wk | **100.00%** |
| C14 | Lettuce_romaine_7wk | **100.00%** |
| C15 | Vinyard_untrained | **99.50%** |
| C16 | Vinyard_vertical_trellis | **100.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssftt_salinas/outputs/pred_SSFTT.png` | [pred_SSFTT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/pred_SSFTT.png) |
| **Standardized Ground Truth Map** | `outputs_ssftt_salinas/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssftt_salinas/outputs/loss_SSFTT.png` | [loss_SSFTT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/loss_SSFTT.png) |
| **Summary Metrics CSV** | `outputs_ssftt_salinas/outputs/metrics_SSFTT.csv` | [metrics_SSFTT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/metrics_SSFTT.csv) |
| **Execution Telemetry Log** | `outputs_ssftt_salinas/actual-hsi-ssftt-salinas-benchmark.log` | [actual-hsi-ssftt-salinas-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/actual-hsi-ssftt-salinas-benchmark.log) |
| **Execution Telemetry Log** | `outputs_ssftt_salinas/outputs/run_SSFTT.log` | [run_SSFTT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/run_SSFTT.log) |
| **Detailed Evaluation Report** | `outputs_ssftt_salinas/cls_SSFTT_IP/cls_result/classification_report.txt` | [classification_report.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/cls_SSFTT_IP/cls_result/classification_report.txt) |
| **Raw Output Array (.npy)** | `outputs_ssftt_salinas/SSFTT_gt.npy` | [SSFTT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/SSFTT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_salinas/SSFTT_loss.npy` | [SSFTT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/SSFTT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_salinas/SSFTT_pred.npy` | [SSFTT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/SSFTT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_salinas/SSFTT_vloss.npy` | [SSFTT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/SSFTT_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_salinas/outputs/SSFTT_gt.npy` | [SSFTT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/SSFTT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_salinas/outputs/SSFTT_loss.npy` | [SSFTT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/SSFTT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_salinas/outputs/SSFTT_pred.npy` | [SSFTT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/SSFTT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_salinas/outputs/SSFTT_vloss.npy` | [SSFTT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_salinas/outputs/SSFTT_vloss.npy) |

**Implementation Notes:** 11 of 16 classes achieved perfect 100% accuracy with 99.90% overall accuracy.

---

### 3.12 DeepMatrixCapsules — Salinas Scene

- **Model Identifier:** `DeepMatrixCapsules`
- **Target Dataset:** `Salinas Scene`
- **Local Staging / Output Directory:** [`outputs_dmc_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-deepmatrixcapsules-salinas-benchmark`
- **Training Time:** 94.64 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.90%** | `99.90%` |
| **Average Accuracy (AA)** | **99.88%** | `99.88%` |
| **Cohen's Kappa ($\kappa$)** | **0.9989** | `0.9989` |
| **Training Duration** | **94.64 min** | `94.64 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Brocoli_green_weeds_1 | **100.00%** |
| C02 | Brocoli_green_weeds_2 | **100.00%** |
| C03 | Fallow | **100.00%** |
| C04 | Fallow_rough_plow | **99.91%** |
| C05 | Fallow_smooth | **99.16%** |
| C06 | Stubble | **100.00%** |
| C07 | Celery | **100.00%** |
| C08 | Grapes_untrained | **99.90%** |
| C09 | Soil_vinyard_develop | **100.00%** |
| C10 | Corn_senesced_green_weeds | **99.89%** |
| C11 | Lettuce_romaine_4wk | **100.00%** |
| C12 | Lettuce_romaine_5wk | **100.00%** |
| C13 | Lettuce_romaine_6wk | **100.00%** |
| C14 | Lettuce_romaine_7wk | **99.53%** |
| C15 | Vinyard_untrained | **99.90%** |
| C16 | Vinyard_vertical_trellis | **99.86%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_dmc_salinas/outputs/pred_DMC.png` | [pred_DMC.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/pred_DMC.png) |
| **Standardized Ground Truth Map** | `outputs_dmc_salinas/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_dmc_salinas/outputs/loss_DMC.png` | [loss_DMC.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/loss_DMC.png) |
| **Summary Metrics CSV** | `outputs_dmc_salinas/outputs/metrics_DMC.csv` | [metrics_DMC.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/metrics_DMC.csv) |
| **Execution Telemetry Log** | `outputs_dmc_salinas/actual-deepmatrixcapsules-salinas-benchmark.log` | [actual-deepmatrixcapsules-salinas-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/actual-deepmatrixcapsules-salinas-benchmark.log) |
| **Execution Telemetry Log** | `outputs_dmc_salinas/outputs/run_DMC.log` | [run_DMC.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/run_DMC.log) |
| **Detailed Evaluation Report** | `outputs_dmc_salinas/DeepMatrixCapsules/classification_report_dmc.txt` | [classification_report_dmc.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/DeepMatrixCapsules/classification_report_dmc.txt) |
| **Raw Output Array (.npy)** | `outputs_dmc_salinas/DMC_gt.npy` | [DMC_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/DMC_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_salinas/DMC_loss.npy` | [DMC_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/DMC_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_salinas/DMC_pred.npy` | [DMC_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/DMC_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_salinas/DMC_vloss.npy` | [DMC_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/DMC_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_salinas/outputs/DMC_gt.npy` | [DMC_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/DMC_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_salinas/outputs/DMC_loss.npy` | [DMC_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/DMC_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_salinas/outputs/DMC_pred.npy` | [DMC_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/DMC_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_salinas/outputs/DMC_vloss.npy` | [DMC_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_salinas/outputs/DMC_vloss.npy) |

**Implementation Notes:** 15 of 16 classes achieved 100.00% recall. Heavy computational routing graph required 94.64 min.

---

### 3.13 HyperRoPE-SST — Salinas Scene

- **Model Identifier:** `HyperRoPE-SST`
- **Target Dataset:** `Salinas Scene`
- **Local Staging / Output Directory:** [`outputs_hyperrope_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-hyperrope-sst-salinas-benchmark`
- **Training Time:** 5.92 min (355.45s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **99.85%** | `99.85%` |
| **Average Accuracy (AA)** | **99.89%** | `99.89%` |
| **Cohen's Kappa ($\kappa$)** | **0.9984** | `0.9984` |
| **Training Duration** | **5.92 min (355.45s)** | `5.92 min (355.45s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Brocoli_green_weeds_1 | **100.00%** |
| C02 | Brocoli_green_weeds_2 | **100.00%** |
| C03 | Fallow | **99.94%** |
| C04 | Fallow_rough_plow | **99.73%** |
| C05 | Fallow_smooth | **99.25%** |
| C06 | Stubble | **100.00%** |
| C07 | Celery | **100.00%** |
| C08 | Grapes_untrained | **99.83%** |
| C09 | Soil_vinyard_develop | **100.00%** |
| C10 | Corn_senesced_green_weeds | **100.00%** |
| C11 | Lettuce_romaine_4wk | **100.00%** |
| C12 | Lettuce_romaine_5wk | **100.00%** |
| C13 | Lettuce_romaine_6wk | **100.00%** |
| C14 | Lettuce_romaine_7wk | **100.00%** |
| C15 | Vinyard_untrained | **99.52%** |
| C16 | Vinyard_vertical_trellis | **100.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_hyperrope_salinas/outputs/pred_HyperRoPE.png` | [pred_HyperRoPE.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/pred_HyperRoPE.png) |
| **Standardized Ground Truth Map** | `outputs_hyperrope_salinas/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_hyperrope_salinas/outputs/loss_HyperRoPE.png` | [loss_HyperRoPE.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/loss_HyperRoPE.png) |
| **Summary Metrics CSV** | `outputs_hyperrope_salinas/outputs/metrics_HyperRoPE.csv` | [metrics_HyperRoPE.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/metrics_HyperRoPE.csv) |
| **Execution Telemetry Log** | `outputs_hyperrope_salinas/actual-hyperrope-sst-salinas-benchmark.log` | [actual-hyperrope-sst-salinas-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/actual-hyperrope-sst-salinas-benchmark.log) |
| **Execution Telemetry Log** | `outputs_hyperrope_salinas/outputs/run_HyperRoPE.log` | [run_HyperRoPE.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/run_HyperRoPE.log) |
| **Detailed Evaluation Report** | `outputs_hyperrope_salinas/hyperrope-sst/results/Salinas_Hyper2DRoPE_Classification_Model_10_23_2026_10PM.json` | [Salinas_Hyper2DRoPE_Classification_Model_10_23_2026_10PM.json](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/hyperrope-sst/results/Salinas_Hyper2DRoPE_Classification_Model_10_23_2026_10PM.json) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_salinas/HyperRoPE_gt.npy` | [HyperRoPE_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/HyperRoPE_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_salinas/HyperRoPE_loss.npy` | [HyperRoPE_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/HyperRoPE_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_salinas/HyperRoPE_pred.npy` | [HyperRoPE_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/HyperRoPE_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_salinas/HyperRoPE_vloss.npy` | [HyperRoPE_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/HyperRoPE_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_salinas/outputs/HyperRoPE_gt.npy` | [HyperRoPE_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/HyperRoPE_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_salinas/outputs/HyperRoPE_loss.npy` | [HyperRoPE_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/HyperRoPE_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_salinas/outputs/HyperRoPE_pred.npy` | [HyperRoPE_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/HyperRoPE_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_salinas/outputs/HyperRoPE_vloss.npy` | [HyperRoPE_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_salinas/outputs/HyperRoPE_vloss.npy) |

**Implementation Notes:** Fast convergence with 11 classes reaching 100.00% accuracy.

---

### 3.14 SSAN — Salinas Scene

- **Model Identifier:** `SSAN`
- **Target Dataset:** `Salinas Scene`
- **Local Staging / Output Directory:** [`outputs_ssan_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-ssan-salinas-benchmark`
- **Training Time:** 19.37 min (1162.2s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **88.20%** | `88.20%` |
| **Average Accuracy (AA)** | **94.44%** | `94.44%` |
| **Cohen's Kappa ($\kappa$)** | **0.8688** | `0.8688` |
| **Training Duration** | **19.37 min (1162.2s)** | `19.37 min (1162.2s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Brocoli_green_weeds_1 | **99.94%** |
| C02 | Brocoli_green_weeds_2 | **98.47%** |
| C03 | Fallow | **98.42%** |
| C04 | Fallow_rough_plow | **98.58%** |
| C05 | Fallow_smooth | **97.98%** |
| C06 | Stubble | **99.52%** |
| C07 | Celery | **99.26%** |
| C08 | Grapes_untrained | **57.89%** |
| C09 | Soil_vinyard_develop | **99.07%** |
| C10 | Corn_senesced_green_weeds | **94.80%** |
| C11 | Lettuce_romaine_4wk | **85.25%** |
| C12 | Lettuce_romaine_5wk | **97.92%** |
| C13 | Lettuce_romaine_6wk | **99.16%** |
| C14 | Lettuce_romaine_7wk | **97.47%** |
| C15 | Vinyard_untrained | **90.04%** |
| C16 | Vinyard_vertical_trellis | **97.26%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssan_salinas/outputs/pred_SSAN.png` | [pred_SSAN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas/outputs/pred_SSAN.png) |
| **Standardized Ground Truth Map** | `outputs_ssan_salinas/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssan_salinas/outputs/loss_SSAN.png` | [loss_SSAN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas/outputs/loss_SSAN.png) |
| **Summary Metrics CSV** | `outputs_ssan_salinas/outputs/metrics_SSAN.csv` | [metrics_SSAN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas/outputs/metrics_SSAN.csv) |
| **Execution Telemetry Log** | `outputs_ssan_salinas/actual-ssan-salinas-benchmark.log` | [actual-ssan-salinas-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas/actual-ssan-salinas-benchmark.log) |
| **Raw Output Array (.npy)** | `outputs_ssan_salinas/SSAN_gt.npy` | [SSAN_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas/SSAN_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssan_salinas/SSAN_loss.npy` | [SSAN_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas/SSAN_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssan_salinas/SSAN_pred.npy` | [SSAN_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_salinas/SSAN_pred.npy) |

**Implementation Notes:** TensorFlow attention mechanism trained with 3,200 optimization steps.

---

### 3.15 ViT — Salinas Scene

- **Model Identifier:** `ViT`
- **Target Dataset:** `Salinas Scene`
- **Local Staging / Output Directory:** [`outputs_vit_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-vit-salinas-benchmark`
- **Training Time:** 8.94 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **98.34%** | `98.34%` |
| **Average Accuracy (AA)** | **99.16%** | `99.16%` |
| **Cohen's Kappa ($\kappa$)** | **0.9816** | `0.9816` |
| **Training Duration** | **8.94 min** | `8.94 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Brocoli_green_weeds_1 | **98.57%** |
| C02 | Brocoli_green_weeds_2 | **100.00%** |
| C03 | Fallow | **99.24%** |
| C04 | Fallow_rough_plow | **99.46%** |
| C05 | Fallow_smooth | **99.81%** |
| C06 | Stubble | **99.94%** |
| C07 | Celery | **100.00%** |
| C08 | Grapes_untrained | **95.51%** |
| C09 | Soil_vinyard_develop | **100.00%** |
| C10 | Corn_senesced_green_weeds | **98.51%** |
| C11 | Lettuce_romaine_4wk | **99.53%** |
| C12 | Lettuce_romaine_5wk | **99.87%** |
| C13 | Lettuce_romaine_6wk | **100.00%** |
| C14 | Lettuce_romaine_7wk | **99.88%** |
| C15 | Vinyard_untrained | **96.25%** |
| C16 | Vinyard_vertical_trellis | **99.93%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_vit_salinas/outputs/pred_ViT.png` | [pred_ViT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/pred_ViT.png) |
| **Standardized Ground Truth Map** | `outputs_vit_salinas/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_vit_salinas/outputs/loss_ViT.png` | [loss_ViT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/loss_ViT.png) |
| **Summary Metrics CSV** | `outputs_vit_salinas/outputs/metrics_ViT.csv` | [metrics_ViT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/metrics_ViT.csv) |
| **Execution Telemetry Log** | `outputs_vit_salinas/actual-vit-salinas-benchmark.log` | [actual-vit-salinas-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/actual-vit-salinas-benchmark.log) |
| **Execution Telemetry Log** | `outputs_vit_salinas/outputs/run_ViT.log` | [run_ViT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/run_ViT.log) |
| **Raw Output Array (.npy)** | `outputs_vit_salinas/ViT_gt.npy` | [ViT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/ViT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_salinas/ViT_loss.npy` | [ViT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/ViT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_salinas/ViT_pred.npy` | [ViT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/ViT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_salinas/ViT_vloss.npy` | [ViT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/ViT_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_salinas/outputs/ViT_gt.npy` | [ViT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/ViT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_salinas/outputs/ViT_loss.npy` | [ViT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/ViT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_salinas/outputs/ViT_pred.npy` | [ViT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/ViT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_salinas/outputs/ViT_vloss.npy` | [ViT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_salinas/outputs/ViT_vloss.npy) |

**Implementation Notes:** High Average Accuracy (99.16%) across all 16 agricultural land-cover classes.

---

### 3.16 CRV-MHDL — Salinas Scene

- **Model Identifier:** `CRV-MHDL`
- **Target Dataset:** `Salinas Scene`
- **Local Staging / Output Directory:** [`outputs_crv_salinas`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-crv-mhdl-salinas-benchmark`
- **Training Time:** 3.55 min (213.14s)

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **96.30%** | `96.30%` |
| **Average Accuracy (AA)** | **97.75%** | `97.75%` |
| **Cohen's Kappa ($\kappa$)** | **0.9587** | `0.9587` |
| **Training Duration** | **3.55 min (213.14s)** | `3.55 min (213.14s)` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Brocoli_green_weeds_1 | **99.63%** |
| C02 | Brocoli_green_weeds_2 | **100.00%** |
| C03 | Fallow | **97.41%** |
| C04 | Fallow_rough_plow | **99.10%** |
| C05 | Fallow_smooth | **99.49%** |
| C06 | Stubble | **100.00%** |
| C07 | Celery | **99.16%** |
| C08 | Grapes_untrained | **97.52%** |
| C09 | Soil_vinyard_develop | **99.05%** |
| C10 | Corn_senesced_green_weeds | **97.67%** |
| C11 | Lettuce_romaine_4wk | **97.89%** |
| C12 | Lettuce_romaine_5wk | **98.90%** |
| C13 | Lettuce_romaine_6wk | **98.36%** |
| C14 | Lettuce_romaine_7wk | **99.18%** |
| C15 | Vinyard_untrained | **80.67%** |
| C16 | Vinyard_vertical_trellis | **99.93%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_crv_salinas/outputs/pred_CRV.png` | [pred_CRV.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/pred_CRV.png) |
| **Standardized Ground Truth Map** | `outputs_crv_salinas/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_crv_salinas/outputs/loss_CRV.png` | [loss_CRV.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/loss_CRV.png) |
| **Summary Metrics CSV** | `outputs_crv_salinas/outputs/metrics_CRV.csv` | [metrics_CRV.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/metrics_CRV.csv) |
| **Execution Telemetry Log** | `outputs_crv_salinas/actual-crv-mhdl-salinas-benchmark.log` | [actual-crv-mhdl-salinas-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/actual-crv-mhdl-salinas-benchmark.log) |
| **Execution Telemetry Log** | `outputs_crv_salinas/outputs/run_CRV.log` | [run_CRV.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/run_CRV.log) |
| **Detailed Evaluation Report** | `outputs_crv_salinas/CRV/classification_Results.txt` | [classification_Results.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/CRV/classification_Results.txt) |
| **Raw Output Array (.npy)** | `outputs_crv_salinas/CRV_gt.npy` | [CRV_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/CRV_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_salinas/CRV_loss.npy` | [CRV_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/CRV_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_salinas/CRV_pred.npy` | [CRV_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/CRV_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_salinas/CRV_vloss.npy` | [CRV_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/CRV_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_salinas/outputs/CRV_gt.npy` | [CRV_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/CRV_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_salinas/outputs/CRV_loss.npy` | [CRV_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/CRV_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_salinas/outputs/CRV_pred.npy` | [CRV_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/CRV_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_salinas/outputs/CRV_vloss.npy` | [CRV_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_salinas/outputs/CRV_vloss.npy) |

**Implementation Notes:** Keras/TensorFlow multi-head deep residual pipeline.

---

### 3.17 SSRN — Indian Pines

- **Model Identifier:** `SSRN`
- **Target Dataset:** `Indian Pines`
- **Local Staging / Output Directory:** [`outputs_ssrn_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-ssrn-indian-pines-benchmark`
- **Training Time:** 6.29 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **98.52%** | `98.52%` |
| **Average Accuracy (AA)** | **79.99%** | `79.99%` |
| **Cohen's Kappa ($\kappa$)** | **0.9831** | `0.9831` |
| **Training Duration** | **6.29 min** | `6.29 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Alfalfa | **0.00%** |
| C02 | Corn-notill | **96.44%** |
| C03 | Corn-mintill | **98.81%** |
| C04 | Corn | **98.18%** |
| C05 | Grass-pasture | **97.17%** |
| C06 | Grass-trees | **99.42%** |
| C07 | Grass-pasture-mowed | **0.00%** |
| C08 | Hay-windrowed | **95.89%** |
| C09 | Oats | **0.00%** |
| C10 | Soybean-notill | **99.12%** |
| C11 | Soybean-mintill | **99.24%** |
| C12 | Soybean-clean | **99.28%** |
| C13 | Wheat | **100.00%** |
| C14 | Woods | **99.10%** |
| C15 | Buildings-Grass-Trees-Drives | **100.00%** |
| C16 | Stone-Steel-Towers | **97.18%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssrn_indian_pines/outputs/pred_SSRN.png` | [pred_SSRN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/pred_SSRN.png) |
| **Standardized Ground Truth Map** | `outputs_ssrn_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssrn_indian_pines/outputs/loss_SSRN.png` | [loss_SSRN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/loss_SSRN.png) |
| **Summary Metrics CSV** | `outputs_ssrn_indian_pines/outputs/metrics_SSRN.csv` | [metrics_SSRN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/metrics_SSRN.csv) |
| **Execution Telemetry Log** | `outputs_ssrn_indian_pines/actual-ssrn-indian-pines-benchmark.log` | [actual-ssrn-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/actual-ssrn-indian-pines-benchmark.log) |
| **Detailed Evaluation Report** | `outputs_ssrn_indian_pines/SSRN/records/IN_evaluation_results.txt` | [IN_evaluation_results.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN/records/IN_evaluation_results.txt) |
| **Raw Output Array (.npy)** | `outputs_ssrn_indian_pines/SSRN_gt.npy` | [SSRN_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_indian_pines/SSRN_loss.npy` | [SSRN_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_indian_pines/SSRN_pred.npy` | [SSRN_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_indian_pines/SSRN_vloss.npy` | [SSRN_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN_vloss.npy) |

**Implementation Notes:** Author's 3D-2D Spectral-Spatial Residual architecture (Zhong et al.) from Kaggle account `tejassurya2004/actual-ssrn-indian-pines-benchmark`. Achieved highest OA (98.52%) and Kappa (0.9831) on Indian Pines test split with 6.29 min training. Severe imbalance in minority classes (Alfalfa, Grass-pasture-mowed, Oats) accounts for test AA of 79.99%.

---

### 3.18 SSFTT — Indian Pines

- **Model Identifier:** `SSFTT`
- **Target Dataset:** `Indian Pines`
- **Local Staging / Output Directory:** [`outputs_ssftt_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-hsi-ssftt-indian-pines-benchmark`
- **Training Time:** 1.05 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **98.45%** | `98.45%` |
| **Average Accuracy (AA)** | **94.29%** | `94.29%` |
| **Cohen's Kappa ($\kappa$)** | **0.9823** | `0.9823` |
| **Training Duration** | **1.05 min** | `1.05 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Alfalfa | **83.78%** |
| C02 | Corn-notill | **95.36%** |
| C03 | Corn-mintill | **99.70%** |
| C04 | Corn | **98.41%** |
| C05 | Grass-pasture | **99.22%** |
| C06 | Grass-trees | **99.32%** |
| C07 | Grass-pasture-mowed | **100.00%** |
| C08 | Hay-windrowed | **100.00%** |
| C09 | Oats | **43.75%** |
| C10 | Soybean-notill | **98.84%** |
| C11 | Soybean-mintill | **99.24%** |
| C12 | Soybean-clean | **96.63%** |
| C13 | Wheat | **100.00%** |
| C14 | Woods | **99.70%** |
| C15 | Buildings-Grass-Trees-Drives | **100.00%** |
| C16 | Stone-Steel-Towers | **94.67%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssftt_indian_pines/outputs/pred_SSFTT.png` | [pred_SSFTT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/pred_SSFTT.png) |
| **Standardized Ground Truth Map** | `outputs_ssftt_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssftt_indian_pines/outputs/loss_SSFTT.png` | [loss_SSFTT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/loss_SSFTT.png) |
| **Summary Metrics CSV** | `outputs_ssftt_indian_pines/outputs/metrics_SSFTT.csv` | [metrics_SSFTT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/metrics_SSFTT.csv) |
| **Execution Telemetry Log** | `outputs_ssftt_indian_pines/actual-hsi-ssftt-indian-pines-benchmark.log` | [actual-hsi-ssftt-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/actual-hsi-ssftt-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_ssftt_indian_pines/outputs/run_SSFTT.log` | [run_SSFTT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/run_SSFTT.log) |
| **Detailed Evaluation Report** | `outputs_ssftt_indian_pines/cls_SSFTT_IP/cls_result/classification_report.txt` | [classification_report.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/cls_SSFTT_IP/cls_result/classification_report.txt) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/SSFTT_gt.npy` | [SSFTT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/SSFTT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/SSFTT_loss.npy` | [SSFTT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/SSFTT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/SSFTT_pred.npy` | [SSFTT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/SSFTT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/SSFTT_vloss.npy` | [SSFTT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/SSFTT_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/outputs/SSFTT_gt.npy` | [SSFTT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/SSFTT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/outputs/SSFTT_loss.npy` | [SSFTT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/SSFTT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/outputs/SSFTT_pred.npy` | [SSFTT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/SSFTT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/outputs/SSFTT_vloss.npy` | [SSFTT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/SSFTT_vloss.npy) |

**Implementation Notes:** Spectral-Spatial Feature Tokenizer Transformer. Ultra-fast convergence (~1.05 min) on Kaggle GPU, achieving 98.45% OA and 94.29% AA.

---

### 3.19 A2S2K-ResNet — Indian Pines

- **Model Identifier:** `A2S2K-ResNet`
- **Target Dataset:** `Indian Pines`
- **Local Staging / Output Directory:** [`outputs_a2s2k_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-a2s2k-resnet-indian-pines-benchmark`
- **Training Time:** 14.89 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **98.11%** | `98.11%` |
| **Average Accuracy (AA)** | **94.41%** | `94.41%` |
| **Cohen's Kappa ($\kappa$)** | **0.9785** | `0.9785` |
| **Training Duration** | **14.89 min** | `14.89 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Alfalfa | **94.44%** |
| C02 | Corn-notill | **98.50%** |
| C03 | Corn-mintill | **98.47%** |
| C04 | Corn | **100.00%** |
| C05 | Grass-pasture | **97.96%** |
| C06 | Grass-trees | **97.18%** |
| C07 | Grass-pasture-mowed | **100.00%** |
| C08 | Hay-windrowed | **99.48%** |
| C09 | Oats | **50.00%** |
| C10 | Soybean-notill | **98.04%** |
| C11 | Soybean-mintill | **99.03%** |
| C12 | Soybean-clean | **94.29%** |
| C13 | Wheat | **97.53%** |
| C14 | Woods | **99.31%** |
| C15 | Buildings-Grass-Trees-Drives | **96.19%** |
| C16 | Stone-Steel-Towers | **90.12%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_a2s2k_indian_pines/outputs/pred_A2S2K.png` | [pred_A2S2K.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/pred_A2S2K.png) |
| **Standardized Ground Truth Map** | `outputs_a2s2k_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_a2s2k_indian_pines/outputs/loss_A2S2K.png` | [loss_A2S2K.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/loss_A2S2K.png) |
| **Summary Metrics CSV** | `outputs_a2s2k_indian_pines/outputs/metrics_A2S2K.csv` | [metrics_A2S2K.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/metrics_A2S2K.csv) |
| **Execution Telemetry Log** | `outputs_a2s2k_indian_pines/actual-a2s2k-resnet-indian-pines-benchmark.log` | [actual-a2s2k-resnet-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/actual-a2s2k-resnet-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_a2s2k_indian_pines/outputs/run_A2S2K.log` | [run_A2S2K.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/run_A2S2K.log) |
| **Detailed Evaluation Report** | `outputs_a2s2k_indian_pines/A2S2KResNet/report/S3KAIResNetpatch_11_INsplit0.9lr0.001adam_kernel_24.txt` | [S3KAIResNetpatch_11_INsplit0.9lr0.001adam_kernel_24.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2KResNet/report/S3KAIResNetpatch_11_INsplit0.9lr0.001adam_kernel_24.txt) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/A2S2K_gt.npy` | [A2S2K_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2K_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/A2S2K_loss.npy` | [A2S2K_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2K_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/A2S2K_pred.npy` | [A2S2K_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2K_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/A2S2K_vloss.npy` | [A2S2K_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2K_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/outputs/A2S2K_gt.npy` | [A2S2K_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/A2S2K_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/outputs/A2S2K_loss.npy` | [A2S2K_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/A2S2K_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/outputs/A2S2K_pred.npy` | [A2S2K_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/A2S2K_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/outputs/A2S2K_vloss.npy` | [A2S2K_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/A2S2K_vloss.npy) |

**Implementation Notes:** Attention-based Adaptive Spectral-Spatial Kernel ResNet with 24 adaptive 3D-2D kernels. Highest Average Accuracy (AA: 94.41%) across all models on Indian Pines.

---

### 3.20 HyperRoPE-SST — Indian Pines

- **Model Identifier:** `HyperRoPE-SST`
- **Target Dataset:** `Indian Pines`
- **Local Staging / Output Directory:** [`outputs_hyperrope_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-hyperrope-sst-indian-pines-benchmark`
- **Training Time:** 1.43 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **94.07%** | `94.07%` |
| **Average Accuracy (AA)** | **89.37%** | `89.37%` |
| **Cohen's Kappa ($\kappa$)** | **0.9324** | `0.9324` |
| **Training Duration** | **1.43 min** | `1.43 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Alfalfa | **94.44%** |
| C02 | Corn-notill | **93.87%** |
| C03 | Corn-mintill | **95.63%** |
| C04 | Corn | **83.07%** |
| C05 | Grass-pasture | **89.66%** |
| C06 | Grass-trees | **96.23%** |
| C07 | Grass-pasture-mowed | **90.91%** |
| C08 | Hay-windrowed | **100.00%** |
| C09 | Oats | **25.00%** |
| C10 | Soybean-notill | **94.94%** |
| C11 | Soybean-mintill | **95.87%** |
| C12 | Soybean-clean | **83.37%** |
| C13 | Wheat | **98.79%** |
| C14 | Woods | **94.37%** |
| C15 | Buildings-Grass-Trees-Drives | **93.83%** |
| C16 | Stone-Steel-Towers | **100.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_hyperrope_indian_pines/outputs/pred_HyperRoPE.png` | [pred_HyperRoPE.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/pred_HyperRoPE.png) |
| **Standardized Ground Truth Map** | `outputs_hyperrope_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_hyperrope_indian_pines/outputs/loss_HyperRoPE.png` | [loss_HyperRoPE.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/loss_HyperRoPE.png) |
| **Summary Metrics CSV** | `outputs_hyperrope_indian_pines/outputs/metrics_HyperRoPE.csv` | [metrics_HyperRoPE.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/metrics_HyperRoPE.csv) |
| **Execution Telemetry Log** | `outputs_hyperrope_indian_pines/actual-hyperrope-sst-indian-pines-benchmark.log` | [actual-hyperrope-sst-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/actual-hyperrope-sst-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_hyperrope_indian_pines/outputs/run_HyperRoPE.log` | [run_HyperRoPE.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/run_HyperRoPE.log) |
| **Detailed Evaluation Report** | `outputs_hyperrope_indian_pines/hyperrope-sst/results/IndianPine_Hyper2DRoPE_Classification_Model_10_27_2026_12PM.json` | [IndianPine_Hyper2DRoPE_Classification_Model_10_27_2026_12PM.json](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/hyperrope-sst/results/IndianPine_Hyper2DRoPE_Classification_Model_10_27_2026_12PM.json) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/HyperRoPE_gt.npy` | [HyperRoPE_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/HyperRoPE_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/HyperRoPE_loss.npy` | [HyperRoPE_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/HyperRoPE_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/HyperRoPE_pred.npy` | [HyperRoPE_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/HyperRoPE_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/HyperRoPE_vloss.npy` | [HyperRoPE_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/HyperRoPE_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/outputs/HyperRoPE_gt.npy` | [HyperRoPE_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/HyperRoPE_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/outputs/HyperRoPE_loss.npy` | [HyperRoPE_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/HyperRoPE_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/outputs/HyperRoPE_pred.npy` | [HyperRoPE_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/HyperRoPE_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/outputs/HyperRoPE_vloss.npy` | [HyperRoPE_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/HyperRoPE_vloss.npy) |

**Implementation Notes:** Spatial-Spectral Transformer with 2D Rotary Position Embeddings. Fast training (1.43 min) on Kaggle GPU with 94.07% OA.

---

### 3.21 ViT — Indian Pines

- **Model Identifier:** `ViT`
- **Target Dataset:** `Indian Pines`
- **Local Staging / Output Directory:** [`outputs_vit_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-vit-indian-pines-benchmark`
- **Training Time:** 1.68 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **90.33%** | `90.33%` |
| **Average Accuracy (AA)** | **89.12%** | `89.12%` |
| **Cohen's Kappa ($\kappa$)** | **0.8895** | `0.8895` |
| **Training Duration** | **1.68 min** | `1.68 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Alfalfa | **70.27%** |
| C02 | Corn-notill | **82.14%** |
| C03 | Corn-mintill | **78.46%** |
| C04 | Corn | **90.48%** |
| C05 | Grass-pasture | **90.18%** |
| C06 | Grass-trees | **96.75%** |
| C07 | Grass-pasture-mowed | **95.45%** |
| C08 | Hay-windrowed | **99.48%** |
| C09 | Oats | **68.75%** |
| C10 | Soybean-notill | **83.55%** |
| C11 | Soybean-mintill | **92.37%** |
| C12 | Soybean-clean | **89.26%** |
| C13 | Wheat | **97.56%** |
| C14 | Woods | **99.80%** |
| C15 | Buildings-Grass-Trees-Drives | **95.45%** |
| C16 | Stone-Steel-Towers | **96.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_vit_indian_pines/outputs/pred_ViT.png` | [pred_ViT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/pred_ViT.png) |
| **Standardized Ground Truth Map** | `outputs_vit_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_vit_indian_pines/outputs/loss_ViT.png` | [loss_ViT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/loss_ViT.png) |
| **Summary Metrics CSV** | `outputs_vit_indian_pines/outputs/metrics_ViT.csv` | [metrics_ViT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/metrics_ViT.csv) |
| **Execution Telemetry Log** | `outputs_vit_indian_pines/actual-vit-indian-pines-benchmark.log` | [actual-vit-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/actual-vit-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_vit_indian_pines/outputs/run_ViT.log` | [run_ViT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/run_ViT.log) |
| **Detailed Evaluation Report** | `outputs_vit_indian_pines/outputs/run_ViT.log` | [run_ViT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/run_ViT.log) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/ViT_gt.npy` | [ViT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/ViT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/ViT_loss.npy` | [ViT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/ViT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/ViT_pred.npy` | [ViT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/ViT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/ViT_vloss.npy` | [ViT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/ViT_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/outputs/ViT_gt.npy` | [ViT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ViT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/outputs/ViT_loss.npy` | [ViT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ViT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/outputs/ViT_pred.npy` | [ViT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ViT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/outputs/ViT_vloss.npy` | [ViT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ViT_vloss.npy) |

**Implementation Notes:** Vision Transformer baseline adapted for HSI patch tokenization. Trains in 1.68 min with 90.33% OA and 89.12% AA.

---

### 3.22 DeepMatrixCapsules — Indian Pines

- **Model Identifier:** `DeepMatrixCapsules`
- **Target Dataset:** `Indian Pines`
- **Local Staging / Output Directory:** [`outputs_dmc_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-deepmatrixcapsules-indian-pines-benchmark`
- **Training Time:** 17.73 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **82.55%** | `82.55%` |
| **Average Accuracy (AA)** | **64.11%** | `64.11%` |
| **Cohen's Kappa ($\kappa$)** | **0.7989** | `0.7989` |
| **Training Duration** | **17.73 min** | `17.73 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Alfalfa | **2.70%** |
| C02 | Corn-notill | **77.41%** |
| C03 | Corn-mintill | **73.04%** |
| C04 | Corn | **24.87%** |
| C05 | Grass-pasture | **90.96%** |
| C06 | Grass-trees | **96.40%** |
| C07 | Grass-pasture-mowed | **0.00%** |
| C08 | Hay-windrowed | **99.74%** |
| C09 | Oats | **0.00%** |
| C10 | Soybean-notill | **76.61%** |
| C11 | Soybean-mintill | **91.30%** |
| C12 | Soybean-clean | **47.16%** |
| C13 | Wheat | **91.46%** |
| C14 | Woods | **97.53%** |
| C15 | Buildings-Grass-Trees-Drives | **80.52%** |
| C16 | Stone-Steel-Towers | **76.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_dmc_indian_pines/outputs/pred_DMC.png` | [pred_DMC.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/pred_DMC.png) |
| **Standardized Ground Truth Map** | `outputs_dmc_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_dmc_indian_pines/outputs/loss_DMC.png` | [loss_DMC.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/loss_DMC.png) |
| **Summary Metrics CSV** | `outputs_dmc_indian_pines/outputs/metrics_DMC.csv` | [metrics_DMC.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/metrics_DMC.csv) |
| **Execution Telemetry Log** | `outputs_dmc_indian_pines/actual-deepmatrixcapsules-indian-pines-benchmark.log` | [actual-deepmatrixcapsules-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/actual-deepmatrixcapsules-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_dmc_indian_pines/outputs/run_DMC.log` | [run_DMC.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/run_DMC.log) |
| **Detailed Evaluation Report** | `outputs_dmc_indian_pines/DeepMatrixCapsules/classification_report_dmc.txt` | [classification_report_dmc.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DeepMatrixCapsules/classification_report_dmc.txt) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/DMC_gt.npy` | [DMC_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DMC_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/DMC_loss.npy` | [DMC_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DMC_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/DMC_pred.npy` | [DMC_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DMC_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/DMC_vloss.npy` | [DMC_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DMC_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/outputs/DMC_gt.npy` | [DMC_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/DMC_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/outputs/DMC_loss.npy` | [DMC_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/DMC_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/outputs/DMC_pred.npy` | [DMC_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/DMC_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/outputs/DMC_vloss.npy` | [DMC_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/DMC_vloss.npy) |

**Implementation Notes:** Matrix Capsule Network with EM routing. Input padded to 13x13 to satisfy capsule layer squash receptive field. Heavy routing computation (17.73 min).

---

### 3.23 CRV-MHDL — Indian Pines

- **Model Identifier:** `CRV-MHDL`
- **Target Dataset:** `Indian Pines`
- **Local Staging / Output Directory:** [`outputs_crv_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-crv-mhdl-indian-pines-benchmark`
- **Training Time:** 1.33 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **67.20%** | `67.20%` |
| **Average Accuracy (AA)** | **51.71%** | `51.71%` |
| **Cohen's Kappa ($\kappa$)** | **0.6222** | `0.6222` |
| **Training Duration** | **1.33 min** | `1.33 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Alfalfa | **10.81%** |
| C02 | Corn-notill | **47.64%** |
| C03 | Corn-mintill | **43.07%** |
| C04 | Corn | **9.52%** |
| C05 | Grass-pasture | **85.01%** |
| C06 | Grass-trees | **97.26%** |
| C07 | Grass-pasture-mowed | **9.09%** |
| C08 | Hay-windrowed | **84.29%** |
| C09 | Oats | **18.75%** |
| C10 | Soybean-notill | **63.37%** |
| C11 | Soybean-mintill | **75.27%** |
| C12 | Soybean-clean | **39.79%** |
| C13 | Wheat | **94.51%** |
| C14 | Woods | **97.63%** |
| C15 | Buildings-Grass-Trees-Drives | **39.29%** |
| C16 | Stone-Steel-Towers | **12.00%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_crv_indian_pines/outputs/pred_CRV.png` | [pred_CRV.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/pred_CRV.png) |
| **Standardized Ground Truth Map** | `outputs_crv_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_crv_indian_pines/outputs/loss_CRV.png` | [loss_CRV.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/loss_CRV.png) |
| **Summary Metrics CSV** | `outputs_crv_indian_pines/outputs/metrics_CRV.csv` | [metrics_CRV.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/metrics_CRV.csv) |
| **Execution Telemetry Log** | `outputs_crv_indian_pines/actual-crv-mhdl-indian-pines-benchmark.log` | [actual-crv-mhdl-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/actual-crv-mhdl-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_crv_indian_pines/outputs/run_CRV.log` | [run_CRV.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/run_CRV.log) |
| **Detailed Evaluation Report** | `outputs_crv_indian_pines/CRV/classification_Results.txt` | [classification_Results.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/CRV/classification_Results.txt) |
| **Raw Output Array (.npy)** | `outputs_crv_indian_pines/CRV_gt.npy` | [CRV_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/CRV_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_indian_pines/CRV_loss.npy` | [CRV_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/CRV_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_indian_pines/CRV_pred.npy` | [CRV_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/CRV_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_indian_pines/CRV_vloss.npy` | [CRV_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/CRV_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_indian_pines/outputs/CRV_gt.npy` | [CRV_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/CRV_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_indian_pines/outputs/CRV_loss.npy` | [CRV_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/CRV_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_indian_pines/outputs/CRV_pred.npy` | [CRV_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/CRV_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_crv_indian_pines/outputs/CRV_vloss.npy` | [CRV_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/CRV_vloss.npy) |

**Implementation Notes:** Cascaded Residual Vision Multi-Head Deep Learning (TensorFlow/Keras 3). Fast training (1.33 min) on Kaggle GPU.

---

### 3.24 SSAN — Indian Pines

- **Model Identifier:** `SSAN`
- **Target Dataset:** `Indian Pines`
- **Local Staging / Output Directory:** [`outputs_ssan_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-ssan-indian-pines-benchmark`
- **Training Time:** 14.00 min

#### Quantitative Performance Summary
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **64.19%** | `64.19%` |
| **Average Accuracy (AA)** | **72.77%** | `72.77%` |
| **Cohen's Kappa ($\kappa$)** | **0.5979** | `0.5979` |
| **Training Duration** | **14.00 min** | `14.00 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Accuracy (%) |
| :---: | :--- | :---: |
| C01 | Alfalfa | **71.88%** |
| C02 | Corn-notill | **65.96%** |
| C03 | Corn-mintill | **78.41%** |
| C04 | Corn | **76.36%** |
| C05 | Grass-pasture | **84.10%** |
| C06 | Grass-trees | **96.79%** |
| C07 | Grass-pasture-mowed | **60.00%** |
| C08 | Hay-windrowed | **98.20%** |
| C09 | Oats | **14.29%** |
| C10 | Soybean-notill | **72.15%** |
| C11 | Soybean-mintill | **25.76%** |
| C12 | Soybean-clean | **52.16%** |
| C13 | Wheat | **94.41%** |
| C14 | Woods | **92.58%** |
| C15 | Buildings-Grass-Trees-Drives | **82.80%** |
| C16 | Stone-Steel-Towers | **98.46%** |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssan_indian_pines/outputs/pred_SSAN.png` | [pred_SSAN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/pred_SSAN.png) |
| **Standardized Ground Truth Map** | `outputs_ssan_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssan_indian_pines/outputs/loss_SSAN.png` | [loss_SSAN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/loss_SSAN.png) |
| **Summary Metrics CSV** | `outputs_ssan_indian_pines/outputs/metrics_SSAN.csv` | [metrics_SSAN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/metrics_SSAN.csv) |
| **Execution Telemetry Log** | `outputs_ssan_indian_pines/actual-ssan-indian-pines-benchmark.log` | [actual-ssan-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/actual-ssan-indian-pines-benchmark.log) |
| **Detailed Evaluation Report** | `outputs_ssan_indian_pines/actual-ssan-indian-pines-benchmark.log` | [actual-ssan-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/actual-ssan-indian-pines-benchmark.log) |
| **Raw Output Array (.npy)** | `outputs_ssan_indian_pines/SSAN_gt.npy` | [SSAN_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/SSAN_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssan_indian_pines/SSAN_loss.npy` | [SSAN_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/SSAN_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssan_indian_pines/SSAN_pred.npy` | [SSAN_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/SSAN_pred.npy) |

**Implementation Notes:** Spatial-Spectral Attention Network in TensorFlow 1.x compatibility mode from Kaggle account `tejassurya2004/actual-ssan-indian-pines-benchmark`. 64.19% OA, 72.77% AA, 14.00 min training duration.

---

## 4. Benchmark Dataset Ground Truth & Canonical Color Palette

### 4.1 Pavia University Ground Truth Classes (9 Classes)
| Class Index | Class Name | Color Preview | Hex Code | RGB Triplet | Total Ground Truth Samples |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C01** | Asphalt | `#E6194B` | `#E6194B` | `(230, 25, 75)` | 6,631 samples |
| **C02** | Meadows | `#3CB44E` | `#3CB44E` | `(60, 180, 75)` | 18,649 samples |
| **C03** | Gravel | `#FFE119` | `#FFE119` | `(255, 225, 25)` | 2,099 samples |
| **C04** | Trees | `#4363D8` | `#4363D8` | `(67, 99, 216)` | 3,064 samples |
| **C05** | Painted metal sheets | `#F58231` | `#F58231` | `(245, 130, 49)` | 1,345 samples |
| **C06** | Bare Soil | `#911EB4` | `#911EB4` | `(145, 30, 180)` | 5,029 samples |
| **C07** | Bitumen | `#46F0F0` | `#46F0F0` | `(70, 240, 240)` | 1,330 samples |
| **C08** | Self-Blocking Bricks | `#F032E6` | `#F032E6` | `(240, 50, 230)` | 3,682 samples |
| **C09** | Shadows | `#BCF60C` | `#BCF60C` | `(188, 246, 12)` | 947 samples |
| **Total** | **All Labeled Pixels** | — | — | — | **42,776 samples** |

### 4.2 Salinas Scene Ground Truth Classes (16 Classes)
| Class Index | Class Name | Color Preview | Hex Code | RGB Triplet | Total Ground Truth Samples |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C01** | Brocoli_green_weeds_1 | `#E6194B` | `#E6194B` | `(230, 25, 75)` | 2,009 samples |
| **C02** | Brocoli_green_weeds_2 | `#3CB44E` | `#3CB44E` | `(60, 180, 75)` | 3,726 samples |
| **C03** | Fallow | `#FFE119` | `#FFE119` | `(255, 225, 25)` | 1,976 samples |
| **C04** | Fallow_rough_plow | `#4363D8` | `#4363D8` | `(67, 99, 216)` | 1,394 samples |
| **C05** | Fallow_smooth | `#F58231` | `#F58231` | `(245, 130, 49)` | 2,678 samples |
| **C06** | Stubble | `#911EB4` | `#911EB4` | `(145, 30, 180)` | 3,959 samples |
| **C07** | Celery | `#46F0F0` | `#46F0F0` | `(70, 240, 240)` | 3,579 samples |
| **C08** | Grapes_untrained | `#F032E6` | `#F032E6` | `(240, 50, 230)` | 11,271 samples |
| **C09** | Soil_vinyard_develop | `#BCF60C` | `#BCF60C` | `(188, 246, 12)` | 6,203 samples |
| **C10** | Corn_senesced_green_weeds | `#FABEBE` | `#FABEBE` | `(250, 190, 190)` | 3,278 samples |
| **C11** | Lettuce_romaine_4wk | `#008080` | `#008080` | `(0, 128, 128)` | 1,068 samples |
| **C12** | Lettuce_romaine_5wk | `#E6BEFF` | `#E6BEFF` | `(230, 190, 255)` | 1,927 samples |
| **C13** | Lettuce_romaine_6wk | `#9A6324` | `#9A6324` | `(154, 99, 36)` | 916 samples |
| **C14** | Lettuce_romaine_7wk | `#FFFAC8` | `#FFFAC8` | `(255, 250, 200)` | 1,070 samples |
| **C15** | Vinyard_untrained | `#800000` | `#800000` | `(128, 0, 0)` | 7,268 samples |
| **C16** | Vinyard_vertical_trellis | `#AAFFC3` | `#AAFFC3` | `(170, 255, 195)` | 1,807 samples |
| **Total** | **All Labeled Pixels** | — | — | — | **54,129 samples** |

### 4.3 Indian Pines Ground Truth Classes (16 Classes)
| Class Index | Class Name | Color Preview | Hex Code | RGB Triplet | Total Ground Truth Samples |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C01** | Alfalfa | `#E6194B` | `#E6194B` | `(230, 25, 75)` | 46 samples |
| **C02** | Corn-notill | `#3CB44E` | `#3CB44E` | `(60, 180, 75)` | 1,428 samples |
| **C03** | Corn-mintill | `#FFE119` | `#FFE119` | `(255, 225, 25)` | 830 samples |
| **C04** | Corn | `#4363D8` | `#4363D8` | `(67, 99, 216)` | 237 samples |
| **C05** | Grass-pasture | `#F58231` | `#F58231` | `(245, 130, 49)` | 483 samples |
| **C06** | Grass-trees | `#911EB4` | `#911EB4` | `(145, 30, 180)` | 730 samples |
| **C07** | Grass-pasture-mowed | `#46F0F0` | `#46F0F0` | `(70, 240, 240)` | 28 samples |
| **C08** | Hay-windrowed | `#F032E6` | `#F032E6` | `(240, 50, 230)` | 478 samples |
| **C09** | Oats | `#BCF60C` | `#BCF60C` | `(188, 246, 12)` | 20 samples |
| **C10** | Soybean-notill | `#FABEBE` | `#FABEBE` | `(250, 190, 190)` | 972 samples |
| **C11** | Soybean-mintill | `#008080` | `#008080` | `(0, 128, 128)` | 2,455 samples |
| **C12** | Soybean-clean | `#E6BEFF` | `#E6BEFF` | `(230, 190, 255)` | 593 samples |
| **C13** | Wheat | `#9A6324` | `#9A6324` | `(154, 99, 36)` | 205 samples |
| **C14** | Woods | `#FFFAC8` | `#FFFAC8` | `(255, 250, 200)` | 1,265 samples |
| **C15** | Buildings-Grass-Trees-Drives | `#800000` | `#800000` | `(128, 0, 0)` | 386 samples |
| **C16** | Stone-Steel-Towers | `#AAFFC3` | `#AAFFC3` | `(170, 255, 195)` | 93 samples |
| **Total** | **All Labeled Pixels** | — | — | — | **10,249 samples** |

---

## 5. Repository File Structure & Local Verification Guide

```text
FreshRuns/
├── MASTER_BENCHMARK_README.md               <- Comprehensive master document (all 24 runs)
├── README.md                                <- Root documentation mirror
├── README_INDIAN_PINES.md                   <- Dedicated Indian Pines single source of truth
├── auto_kaggle_pipeline.py                  <- 2-slot FIFO automated dispatcher (Pavia & Salinas)
├── auto_kaggle_pipeline_ip.py               <- 2-slot FIFO automated dispatcher (Indian Pines)
├── pipeline_status.json                     <- Status tracking for Pavia & Salinas
├── pipeline_status_ip.json                  <- Status tracking for Indian Pines
├── outputs_ssrn_pavia/                      <- SSRN on Pavia University (OA: 99.87%)
├── outputs_ssrn_salinas/                    <- SSRN on Salinas Scene (OA: 99.96%)
├── outputs_ssrn_indian_pines/               <- SSRN on Indian Pines (OA: 98.52%)
├── outputs_a2s2k_pavia/                     <- A2S2K-ResNet on Pavia University (OA: 99.83%)
├── outputs_a2s2k_salinas/                   <- A2S2K-ResNet on Salinas Scene (OA: 100.00%)
├── outputs_a2s2k_indian_pines/              <- A2S2K-ResNet on Indian Pines (OA: 98.11%)
├── outputs_hyperrope_pavia/                 <- HyperRoPE-SST on Pavia University (OA: 99.83%)
├── outputs_hyperrope_salinas/               <- HyperRoPE-SST on Salinas Scene (OA: 99.85%)
├── outputs_hyperrope_indian_pines/          <- HyperRoPE-SST on Indian Pines (OA: 94.07%)
├── outputs_ssftt_pavia/                     <- SSFTT on Pavia University (OA: 99.72%)
├── outputs_ssftt_salinas/                   <- SSFTT on Salinas Scene (OA: 99.90%)
├── outputs_ssftt_indian_pines/              <- SSFTT on Indian Pines (OA: 98.45%)
├── outputs_dmc_pavia/                       <- DeepMatrixCapsules on Pavia University (OA: 99.34%)
├── outputs_dmc_salinas/                     <- DeepMatrixCapsules on Salinas Scene (OA: 99.90%)
├── outputs_dmc_indian_pines/                <- DeepMatrixCapsules on Indian Pines (OA: 82.55%)
├── outputs_vit_pavia/                       <- ViT on Pavia University (OA: 99.05%)
├── outputs_vit_salinas/                     <- ViT on Salinas Scene (OA: 98.34%)
├── outputs_vit_indian_pines/                <- ViT on Indian Pines (OA: 90.33%)
├── outputs_crv_pavia/                       <- CRV-MHDL on Pavia University (OA: 98.40%)
├── outputs_crv_salinas/                     <- CRV-MHDL on Salinas Scene (OA: 96.30%)
├── outputs_crv_indian_pines/                <- CRV-MHDL on Indian Pines (OA: 67.20%)
├── outputs_ssan_pavia/                      <- SSAN on Pavia University (OA: 76.82%)
├── outputs_ssan_salinas/                    <- SSAN on Salinas Scene (OA: 88.20%)
└── outputs_ssan_indian_pines/               <- SSAN on Indian Pines (OA: 64.19%)
```

### Quick Python Verification Snippet
```python
import glob, os, pandas as pd
for d in sorted(glob.glob('outputs_*')):
    csvs = glob.glob(os.path.join(d, '**', 'metrics_*.csv'), recursive=True)
    if csvs:
        df = pd.read_csv(csvs[0])
        print(f"{d:35s} | Model: {df['Model'].iloc[0]:15s} | OA: {df['OA'].iloc[0]:.4f} | AA: {df['AA'].iloc[0]:.4f} | Kappa: {df['Kappa'].iloc[0]:.4f}")
```

---
*All 24 benchmark runs synthesized autonomously from Kaggle account `tejassurya2004` on 2026-10-27.*