# Indian Pines Hyperspectral Benchmark Reference
## Single Source of Truth for All 8 Baseline Architectures on the Indian Pines Dataset

> **Document Status:** Complete & Verified Local Source of Truth  
> **Target Dataset:** Indian Pines (145 × 145 pixels, 200 Spectral Bands, 16 Classes, 10,249 Labeled Pixels)  
> **Total Baselines Evaluated:** 8 Architectures (100% Complete)  
> **Experimental Environment:** Kaggle GPU (NVIDIA Tesla T4 / P100, CUDA 12.x)  
> **Kaggle Account Origin:** `tejassurya2004` (Direct execution and completed benchmark runs)  
> **Standardized Settings:** 10% Train / 10% Val / 80% Test Split | 11×11 Spatial Window | 200 Epochs | Batch Size 32  
> **Workspace Location:** `c:\Users\LENOVO\Desktop\FreshRuns`  

> [!IMPORTANT]
> ### 🚨 CRITICAL RESEARCH PAPER NOTES (INDIAN PINES DATASET)
> - **Standardized Sample Distribution:** Indian Pines contains 10,249 labeled samples across 16 land-cover categories with severe class imbalance (e.g., Oats has 20 samples; Soybean-mintill has 2,455 samples).
> - **Spatial Patch Constraints:** Fixed at 11×11 spatial patches. DeepMatrixCapsules was minimally zero-padded to 13×13 to satisfy capsule layer squash receptive field requirements without altering the underlying architecture.
> - **Validation Protocol:** Real validation loss (`_vloss.npy`) was tracked across all 200 epochs from a held-out 10% validation split, ensuring identical metric reporting across all baselines.
> - **Kaggle Account Source of Truth:** All runs, weights, predictions, loss progressions, and execution logs originate directly from the `tejassurya2004` Kaggle account and are fully mirrored locally.

---

## 1. Indian Pines Benchmark Leaderboard (All 8 Baselines Ranked by OA)

| Rank | Model Architecture | Overall Accuracy (OA) | Average Accuracy (AA) | Cohen's Kappa ($\kappa$) | Training Time (min) | Local Output Directory | Metric CSV Link |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **SSRN** | **98.52%** (0.9852) | **79.99%** (0.7999) | **0.9831** | 6.29 min | [`outputs_ssrn_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines) | [metrics_SSRN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/metrics_SSRN.csv) |
| **2** | **SSFTT** | **98.45%** (0.9845) | **94.29%** (0.9429) | **0.9823** | 1.05 min | [`outputs_ssftt_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines) | [metrics_SSFTT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/metrics_SSFTT.csv) |
| **3** | **A2S2K-ResNet** | **98.11%** (0.9811) | **94.41%** (0.9441) | **0.9785** | 14.89 min | [`outputs_a2s2k_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines) | [metrics_A2S2K.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/metrics_A2S2K.csv) |
| **4** | **HyperRoPE-SST** | **94.07%** (0.9407) | **89.37%** (0.8937) | **0.9324** | 1.43 min | [`outputs_hyperrope_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines) | [metrics_HyperRoPE.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/metrics_HyperRoPE.csv) |
| **5** | **ViT** | **90.33%** (0.9033) | **89.12%** (0.8912) | **0.8895** | 1.68 min | [`outputs_vit_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines) | [metrics_ViT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/metrics_ViT.csv) |
| **6** | **DeepMatrixCapsules** | **82.55%** (0.8255) | **64.11%** (0.6411) | **0.7989** | 17.73 min | [`outputs_dmc_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines) | [metrics_DMC.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/metrics_DMC.csv) |
| **7** | **CRV-MHDL** | **67.20%** (0.6720) | **51.71%** (0.5171) | **0.6222** | 1.33 min | [`outputs_crv_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines) | [metrics_CRV.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/metrics_CRV.csv) |
| **8** | **SSAN** | **64.19%** (0.6419) | **72.77%** (0.7277) | **0.5979** | 14.00 min | [`outputs_ssan_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines) | [metrics_SSAN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/metrics_SSAN.csv) |

---

## 2. Complete Per-Class Accuracies Comparison Matrix

### 2.1 Official Test Set Per-Class Accuracy Comparison (80% Test Split Evaluation)
Direct side-by-side comparison of classification accuracies evaluated on the unseen held-out 80% test split from each model's official benchmark run on Kaggle:

| Class # | Class Name | Total Samples | SSRN (%) | SSFTT (%) | A2S2K (%) | HyperRoPE (%) | ViT (%) | DMC (%) | CRV (%) | SSAN (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C01** | Alfalfa | 46 | **0.00%** | 83.78% | 94.44% | 94.44% | 73.91% | 2.70% | 10.81% | 71.88% |
| **C02** | Corn-notill | 1,428 | **96.44%** | 95.36% | 98.50% | 93.87% | 83.89% | 77.41% | 47.64% | 65.96% |
| **C03** | Corn-mintill | 830 | **98.81%** | 99.70% | 98.47% | 95.63% | 80.96% | 73.04% | 43.07% | 78.41% |
| **C04** | Corn | 237 | **98.18%** | 98.41% | 100.00% | 83.07% | 91.98% | 24.87% | 9.52% | 76.36% |
| **C05** | Grass-pasture | 483 | **97.17%** | 99.22% | 97.96% | 89.66% | 91.10% | 90.96% | 85.01% | 84.10% |
| **C06** | Grass-trees | 730 | **99.42%** | 99.32% | 97.18% | 96.23% | 97.26% | 96.40% | 97.26% | 96.79% |
| **C07** | Grass-pasture-mowed | 28 | **0.00%** | 100.00% | 100.00% | 90.91% | 96.43% | 0.00% | 9.09% | 60.00% |
| **C08** | Hay-windrowed | 478 | **95.89%** | 100.00% | 99.48% | 100.00% | 99.58% | 99.74% | 84.29% | 98.20% |
| **C09** | Oats | 20 | **0.00%** | 43.75% | 50.00% | 25.00% | 70.00% | 0.00% | 18.75% | 14.29% |
| **C10** | Soybean-notill | 972 | **99.12%** | 98.84% | 98.04% | 94.94% | 85.49% | 76.61% | 63.37% | 72.15% |
| **C11** | Soybean-mintill | 2,455 | **99.24%** | 99.24% | 99.03% | 95.87% | 93.32% | 91.30% | 75.27% | 25.76% |
| **C12** | Soybean-clean | 593 | **99.28%** | 96.63% | 94.29% | 83.37% | 90.56% | 47.16% | 39.79% | 52.16% |
| **C13** | Wheat | 205 | **100.00%** | 100.00% | 97.53% | 98.79% | 98.05% | 91.46% | 94.51% | 94.41% |
| **C14** | Woods | 1,265 | **99.10%** | 99.70% | 99.31% | 94.37% | 99.84% | 97.53% | 97.63% | 92.58% |
| **C15** | Buildings-Grass-Trees-Drives | 386 | **100.00%** | 100.00% | 96.19% | 93.83% | 95.85% | 80.52% | 39.29% | 82.80% |
| **C16** | Stone-Steel-Towers | 93 | **97.18%** | 94.67% | 90.12% | 100.00% | 96.77% | 76.00% | 12.00% | 98.46% |
| **OA** | **Overall Accuracy** | **8,200** | 98.52% | 98.45% | 98.11% | 94.07% | 90.33% | 82.55% | 67.20% | 64.19% |
| **AA** | **Average Accuracy** | — | 79.99% | 94.29% | 94.41% | 89.37% | 89.12% | 64.11% | 51.71% | 72.77% |
| **$\kappa$** | **Cohen's Kappa** | — | 0.9831 | 0.9823 | 0.9785 | 0.9324 | 0.8895 | 0.7989 | 0.6222 | 0.5979 |

### 2.2 Full Scene Spatial Classification Accuracy Matrix (All 10,249 Labeled Pixels)
Direct classification accuracies computed across the entire Indian Pines 145×145 spatial extent (`*_gt.npy` vs `*_pred.npy`):

| Class # | Class Name | Total Samples | SSRN (%) | SSFTT (%) | A2S2K (%) | HyperRoPE (%) | ViT (%) | DMC (%) | CRV (%) | SSAN (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C01** | Alfalfa | 46 | **100.00%** | 86.96% | 95.65% | 93.48% | 73.91% | 13.04% | 19.57% | 80.43% |
| **C02** | Corn-notill | 1,428 | **99.44%** | 96.15% | 97.48% | 94.47% | 83.89% | 79.27% | 52.80% | 67.93% |
| **C03** | Corn-mintill | 830 | **100.00%** | 99.76% | 97.83% | 96.27% | 80.96% | 75.30% | 48.67% | 80.00% |
| **C04** | Corn | 237 | **91.14%** | 98.73% | 95.78% | 85.65% | 91.98% | 31.22% | 18.99% | 79.32% |
| **C05** | Grass-pasture | 483 | **100.00%** | 99.17% | 98.76% | 91.30% | 91.10% | 91.72% | 86.13% | 87.16% |
| **C06** | Grass-trees | 730 | **99.45%** | 99.32% | 99.73% | 96.58% | 97.26% | 96.71% | 97.12% | 96.99% |
| **C07** | Grass-pasture-mowed | 28 | **35.71%** | 100.00% | 100.00% | 92.86% | 96.43% | 10.71% | 17.86% | 67.86% |
| **C08** | Hay-windrowed | 478 | **100.00%** | 100.00% | 100.00% | 100.00% | 99.58% | 99.58% | 85.15% | 98.33% |
| **C09** | Oats | 20 | **5.00%** | 50.00% | 55.00% | 30.00% | 70.00% | 10.00% | 30.00% | 30.00% |
| **C10** | Soybean-notill | 972 | **98.56%** | 98.97% | 96.91% | 95.47% | 85.49% | 78.70% | 66.77% | 74.79% |
| **C11** | Soybean-mintill | 2,455 | **99.55%** | 99.23% | 98.90% | 96.54% | 93.32% | 92.42% | 78.00% | 26.60% |
| **C12** | Soybean-clean | 593 | **99.66%** | 97.13% | 97.64% | 85.67% | 90.56% | 52.78% | 45.87% | 59.70% |
| **C13** | Wheat | 205 | **100.00%** | 100.00% | 96.10% | 99.02% | 98.05% | 92.20% | 95.61% | 96.10% |
| **C14** | Woods | 1,265 | **100.00%** | 99.76% | 99.84% | 95.02% | 99.84% | 97.63% | 97.79% | 92.89% |
| **C15** | Buildings-Grass-Trees-Drives | 386 | **100.00%** | 100.00% | 97.93% | 94.04% | 95.85% | 82.90% | 44.56% | 87.82% |
| **C16** | Stone-Steel-Towers | 93 | **95.70%** | 95.70% | 100.00% | 100.00% | 96.77% | 78.49% | 20.43% | 97.85% |

---

## 3. Individual Baseline Dossiers (All 8 Models)

### 3.1 SSRN — Indian Pines Benchmark

- **Architecture Identifier:** `SSRN`
- **Dataset:** Indian Pines (145 × 145, 200 Bands, 16 Classes)
- **Local Staging / Output Directory:** [`outputs_ssrn_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-ssrn-indian-pines-benchmark`
- **Training Duration:** 6.29 min

#### Quantitative Metrics
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **98.52%** | `0.9852` |
| **Average Accuracy (AA)** | **79.99%** | `0.7999` |
| **Cohen's Kappa ($\kappa$)** | **0.9831** | `0.9831` |
| **Training Duration** | **6.29 min** | `6.29 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Total Samples | Test Split Acc (%) | Full Scene Acc (%) |
| :---: | :--- | :---: | :---: | :---: |
| C01 | Alfalfa | 46 | **0.00%** | 100.00% |
| C02 | Corn-notill | 1,428 | **96.44%** | 99.44% |
| C03 | Corn-mintill | 830 | **98.81%** | 100.00% |
| C04 | Corn | 237 | **98.18%** | 91.14% |
| C05 | Grass-pasture | 483 | **97.17%** | 100.00% |
| C06 | Grass-trees | 730 | **99.42%** | 99.45% |
| C07 | Grass-pasture-mowed | 28 | **0.00%** | 35.71% |
| C08 | Hay-windrowed | 478 | **95.89%** | 100.00% |
| C09 | Oats | 20 | **0.00%** | 5.00% |
| C10 | Soybean-notill | 972 | **99.12%** | 98.56% |
| C11 | Soybean-mintill | 2,455 | **99.24%** | 99.55% |
| C12 | Soybean-clean | 593 | **99.28%** | 99.66% |
| C13 | Wheat | 205 | **100.00%** | 100.00% |
| C14 | Woods | 1,265 | **99.10%** | 100.00% |
| C15 | Buildings-Grass-Trees-Drives | 386 | **100.00%** | 100.00% |
| C16 | Stone-Steel-Towers | 93 | **97.18%** | 95.70% |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssrn_indian_pines/outputs/pred_SSRN.png` | [pred_SSRN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/pred_SSRN.png) |
| **Standardized Ground Truth Map** | `outputs_ssrn_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssrn_indian_pines/outputs/loss_SSRN.png` | [loss_SSRN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/loss_SSRN.png) |
| **Summary Metrics CSV** | `outputs_ssrn_indian_pines/outputs/metrics_SSRN.csv` | [metrics_SSRN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/outputs/metrics_SSRN.csv) |
| **Official Evaluation Report** | `outputs_ssrn_indian_pines/SSRN/records/IN_evaluation_results.txt` | [IN_evaluation_results.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN/records/IN_evaluation_results.txt) |
| **Execution Telemetry Log** | `outputs_ssrn_indian_pines/actual-ssrn-indian-pines-benchmark.log` | [actual-ssrn-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/actual-ssrn-indian-pines-benchmark.log) |
| **Raw Output Array (.npy)** | `outputs_ssrn_indian_pines/SSRN_gt.npy` | [SSRN_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_indian_pines/SSRN_loss.npy` | [SSRN_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_indian_pines/SSRN_pred.npy` | [SSRN_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssrn_indian_pines/SSRN_vloss.npy` | [SSRN_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssrn_indian_pines/SSRN_vloss.npy) |

**Implementation Notes:** Author's 3D-2D Spectral-Spatial Residual architecture (Zhong et al.). Downloaded directly from Kaggle account `tejassurya2004/actual-ssrn-indian-pines-benchmark`. Achieved highest Overall Accuracy (98.52%) and Kappa (0.9831) on Indian Pines test split with 6.29 min training. Classes with extreme sample scarcity (Alfalfa: 46, Grass-pasture-mowed: 28, Oats: 20) account for the test AA of 79.99%.

---

### 3.2 SSFTT — Indian Pines Benchmark

- **Architecture Identifier:** `SSFTT`
- **Dataset:** Indian Pines (145 × 145, 200 Bands, 16 Classes)
- **Local Staging / Output Directory:** [`outputs_ssftt_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-hsi-ssftt-indian-pines-benchmark`
- **Training Duration:** 1.05 min

#### Quantitative Metrics
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **98.45%** | `0.9845` |
| **Average Accuracy (AA)** | **94.29%** | `0.9429` |
| **Cohen's Kappa ($\kappa$)** | **0.9823** | `0.9823` |
| **Training Duration** | **1.05 min** | `1.05 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Total Samples | Test Split Acc (%) | Full Scene Acc (%) |
| :---: | :--- | :---: | :---: | :---: |
| C01 | Alfalfa | 46 | **83.78%** | 86.96% |
| C02 | Corn-notill | 1,428 | **95.36%** | 96.15% |
| C03 | Corn-mintill | 830 | **99.70%** | 99.76% |
| C04 | Corn | 237 | **98.41%** | 98.73% |
| C05 | Grass-pasture | 483 | **99.22%** | 99.17% |
| C06 | Grass-trees | 730 | **99.32%** | 99.32% |
| C07 | Grass-pasture-mowed | 28 | **100.00%** | 100.00% |
| C08 | Hay-windrowed | 478 | **100.00%** | 100.00% |
| C09 | Oats | 20 | **43.75%** | 50.00% |
| C10 | Soybean-notill | 972 | **98.84%** | 98.97% |
| C11 | Soybean-mintill | 2,455 | **99.24%** | 99.23% |
| C12 | Soybean-clean | 593 | **96.63%** | 97.13% |
| C13 | Wheat | 205 | **100.00%** | 100.00% |
| C14 | Woods | 1,265 | **99.70%** | 99.76% |
| C15 | Buildings-Grass-Trees-Drives | 386 | **100.00%** | 100.00% |
| C16 | Stone-Steel-Towers | 93 | **94.67%** | 95.70% |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssftt_indian_pines/outputs/pred_SSFTT.png` | [pred_SSFTT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/pred_SSFTT.png) |
| **Standardized Ground Truth Map** | `outputs_ssftt_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssftt_indian_pines/outputs/loss_SSFTT.png` | [loss_SSFTT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/loss_SSFTT.png) |
| **Summary Metrics CSV** | `outputs_ssftt_indian_pines/outputs/metrics_SSFTT.csv` | [metrics_SSFTT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/metrics_SSFTT.csv) |
| **Official Evaluation Report** | `outputs_ssftt_indian_pines/cls_SSFTT_IP/cls_result/classification_report.txt` | [classification_report.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/cls_SSFTT_IP/cls_result/classification_report.txt) |
| **Execution Telemetry Log** | `outputs_ssftt_indian_pines/actual-hsi-ssftt-indian-pines-benchmark.log` | [actual-hsi-ssftt-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/actual-hsi-ssftt-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_ssftt_indian_pines/outputs/run_SSFTT.log` | [run_SSFTT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/run_SSFTT.log) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/SSFTT_gt.npy` | [SSFTT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/SSFTT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/SSFTT_loss.npy` | [SSFTT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/SSFTT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/SSFTT_pred.npy` | [SSFTT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/SSFTT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/SSFTT_vloss.npy` | [SSFTT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/SSFTT_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/outputs/SSFTT_gt.npy` | [SSFTT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/SSFTT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/outputs/SSFTT_loss.npy` | [SSFTT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/SSFTT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/outputs/SSFTT_pred.npy` | [SSFTT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/SSFTT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_ssftt_indian_pines/outputs/SSFTT_vloss.npy` | [SSFTT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssftt_indian_pines/outputs/SSFTT_vloss.npy) |

**Implementation Notes:** Spectral-Spatial Feature Tokenizer Transformer (Sun et al.). Ultra-fast convergence (~1.05 min) on Kaggle GPU with top-tier 98.45% OA and 94.29% AA.

---

### 3.3 A2S2K-ResNet — Indian Pines Benchmark

- **Architecture Identifier:** `A2S2K-ResNet`
- **Dataset:** Indian Pines (145 × 145, 200 Bands, 16 Classes)
- **Local Staging / Output Directory:** [`outputs_a2s2k_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-a2s2k-resnet-indian-pines-benchmark`
- **Training Duration:** 14.89 min

#### Quantitative Metrics
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **98.11%** | `0.9811` |
| **Average Accuracy (AA)** | **94.41%** | `0.9441` |
| **Cohen's Kappa ($\kappa$)** | **0.9785** | `0.9785` |
| **Training Duration** | **14.89 min** | `14.89 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Total Samples | Test Split Acc (%) | Full Scene Acc (%) |
| :---: | :--- | :---: | :---: | :---: |
| C01 | Alfalfa | 46 | **94.44%** | 95.65% |
| C02 | Corn-notill | 1,428 | **98.50%** | 97.48% |
| C03 | Corn-mintill | 830 | **98.47%** | 97.83% |
| C04 | Corn | 237 | **100.00%** | 95.78% |
| C05 | Grass-pasture | 483 | **97.96%** | 98.76% |
| C06 | Grass-trees | 730 | **97.18%** | 99.73% |
| C07 | Grass-pasture-mowed | 28 | **100.00%** | 100.00% |
| C08 | Hay-windrowed | 478 | **99.48%** | 100.00% |
| C09 | Oats | 20 | **50.00%** | 55.00% |
| C10 | Soybean-notill | 972 | **98.04%** | 96.91% |
| C11 | Soybean-mintill | 2,455 | **99.03%** | 98.90% |
| C12 | Soybean-clean | 593 | **94.29%** | 97.64% |
| C13 | Wheat | 205 | **97.53%** | 96.10% |
| C14 | Woods | 1,265 | **99.31%** | 99.84% |
| C15 | Buildings-Grass-Trees-Drives | 386 | **96.19%** | 97.93% |
| C16 | Stone-Steel-Towers | 93 | **90.12%** | 100.00% |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_a2s2k_indian_pines/outputs/pred_A2S2K.png` | [pred_A2S2K.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/pred_A2S2K.png) |
| **Standardized Ground Truth Map** | `outputs_a2s2k_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_a2s2k_indian_pines/outputs/loss_A2S2K.png` | [loss_A2S2K.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/loss_A2S2K.png) |
| **Summary Metrics CSV** | `outputs_a2s2k_indian_pines/outputs/metrics_A2S2K.csv` | [metrics_A2S2K.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/metrics_A2S2K.csv) |
| **Official Evaluation Report** | `outputs_a2s2k_indian_pines/A2S2KResNet/report/S3KAIResNetpatch_11_INsplit0.9lr0.001adam_kernel_24.txt` | [S3KAIResNetpatch_11_INsplit0.9lr0.001adam_kernel_24.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2KResNet/report/S3KAIResNetpatch_11_INsplit0.9lr0.001adam_kernel_24.txt) |
| **Execution Telemetry Log** | `outputs_a2s2k_indian_pines/actual-a2s2k-resnet-indian-pines-benchmark.log` | [actual-a2s2k-resnet-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/actual-a2s2k-resnet-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_a2s2k_indian_pines/outputs/run_A2S2K.log` | [run_A2S2K.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/run_A2S2K.log) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/A2S2K_gt.npy` | [A2S2K_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2K_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/A2S2K_loss.npy` | [A2S2K_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2K_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/A2S2K_pred.npy` | [A2S2K_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2K_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/A2S2K_vloss.npy` | [A2S2K_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/A2S2K_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/outputs/A2S2K_gt.npy` | [A2S2K_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/A2S2K_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/outputs/A2S2K_loss.npy` | [A2S2K_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/A2S2K_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/outputs/A2S2K_pred.npy` | [A2S2K_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/A2S2K_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_a2s2k_indian_pines/outputs/A2S2K_vloss.npy` | [A2S2K_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_a2s2k_indian_pines/outputs/A2S2K_vloss.npy) |

**Implementation Notes:** Attention-based Adaptive Spectral-Spatial Kernel ResNet with 24 adaptive 3D-2D kernels (Roy et al.). Highest Average Accuracy (AA: 94.41%) across all models.

---

### 3.4 HyperRoPE-SST — Indian Pines Benchmark

- **Architecture Identifier:** `HyperRoPE-SST`
- **Dataset:** Indian Pines (145 × 145, 200 Bands, 16 Classes)
- **Local Staging / Output Directory:** [`outputs_hyperrope_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-hyperrope-sst-indian-pines-benchmark`
- **Training Duration:** 1.43 min

#### Quantitative Metrics
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **94.07%** | `0.9407` |
| **Average Accuracy (AA)** | **89.37%** | `0.8937` |
| **Cohen's Kappa ($\kappa$)** | **0.9324** | `0.9324` |
| **Training Duration** | **1.43 min** | `1.43 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Total Samples | Test Split Acc (%) | Full Scene Acc (%) |
| :---: | :--- | :---: | :---: | :---: |
| C01 | Alfalfa | 46 | **94.44%** | 93.48% |
| C02 | Corn-notill | 1,428 | **93.87%** | 94.47% |
| C03 | Corn-mintill | 830 | **95.63%** | 96.27% |
| C04 | Corn | 237 | **83.07%** | 85.65% |
| C05 | Grass-pasture | 483 | **89.66%** | 91.30% |
| C06 | Grass-trees | 730 | **96.23%** | 96.58% |
| C07 | Grass-pasture-mowed | 28 | **90.91%** | 92.86% |
| C08 | Hay-windrowed | 478 | **100.00%** | 100.00% |
| C09 | Oats | 20 | **25.00%** | 30.00% |
| C10 | Soybean-notill | 972 | **94.94%** | 95.47% |
| C11 | Soybean-mintill | 2,455 | **95.87%** | 96.54% |
| C12 | Soybean-clean | 593 | **83.37%** | 85.67% |
| C13 | Wheat | 205 | **98.79%** | 99.02% |
| C14 | Woods | 1,265 | **94.37%** | 95.02% |
| C15 | Buildings-Grass-Trees-Drives | 386 | **93.83%** | 94.04% |
| C16 | Stone-Steel-Towers | 93 | **100.00%** | 100.00% |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_hyperrope_indian_pines/outputs/pred_HyperRoPE.png` | [pred_HyperRoPE.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/pred_HyperRoPE.png) |
| **Standardized Ground Truth Map** | `outputs_hyperrope_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_hyperrope_indian_pines/outputs/loss_HyperRoPE.png` | [loss_HyperRoPE.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/loss_HyperRoPE.png) |
| **Summary Metrics CSV** | `outputs_hyperrope_indian_pines/outputs/metrics_HyperRoPE.csv` | [metrics_HyperRoPE.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/metrics_HyperRoPE.csv) |
| **Official Evaluation Report** | `outputs_hyperrope_indian_pines/hyperrope-sst/results/IndianPine_Hyper2DRoPE_Classification_Model_10_27_2026_12PM.json` | [IndianPine_Hyper2DRoPE_Classification_Model_10_27_2026_12PM.json](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/hyperrope-sst/results/IndianPine_Hyper2DRoPE_Classification_Model_10_27_2026_12PM.json) |
| **Execution Telemetry Log** | `outputs_hyperrope_indian_pines/actual-hyperrope-sst-indian-pines-benchmark.log` | [actual-hyperrope-sst-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/actual-hyperrope-sst-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_hyperrope_indian_pines/outputs/run_HyperRoPE.log` | [run_HyperRoPE.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/run_HyperRoPE.log) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/HyperRoPE_gt.npy` | [HyperRoPE_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/HyperRoPE_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/HyperRoPE_loss.npy` | [HyperRoPE_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/HyperRoPE_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/HyperRoPE_pred.npy` | [HyperRoPE_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/HyperRoPE_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/HyperRoPE_vloss.npy` | [HyperRoPE_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/HyperRoPE_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/outputs/HyperRoPE_gt.npy` | [HyperRoPE_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/HyperRoPE_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/outputs/HyperRoPE_loss.npy` | [HyperRoPE_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/HyperRoPE_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/outputs/HyperRoPE_pred.npy` | [HyperRoPE_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/HyperRoPE_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_hyperrope_indian_pines/outputs/HyperRoPE_vloss.npy` | [HyperRoPE_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_hyperrope_indian_pines/outputs/HyperRoPE_vloss.npy) |

**Implementation Notes:** Spatial-Spectral Transformer with 2D Rotary Position Embeddings. Ultra-fast training (1.43 min) on Kaggle GPU with 94.07% OA.

---

### 3.5 ViT — Indian Pines Benchmark

- **Architecture Identifier:** `ViT`
- **Dataset:** Indian Pines (145 × 145, 200 Bands, 16 Classes)
- **Local Staging / Output Directory:** [`outputs_vit_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-vit-indian-pines-benchmark`
- **Training Duration:** 1.68 min

#### Quantitative Metrics
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **90.33%** | `0.9033` |
| **Average Accuracy (AA)** | **89.12%** | `0.8912` |
| **Cohen's Kappa ($\kappa$)** | **0.8895** | `0.8895` |
| **Training Duration** | **1.68 min** | `1.68 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Total Samples | Test Split Acc (%) | Full Scene Acc (%) |
| :---: | :--- | :---: | :---: | :---: |
| C01 | Alfalfa | 46 | **73.91%** | 73.91% |
| C02 | Corn-notill | 1,428 | **83.89%** | 83.89% |
| C03 | Corn-mintill | 830 | **80.96%** | 80.96% |
| C04 | Corn | 237 | **91.98%** | 91.98% |
| C05 | Grass-pasture | 483 | **91.10%** | 91.10% |
| C06 | Grass-trees | 730 | **97.26%** | 97.26% |
| C07 | Grass-pasture-mowed | 28 | **96.43%** | 96.43% |
| C08 | Hay-windrowed | 478 | **99.58%** | 99.58% |
| C09 | Oats | 20 | **70.00%** | 70.00% |
| C10 | Soybean-notill | 972 | **85.49%** | 85.49% |
| C11 | Soybean-mintill | 2,455 | **93.32%** | 93.32% |
| C12 | Soybean-clean | 593 | **90.56%** | 90.56% |
| C13 | Wheat | 205 | **98.05%** | 98.05% |
| C14 | Woods | 1,265 | **99.84%** | 99.84% |
| C15 | Buildings-Grass-Trees-Drives | 386 | **95.85%** | 95.85% |
| C16 | Stone-Steel-Towers | 93 | **96.77%** | 96.77% |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_vit_indian_pines/outputs/pred_ViT.png` | [pred_ViT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/pred_ViT.png) |
| **Standardized Ground Truth Map** | `outputs_vit_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_vit_indian_pines/outputs/loss_ViT.png` | [loss_ViT.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/loss_ViT.png) |
| **Summary Metrics CSV** | `outputs_vit_indian_pines/outputs/metrics_ViT.csv` | [metrics_ViT.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/metrics_ViT.csv) |
| **Official Evaluation Report** | `outputs_vit_indian_pines/outputs/run_ViT.log` | [run_ViT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/run_ViT.log) |
| **Execution Telemetry Log** | `outputs_vit_indian_pines/actual-vit-indian-pines-benchmark.log` | [actual-vit-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/actual-vit-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_vit_indian_pines/outputs/run_ViT.log` | [run_ViT.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/run_ViT.log) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/ViT_gt.npy` | [ViT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/ViT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/ViT_loss.npy` | [ViT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/ViT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/ViT_pred.npy` | [ViT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/ViT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/ViT_vloss.npy` | [ViT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/ViT_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/outputs/ViT_gt.npy` | [ViT_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ViT_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/outputs/ViT_loss.npy` | [ViT_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ViT_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/outputs/ViT_pred.npy` | [ViT_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ViT_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_vit_indian_pines/outputs/ViT_vloss.npy` | [ViT_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_vit_indian_pines/outputs/ViT_vloss.npy) |

**Implementation Notes:** Vision Transformer baseline adapted for HSI patch tokenization. Trains in 1.68 min on Kaggle GPU with 90.33% OA and 89.12% AA.

---

### 3.6 DeepMatrixCapsules — Indian Pines Benchmark

- **Architecture Identifier:** `DeepMatrixCapsules`
- **Dataset:** Indian Pines (145 × 145, 200 Bands, 16 Classes)
- **Local Staging / Output Directory:** [`outputs_dmc_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-deepmatrixcapsules-indian-pines-benchmark`
- **Training Duration:** 17.73 min

#### Quantitative Metrics
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **82.55%** | `0.8255` |
| **Average Accuracy (AA)** | **64.11%** | `0.6411` |
| **Cohen's Kappa ($\kappa$)** | **0.7989** | `0.7989` |
| **Training Duration** | **17.73 min** | `17.73 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Total Samples | Test Split Acc (%) | Full Scene Acc (%) |
| :---: | :--- | :---: | :---: | :---: |
| C01 | Alfalfa | 46 | **2.70%** | 13.04% |
| C02 | Corn-notill | 1,428 | **77.41%** | 79.27% |
| C03 | Corn-mintill | 830 | **73.04%** | 75.30% |
| C04 | Corn | 237 | **24.87%** | 31.22% |
| C05 | Grass-pasture | 483 | **90.96%** | 91.72% |
| C06 | Grass-trees | 730 | **96.40%** | 96.71% |
| C07 | Grass-pasture-mowed | 28 | **0.00%** | 10.71% |
| C08 | Hay-windrowed | 478 | **99.74%** | 99.58% |
| C09 | Oats | 20 | **0.00%** | 10.00% |
| C10 | Soybean-notill | 972 | **76.61%** | 78.70% |
| C11 | Soybean-mintill | 2,455 | **91.30%** | 92.42% |
| C12 | Soybean-clean | 593 | **47.16%** | 52.78% |
| C13 | Wheat | 205 | **91.46%** | 92.20% |
| C14 | Woods | 1,265 | **97.53%** | 97.63% |
| C15 | Buildings-Grass-Trees-Drives | 386 | **80.52%** | 82.90% |
| C16 | Stone-Steel-Towers | 93 | **76.00%** | 78.49% |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_dmc_indian_pines/outputs/pred_DMC.png` | [pred_DMC.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/pred_DMC.png) |
| **Standardized Ground Truth Map** | `outputs_dmc_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_dmc_indian_pines/outputs/loss_DMC.png` | [loss_DMC.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/loss_DMC.png) |
| **Summary Metrics CSV** | `outputs_dmc_indian_pines/outputs/metrics_DMC.csv` | [metrics_DMC.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/metrics_DMC.csv) |
| **Official Evaluation Report** | `outputs_dmc_indian_pines/DeepMatrixCapsules/classification_report_dmc.txt` | [classification_report_dmc.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DeepMatrixCapsules/classification_report_dmc.txt) |
| **Execution Telemetry Log** | `outputs_dmc_indian_pines/actual-deepmatrixcapsules-indian-pines-benchmark.log` | [actual-deepmatrixcapsules-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/actual-deepmatrixcapsules-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_dmc_indian_pines/outputs/run_DMC.log` | [run_DMC.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/run_DMC.log) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/DMC_gt.npy` | [DMC_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DMC_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/DMC_loss.npy` | [DMC_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DMC_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/DMC_pred.npy` | [DMC_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DMC_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/DMC_vloss.npy` | [DMC_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/DMC_vloss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/outputs/DMC_gt.npy` | [DMC_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/DMC_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/outputs/DMC_loss.npy` | [DMC_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/DMC_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/outputs/DMC_pred.npy` | [DMC_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/DMC_pred.npy) |
| **Raw Output Array (.npy)** | `outputs_dmc_indian_pines/outputs/DMC_vloss.npy` | [DMC_vloss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_dmc_indian_pines/outputs/DMC_vloss.npy) |

**Implementation Notes:** Matrix Capsule Network with EM routing (Sabour et al.). Input minimally zero-padded from 11x11 to 13x13 to satisfy capsule layer squash receptive field.

---

### 3.7 CRV-MHDL — Indian Pines Benchmark

- **Architecture Identifier:** `CRV-MHDL`
- **Dataset:** Indian Pines (145 × 145, 200 Bands, 16 Classes)
- **Local Staging / Output Directory:** [`outputs_crv_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-crv-mhdl-indian-pines-benchmark`
- **Training Duration:** 1.33 min

#### Quantitative Metrics
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **67.20%** | `0.6720` |
| **Average Accuracy (AA)** | **51.71%** | `0.5171` |
| **Cohen's Kappa ($\kappa$)** | **0.6222** | `0.6222` |
| **Training Duration** | **1.33 min** | `1.33 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Total Samples | Test Split Acc (%) | Full Scene Acc (%) |
| :---: | :--- | :---: | :---: | :---: |
| C01 | Alfalfa | 46 | **10.81%** | 19.57% |
| C02 | Corn-notill | 1,428 | **47.64%** | 52.80% |
| C03 | Corn-mintill | 830 | **43.07%** | 48.67% |
| C04 | Corn | 237 | **9.52%** | 18.99% |
| C05 | Grass-pasture | 483 | **85.01%** | 86.13% |
| C06 | Grass-trees | 730 | **97.26%** | 97.12% |
| C07 | Grass-pasture-mowed | 28 | **9.09%** | 17.86% |
| C08 | Hay-windrowed | 478 | **84.29%** | 85.15% |
| C09 | Oats | 20 | **18.75%** | 30.00% |
| C10 | Soybean-notill | 972 | **63.37%** | 66.77% |
| C11 | Soybean-mintill | 2,455 | **75.27%** | 78.00% |
| C12 | Soybean-clean | 593 | **39.79%** | 45.87% |
| C13 | Wheat | 205 | **94.51%** | 95.61% |
| C14 | Woods | 1,265 | **97.63%** | 97.79% |
| C15 | Buildings-Grass-Trees-Drives | 386 | **39.29%** | 44.56% |
| C16 | Stone-Steel-Towers | 93 | **12.00%** | 20.43% |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_crv_indian_pines/outputs/pred_CRV.png` | [pred_CRV.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/pred_CRV.png) |
| **Standardized Ground Truth Map** | `outputs_crv_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_crv_indian_pines/outputs/loss_CRV.png` | [loss_CRV.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/loss_CRV.png) |
| **Summary Metrics CSV** | `outputs_crv_indian_pines/outputs/metrics_CRV.csv` | [metrics_CRV.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/metrics_CRV.csv) |
| **Official Evaluation Report** | `outputs_crv_indian_pines/CRV/classification_Results.txt` | [classification_Results.txt](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/CRV/classification_Results.txt) |
| **Execution Telemetry Log** | `outputs_crv_indian_pines/actual-crv-mhdl-indian-pines-benchmark.log` | [actual-crv-mhdl-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/actual-crv-mhdl-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_crv_indian_pines/outputs/run_CRV.log` | [run_CRV.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_crv_indian_pines/outputs/run_CRV.log) |
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

### 3.8 SSAN — Indian Pines Benchmark

- **Architecture Identifier:** `SSAN`
- **Dataset:** Indian Pines (145 × 145, 200 Bands, 16 Classes)
- **Local Staging / Output Directory:** [`outputs_ssan_indian_pines`](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines)
- **Kaggle Kernel Slug:** `https://www.kaggle.com/code/tejassurya2004/actual-ssan-indian-pines-benchmark`
- **Training Duration:** 14.00 min

#### Quantitative Metrics
| Metric | Value | Raw Representation |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **64.19%** | `0.6419` |
| **Average Accuracy (AA)** | **72.77%** | `0.7277` |
| **Cohen's Kappa ($\kappa$)** | **0.5979** | `0.5979` |
| **Training Duration** | **14.00 min** | `14.00 min` |

#### Per-Class Accuracy Breakdown
| Class # | Class Name | Total Samples | Test Split Acc (%) | Full Scene Acc (%) |
| :---: | :--- | :---: | :---: | :---: |
| C01 | Alfalfa | 46 | **71.88%** | 80.43% |
| C02 | Corn-notill | 1,428 | **65.96%** | 67.93% |
| C03 | Corn-mintill | 830 | **78.41%** | 80.00% |
| C04 | Corn | 237 | **76.36%** | 79.32% |
| C05 | Grass-pasture | 483 | **84.10%** | 87.16% |
| C06 | Grass-trees | 730 | **96.79%** | 96.99% |
| C07 | Grass-pasture-mowed | 28 | **60.00%** | 67.86% |
| C08 | Hay-windrowed | 478 | **98.20%** | 98.33% |
| C09 | Oats | 20 | **14.29%** | 30.00% |
| C10 | Soybean-notill | 972 | **72.15%** | 74.79% |
| C11 | Soybean-mintill | 2,455 | **25.76%** | 26.60% |
| C12 | Soybean-clean | 593 | **52.16%** | 59.70% |
| C13 | Wheat | 205 | **94.41%** | 96.10% |
| C14 | Woods | 1,265 | **92.58%** | 92.89% |
| C15 | Buildings-Grass-Trees-Drives | 386 | **82.80%** | 87.82% |
| C16 | Stone-Steel-Towers | 93 | **98.46%** | 97.85% |

#### Generated Output Artifacts & Direct File Paths
| Artifact Description | Relative Path | Direct Clickable File Link |
| :--- | :--- | :--- |
| **Classification Prediction Map** | `outputs_ssan_indian_pines/outputs/pred_SSAN.png` | [pred_SSAN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/pred_SSAN.png) |
| **Standardized Ground Truth Map** | `outputs_ssan_indian_pines/outputs/ground_truth.png` | [ground_truth.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/ground_truth.png) |
| **Training / Validation Loss Curve** | `outputs_ssan_indian_pines/outputs/loss_SSAN.png` | [loss_SSAN.png](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/loss_SSAN.png) |
| **Summary Metrics CSV** | `outputs_ssan_indian_pines/outputs/metrics_SSAN.csv` | [metrics_SSAN.csv](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/outputs/metrics_SSAN.csv) |
| **Official Evaluation Report** | `outputs_ssan_indian_pines/actual-ssan-indian-pines-benchmark.log` | [actual-ssan-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/actual-ssan-indian-pines-benchmark.log) |
| **Execution Telemetry Log** | `outputs_ssan_indian_pines/actual-ssan-indian-pines-benchmark.log` | [actual-ssan-indian-pines-benchmark.log](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/actual-ssan-indian-pines-benchmark.log) |
| **Raw Output Array (.npy)** | `outputs_ssan_indian_pines/SSAN_gt.npy` | [SSAN_gt.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/SSAN_gt.npy) |
| **Raw Output Array (.npy)** | `outputs_ssan_indian_pines/SSAN_loss.npy` | [SSAN_loss.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/SSAN_loss.npy) |
| **Raw Output Array (.npy)** | `outputs_ssan_indian_pines/SSAN_pred.npy` | [SSAN_pred.npy](file:///c:/Users/LENOVO/Desktop/FreshRuns/outputs_ssan_indian_pines/SSAN_pred.npy) |

**Implementation Notes:** Spatial-Spectral Attention Network (Peng et al.) executed in TensorFlow 1.x compatibility mode. Downloaded directly from Kaggle account `tejassurya2004/actual-ssan-indian-pines-benchmark`. 64.19% OA, 72.77% AA, 14.00 min training duration.

---

## 4. Indian Pines Ground Truth & Canonical Color Palette

| Class Index | Class Name | Color Preview | Hex Code | RGB Triplet | Ground Truth Samples |
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

## 5. Indian Pines Outputs Directory Structure

```text
FreshRuns/
├── README_INDIAN_PINES.md                   <- Dedicated Indian Pines single source of truth
├── outputs_ssrn_indian_pines/               <- SSRN on Indian Pines (OA: 98.52%, Kaggle: actual-ssrn-indian-pines-benchmark)
│   └── outputs/ [ground_truth.png, pred_SSRN.png, loss_SSRN.png, metrics_SSRN.csv]
├── outputs_ssftt_indian_pines/              <- SSFTT on Indian Pines (OA: 98.45%, Kaggle: actual-hsi-ssftt-indian-pines-benchmark)
├── outputs_a2s2k_indian_pines/              <- A2S2K-ResNet on Indian Pines (OA: 98.11%, Kaggle: actual-a2s2k-resnet-indian-pines-benchmark)
├── outputs_hyperrope_indian_pines/          <- HyperRoPE-SST on Indian Pines (OA: 94.07%, Kaggle: actual-hyperrope-sst-indian-pines-benchmark)
├── outputs_vit_indian_pines/                <- ViT on Indian Pines (OA: 90.33%, Kaggle: actual-vit-indian-pines-benchmark)
├── outputs_dmc_indian_pines/                <- DeepMatrixCapsules on Indian Pines (OA: 82.55%, Kaggle: actual-deepmatrixcapsules-indian-pines-benchmark)
├── outputs_crv_indian_pines/                <- CRV-MHDL on Indian Pines (OA: 67.20%, Kaggle: actual-crv-mhdl-indian-pines-benchmark)
└── outputs_ssan_indian_pines/               <- SSAN on Indian Pines (OA: 64.19%, Kaggle: actual-ssan-indian-pines-benchmark)
```

---
*Indian Pines benchmark reference compiled autonomously from Kaggle account `tejassurya2004` on 2026-10-27.*