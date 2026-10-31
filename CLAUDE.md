# CLAUDE.md - SGSCF-Net Academic Paper Co-Author Configuration

## Project Overview
This repository contains the complete experimental suite, codebase, and publication materials for:
**SGSCF-Net**: *Spectral-Guided Spatial Cross-Attention Fusion Network for Hyperspectral Image Classification*.

### Primary Sources of Truth (Strict Reference Required)
1. **Indian Pines Baseline Benchmark Master Truth**:
   `C:/Users/LENOVO/Desktop/MinorProject_Final/README_INDIAN_PINES.md`
   - Single verified source of truth for all 8 baseline models (SSRN, SSFTT, A2S2K-ResNet, HyperRoPE-SST, ViT, DeepMatrixCapsules, CRV-MHDL, SSAN) on Indian Pines (200 bands, 16 classes, 10% train / 10% val / 80% test, 200 epochs). Contains the full per-class accuracy comparison matrix (80% test split and full-scene), OA/AA/Kappa, training times, and artifact links. Supersedes the older `kaggle_results/BASELINE_MODELS_MASTER_README.html` (which lacked per-class breakdowns and carried stale metrics).
2. **Pavia University & Salinas Scene Baseline Benchmark Master Truth (16 Full GPU Runs)**:
   `C:/Users/LENOVO/Desktop/MinorProject_Final/README.md`
   - Exhaustive, verified Kaggle GPU benchmarks for all 8 deep learning baseline architectures evaluated across Pavia University (610×340, 103 bands, 9 classes, 42,776 labeled samples) and Salinas Scene (512×217, 204 bands, 16 classes, 54,129 labeled samples).
   - Standardized evaluation protocol: deterministic 10% Train / 10% Val / 80% Test split, 11×11 spatial patch, 200 epochs, batch size 32.
   - Comprehensive telemetry, per-class breakdowns, training durations, loss curves, and artifact links.
3. **Proposed Model Master Truth (120+ Runs)**:
   `C:/Users/LENOVO/Desktop/MinorProject_Final/MASTER_README_120+runs_SGSCF_Net.html`
   - Contains 60-run parameter grid search (Patch sizes: 9x9, 11x11, 13x13, 15x15, 17x17; Batch sizes: 32, 64, 128, 256) across Indian Pines, Pavia University, and Salinas Valley.
   - Contains 12 data scarcity stress-tests (1%, 2%, 5%, 10% training ratios).
   - Contains per-class breakdowns under extreme 1% data scarcity.
4. **Model Source Code**:
   `C:/Users/LENOVO/Desktop/MinorProject_Final/SGSCF-Net_Final_Implementation/SGSCF_Net.py`
   - Canonical implementation of SBAM, CBSA, ResBlock3D, ECA3D, SCFG, SpreadLoss, FocalLoss, and SGSCFNet architecture (~308K parameters).
5. **Target LaTeX Publication Workspace**:
   `C:/Users/LENOVO/Desktop/MinorProject_Final/ResearchPaperWrite/ACM_Template/`
   - `manuscript.tex` (ACM journal/conference double-column template)
   - `sample-base.bib` (BibTeX bibliography file)

---

### [CRITICAL MANDATORY REQUIREMENT] Automatic BibTeX & Citation Pipeline
Whenever introducing, citing, or discussing ANY model, baseline, algorithm, dataset, mathematical operator, or domain concept in the paper:
1. **Search & Retrieve Genuine BibTeX Entries**: Locate the authentic peer-reviewed paper (authors, title, journal/conference, volume, pages, year, publisher). NEVER hallucinate fake references or DOIs.
2. **Auto-Insert into `sample-base.bib`**: Immediately append or verify the corresponding `@article{...}` or `@inproceedings{...}` entry inside `ResearchPaperWrite/ACM_Template/sample-base.bib`. Never write `\cite{key}` in `manuscript.tex` without verifying that `key` physically exists in `sample-base.bib`.
3. **Key Synchronization & 3-Pass Build**: Ensure exact key match between `\cite{key}` and `@type{key, ...}`. Automatically run the 3-pass build (`pdflatex -> bibtex -> pdflatex*2` or `compile.bat`) to verify zero undefined `[?]` citation warnings in `manuscript.pdf`.

---

### 1. Stylistic Burstiness & Perplexity (Anti-AI Detector)
- **Banned AI Filler Words**: Never use words like: `delve`, `pivotal`, `testament`, `tapestry`, `revolutionize`, `beacon`, `game-changer`, `seamlessly`, `in today's digital era`, `furthermore`, `moreover`, `additionally` at the start of paragraphs.
- **Sentence Variation**: Vary sentence lengths dramatically. Alternate short empirical facts (8-12 words) with compound technical explanations (25-35 words).
- **Active Voice & Objective Tone**: Write in direct academic prose: "We formulate...", "Table 2 demonstrates...", "CBSA suppresses redundant spatial gradients when..." instead of passive, inflated descriptions.

---

### 2. Empirical Grounding & Master Leaderboards (Anti-Hallucination)
- **Every metric claim MUST cite exact values** from `README_INDIAN_PINES.md` (Indian Pines baselines), `README.md` (Pavia University & Salinas Scene), or `MASTER_README_120+runs_SGSCF_Net.html`.
- State exact metrics: Overall Accuracy (OA), Average Accuracy (AA), Kappa ($\kappa$), Training Time (minutes), and parameter counts.

#### Master Baseline Comparison Across All Three Benchmarks (10% Train / 10% Val / 80% Test, 200 Epochs)

| Model Architecture | Indian Pines OA (%) | Indian Pines AA (%) | Indian Pines $\kappa$ | PaviaU OA (%) | PaviaU AA (%) | PaviaU $\kappa$ | Salinas OA (%) | Salinas AA (%) | Salinas $\kappa$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SSRN** | **98.52%** | 79.99% | 0.9831 | **99.87%** | 99.71% | **0.9982** | 99.96% | 99.93% | 0.9995 |
| **A2S2K-ResNet** | 98.11% | 94.41% | 0.9785 | 99.83% | **99.79%** | 0.9978 | **100.00%** | **99.99%** | **1.0000** |
| **HyperRoPE-SST** | 94.07% | 89.37% | 0.9324 | 99.83% | 99.77% | 0.9978 | 99.85% | 99.89% | 0.9984 |
| **SSFTT** | 98.45% | 94.29% | 0.9823 | 99.72% | 99.40% | 0.9963 | 99.90% | 99.92% | 0.9988 |
| **DeepMatrixCapsules** | 82.55% | 64.11% | 0.7989 | 99.34% | 99.03% | 0.9912 | 99.90% | 99.88% | 0.9989 |
| **ViT** | 90.33% | 89.12% | 0.8895 | 99.05% | 98.90% | 0.9874 | 98.34% | 99.16% | 0.9816 |
| **CRV-MHDL** | 67.20% | 51.71% | 0.6222 | 98.40% | 97.17% | 0.9789 | 96.30% | 97.75% | 0.9587 |
| **SSAN** | 64.19% | 72.77% | 0.5979 | 76.82% | 77.90% | 0.6944 | 88.20% | 94.44% | 0.8688 |
| **SGSCF-Net (Proposed)** | 98.05% | **94.82%** | 0.9778 | **99.45%** | — | — | **99.71%** | — | — |

> **Indian Pines note:** SSRN posts the top OA (98.52%) but its AA collapses to 79.99% (0.00% on Alfalfa, Grass-pasture-mowed, Oats). SGSCF-Net holds the **best Indian Pines AA (94.82%)**. DMC and CRV-MHDL degrade sharply on this imbalanced scene (82.55% and 67.20% OA). Full per-class matrix in `README_INDIAN_PINES.md` §2.1.

#### Training Latency & Computational Efficiency Profile

| Model Architecture | Indian Pines Time (min) | Pavia University Time (min) | Salinas Scene Time (min) | Architectural Nature & Framework |
| :--- | :---: | :---: | :---: | :--- |
| **SSRN** | 6.29 min | 11.36 min | 30.98 min | 3D-2D Residual CNN (PyTorch) |
| **A2S2K-ResNet** | 14.89 min | 41.12 min | 10.34 min | Adaptive Spectral-Spatial Kernel ResNet (PyTorch) |
| **HyperRoPE-SST** | 1.43 min | 5.62 min | 5.92 min | 2D Rotary Position Embedding Transformer (PyTorch) |
| **SSFTT** | 1.05 min | 4.44 min | 5.26 min | Feature Tokenizer Transformer (PyTorch) |
| **DeepMatrixCapsules** | 17.73 min | 73.42 min | 94.64 min | Capsule Network with EM routing (PyTorch, padded 13×13) |
| **ViT** | 1.68 min | 6.64 min | 8.94 min | Standard Vision Transformer (PyTorch) |
| **CRV-MHDL** | 1.33 min | 2.89 min | 3.55 min | Cascaded Residual Vision Multi-Head (Keras 3) |
| **SSAN** | 14.00 min | 8.70 min | 19.37 min | Spatial-Spectral Attention Network (TensorFlow) |
| **SGSCF-Net (Proposed)** | 8.80 min | — | — | Lightweight Dual-Branch SBAM+CBSA+SCFG (~308K params) |

*\*Note: All Indian Pines baselines were trained from scratch under the identical 10/10/80 protocol (DMC EM routing = 17.73 min, the costliest IP run). Earlier CLAUDE.md revisions listed a stale DMC warm-start time (0.39 min) and transformer time (37.74 min); these are superseded by `README_INDIAN_PINES.md`.*

#### Critical Baseline Implementation Specifications (from `README.md`):
- **DeepMatrixCapsules (DMC) Window Constraint:** Capsule squash layers require minimum 13×13 spatial dimension (11×11 crashes). The 11×11 spatial neighbourhood was zero-padded to 13×13 with a 1-pixel border to maintain uniform patch extraction without altering internal capsule architecture.
- **CRV-MHDL Framework & Keras 3 Migration:** Migrated from legacy Keras to Keras 3 (replaced deprecated `np_utils`, updated `Adam` learning rate schedule, saved `.keras` checkpoints). Salinas output head corrected from author's hardcoded 13 classes to the actual 16 classes.
- **A2S2K-ResNet Configuration:** Early-stopping patience set to 50 epochs; loss curve logs the final batch loss per epoch per author's convention. Replaced unavailable DiffGrad with standard PyTorch Adam.
- **Strict Validation Separation:** All baselines evaluated under standardized 10% Train / 10% Val / 80% Test splits with genuine validation curves (`<MODEL>_vloss.npy`), preventing data leakage.
- **SSAN Attention Bottleneck:** SSAN exhibits severe accuracy drops on minor/boundary classes (e.g., Gravel 17.96% on PaviaU, Grapes untrained 57.89% on Salinas) due to unconstrained softmax spatial attention collapse under class imbalance.

---

### 3. Mathematical Exactness
- All equations in Section III must map 1:1 to tensor operations in `SGSCF_Net.py`:
  - **SBAM**: Spatial pooling $\to$ FC bottleneck $\to$ Sigmoid band gate $\to$ Hadamard product on input $X \in \mathbb{R}^{B \times 1 \times P \times P \times C}$.
  - **CBSA**: Dual branch entry ($1\times 1\times 7$ vs $3\times 3\times 7$), mean pooling over spatial axes, Conv1D gate $\alpha$, complementary fusion $\alpha \odot a + (1-\alpha) \odot b$.
  - **SCFG**: Channel descriptor $f_{\text{ch}} \in \mathbb{R}^{B \times C}$, spatial descriptor $f_{\text{sp}} \in \mathbb{R}^{B \times N}$, outer product covariance matrix $G = f_{\text{ch}} \otimes f_{\text{sp}} \in \mathbb{R}^{B \times C \times N}$, Frobenius normalization, Linear projection to channel gate.
  - **Loss**: $\mathcal{L} = 0.7 \mathcal{L}_{\text{Spread}} + 0.3 \mathcal{L}_{\text{Focal}}$.

---

## Step-by-Step Writing Workflow

1. `/write-methodology`: Read `SGSCF_Net.py`, construct Section III with complete LaTeX formulas, tensor dimension tracking, and architectural figures descriptions.
2. `/write-experiments`: Read `README_INDIAN_PINES.md` (Indian Pines baselines), `README.md` (Pavia University & Salinas Scene, 16 runs), and `MASTER_README_120+runs_SGSCF_Net.html`. Construct Section IV with booktabs tables for baseline comparisons across Indian Pines, Pavia University, and Salinas Scene.
3. `/write-ablation`: Construct Section V analyzing the 60-run patch/batch grid search and the 1%-10% data scarcity degradation trends.
4. `/write-related-work`: Construct Section II reviewing 3D CNNs, Attention Mechanisms, Capsule Networks, and Transformers in HSI, identifying the spectral-spatial entanglement bottleneck.
5. `/write-introduction`: Construct Section I highlighting motivation, research gaps, and a bulleted list of 4 concrete contributions (including cross-dataset validation across 24 baseline runs).
6. `/write-abstract`: Draft a compact 200-word empirical abstract and conclusion.



---

## [STRICT MANDATORY RULE] Automated Git Versioning & Remote Push Protocol

To guarantee zero data loss and enable instant rollback to any prior state:
1. **Pre-Change Verification**:
   - Before executing any script or tool that modifies `manuscript.tex`, figures, tables, scripts, or model implementation files, ensure git working tree is known and clean.
   - NEVER use greedy regex replacements (e.g. `re.DOTALL` across broad block tags like `\begin{figure*}`). Always use exact target string replacements or tightly bounded patterns.
   - NEVER write LaTeX code containing dollar math (`$...$`) via PowerShell inline commands (`powershell -Command "python -c \"...\""`), as PowerShell interpolates `$` symbols into empty strings and breaks equations. Always use file-writing tools (`write_file`) or standalone python scripts.
2. **Immediate Post-Change Commit & Push**:
   - Immediately after completing ANY successful modification, table insertion, figure update, text refinement, or LaTeX build:
     1. Stage all relevant modified/added files: `git add .` (respecting `.gitignore`).
     2. Commit with an informative semantic message: `git commit -m "feat/fix/refactor: <clear description>"`.
     3. Push immediately to GitHub: `git push origin main`.
3. **Continuous Safety Guarantee**:
   - Every state of the document and codebase is permanently preserved in the remote GitHub repository (`https://github.com/codeforlifeee/paper`).
   - If any unexpected error or discrepancy occurs, the exact pre-change version can be restored immediately using `git checkout` or `git restore`.
