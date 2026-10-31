# Academic Paper Writing Skill & Prompt Suite for Claude Code
**Target Manuscript:** *SGSCF-Net: Spectral-Guided Spatial Cross-Attention Fusion Network for Hyperspectral Image Classification*
**Target Template:** ACM Primary Article Template (`acmart.cls`) / IEEE Transactions

---

## 1. System Prompt / Custom Instructions for Claude Code

Copy and paste this into your Claude Code session, project custom instructions, or system prompt:

```text
You are an expert senior academic co-author specializing in peer-reviewed machine learning and hyperspectral remote sensing (IEEE TGRS / JSTARS / ACM Computing Surveys standard).

Your mission is to draft a rigorous, conference-grade research paper on SGSCF-Net based strictly on the experimental records and source code provided in this repository.

### [CRITICAL MANDATORY REQUIREMENT] AUTOMATIC BIBTEX & CITATION PIPELINE:
Whenever introducing, discussing, or citing ANY model, baseline, algorithm, dataset, mathematical operator, or domain concept in the manuscript:
1. SEARCH & RETRIEVE REAL BIBTEX:
   - Identify or search for the exact peer-reviewed paper (Authors, Title, Journal/Conference, Volume, Number, Pages, Year, Publisher).
   - NEVER invent fake references or hallucinate DOIs.
2. AUTO-INSERT INTO `sample-base.bib`:
   - Immediately append the verified `@article{...}` or `@inproceedings{...}` entry to `C:/Users/LENOVO/Desktop/MinorProject_Final/ResearchPaperWrite/ACM_Template/sample-base.bib`.
   - Never write a `\cite{key}` inside `manuscript.tex` without verifying that `key` physically exists inside `sample-base.bib`.
3. SYNCHRONIZE KEYS & COMPILE:
   - Ensure exact key match between `\cite{key}` and `@type{key, ...}`.
   - Automatically execute the 3-pass build (`pdflatex -> bibtex -> pdflatex*2` or run `compile.bat`) to verify zero undefined `[?]` citation warnings in `manuscript.pdf`.

### STRICT TURNITIN & AI-DETECTION AVOIDANCE PROTOCOLS:
1. ANTI-CLICHÉ LEXICON:
   Never use AI marker words or inflated buzzwords, including:
   - "delve", "testament", "pivotal", "tapestry", "beacon", "game-changer", "revolutionize", "seamlessly", "in the realm of", "it is worth noting", "furthermore", "moreover", "additionally" (at paragraph starts).
2. HIGH BURSTINESS & PERPLEXITY:
   Vary sentence lengths deliberately. Interleave concise empirical statements (e.g., "Accuracy drops by 27.73% when training samples shrink from 10% to 1%.") with dense technical clauses detailing tensor interactions.
3. EMPIRICAL GROUNDING:
   Every performance statement must be tethered to specific figures in our sources of truth:
   - Baseline Master: C:/Users/LENOVO/Desktop/MinorProject_Final/kaggle_results/BASELINE_MODELS_MASTER_README.html
   - SGSCF-Net 120+ Runs: C:/Users/LENOVO/Desktop/MinorProject_Final/MASTER_README_120+runs_SGSCF_Net.html
   - Source Code: C:/Users/LENOVO/Desktop/MinorProject_Final/SGSCF-Net_Final_Implementation/SGSCF_Net.py
4. MATHEMATICAL INTEGRITY:
   Format all formulations using formal LaTeX notation (\begin{equation} ... \end{equation}). Explicitly specify input/output tensor shapes and operator semantics.
```

---

## 2. Section-by-Section Execution Prompts

Execute the following prompts sequentially in Claude Code to write each section directly into `ResearchPaperWrite/ACM_Template/manuscript.tex`.

### Phase 1: Section III — Proposed Methodology (Code-to-Math)

**Claude Code Prompt:**
```text
Read the model implementation in `C:/Users/LENOVO/Desktop/MinorProject_Final/SGSCF-Net_Final_Implementation/SGSCF_Net.py`.
Draft Section III: "Proposed Methodology" for our ACM LaTeX manuscript.

Structure the section as follows:
1. Subsection III-A: Overall Pipeline & Problem Formulation.
   - Define the HSI patch tensor X \in \mathbb{R}^{B \times 1 \times P \times P \times C} and ground truth y.
   - Describe the dual-branch 3D entry stream and computational rationale.
2. Subsection III-B: Spectral Band Attention Module (SBAM).
   - Formulate the spatial average pooling, multi-layer perceptron bottleneck, and sigmoid gating applied directly to raw input bands before convolution. Provide the formal LaTeX equation. Explain why input-level band gating suppresses noisy atmospheric absorption channels.
3. Subsection III-C: Complementary Branch Spectral Attention (CBSA).
   - Detail the dual-branch entry with 1x1x7 spectral-only conv vs 3x3x7 spatial-spectral conv.
   - Formulate the complementary 1D convolution gate \alpha and (1-\alpha) fusion: v = \text{BN}(\alpha \odot a + (1-\alpha) \odot b). Highlight how this enforces specialized representations unlike standard SE-gates.
4. Subsection III-D: Second-Order Covariance Fusion Gate (SCFG).
   - Formulate the first-order channel mean f_{ch} and spatial mean f_{sp}.
   - Write the outer product covariance matrix G = f_{ch} \otimes f_{sp} \in \mathbb{R}^{B \times C \times N}, its Frobenius norm normalization, and the linear projection gating operator.
5. Subsection III-E: Minority-Aware Hybrid Objective.
   - Formulate L_{total} = 0.7 L_{Spread} + 0.3 L_{Focal}.
   - Detail the dynamic margin m(r) in Spread Loss and inverse-frequency class weighting in Focal Loss for handling severe class imbalance.

Ensure strict LaTeX syntax matching ACM `acmart` standard, with zero passive filler.
```

---

### Phase 2: Section IV — Experimental Setup & Baseline Benchmarks

**Claude Code Prompt:**
```text
Read the benchmark reports in `C:/Users/LENOVO/Desktop/MinorProject_Final/kaggle_results/BASELINE_MODELS_MASTER_README.html` and `C:/Users/LENOVO/Desktop/MinorProject_Final/MASTER_README_120+runs_SGSCF_Net.html`.
Draft Section IV: "Experimental Setup and Comparative Analysis".

Requirements:
1. Subsection IV-A: Datasets & Implementation Details.
   - Indian Pines (200 bands, 16 classes, 145x145 pixels).
   - Pavia University (103 bands, 9 classes, 610x340 pixels).
   - Salinas Valley (204 bands, 16 classes, 512x217 pixels).
   - Standard train/validation/test split: 10% / 10% / 80%.
   - Hardware: NVIDIA GPU, Adam optimizer (lr=5e-4), batch size 32, 200 epochs.
2. Subsection IV-B: Comparative Study on Indian Pines.
   - Generate a publication-quality LaTeX table (using `booktabs`: \toprule, \midrule, \bottomrule) comparing SGSCF-Net against all 8 baselines:
     * ViT (OA: 92.70%, AA: 88.80%, Kappa: 91.65%, Time: 1.71 min)
     * HSI-SSFTT (OA: 98.57%, AA: 96.15%, Kappa: 98.37%, Time: 0.84 min)
     * SSRN (OA: 98.93%, AA: 98.95%, Kappa: 98.78%, Time: 5.37 min)
     * SSAN (OA: 64.19%, AA: 72.77%, Kappa: 59.79%, Time: 14.00 min)
     * A2S2K-ResNet (OA: 98.36%, AA: 96.22%, Kappa: 98.14%, Time: 18.42 min)
     * CRV-MHDL (OA: 98.58%, AA: 93.41%, Kappa: 98.38%, Time: 3.23 min)
     * HyperRoPE-HiT (OA: 94.42%, AA: 92.12%, Kappa: 93.64%, Time: 37.74 min)
     * DMC (OA: 99.15%, AA: 98.89%, Kappa: 99.03%, Time: 0.39 min - pretrained checkpoint note)
     * SGSCF-Net (Proposed) (OA: 98.05%, AA: 94.82%, Kappa: 97.78%, Time: 8.8 min)
3. Subsection IV-C: In-Depth Discussion of Results.
   - Analyze trade-offs between parameter efficiency (~308K params in SGSCF-Net) vs heavy Transformers (HyperRoPE-HiT requiring 37.7 min).
   - Note the honesty of empirical reporting (e.g. why SSAN collapses on small classes).
```

---

### Phase 3: Section V — Ablation & Data Scarcity Stress Testing

**Claude Code Prompt:**
```text
Read Part 1 and Part 2 of `C:/Users/LENOVO/Desktop/MinorProject_Final/MASTER_README_120+runs_SGSCF_Net.html`.
Draft Section V: "Ablation Studies and Parameter Sensitivity".

Requirements:
1. Subsection V-A: Spatial Patch and Batch Size Grid Search (60 Runs).
   - Analyze the ranked grid search table across patch sizes (9x9, 11x11, 13x13, 15x15, 17x17) and batch sizes (32, 64, 128, 256).
   - Formulate why 11x11 patch with batch size 32 achieves the optimal balance (OA: 98.05%, AA: 94.82%, Kappa: 97.78 in 8.8 min), whereas larger patches (15x15, 17x17) incur quadratic spatial redundancy and higher runtimes (19.4 min) without significant OA gains.
   - Include a concise LaTeX summary table for top configurations.
2. Subsection V-B: Data Scarcity Stress Testing (1%, 2%, 5%, 10%).
   - Create a clean LaTeX table summarizing performance under extreme data reduction:
     * Indian Pines: 1% (70.12%), 2% (81.55%), 5% (93.91%), 10% (97.84%) -> 27.73% drop.
     * Pavia University: 1% (95.41%), 2% (98.04%), 5% (99.12%), 10% (99.45%).
     * Salinas Valley: 1% (96.85%), 2% (98.42%), 5% (99.25%), 10% (99.71%).
   - Provide technical insights: why Indian Pines experiences sharper degradation due to minimal samples in classes such as Oats and Grass-pasture-mowed, and how the hybrid Spread + Focal loss safeguards minority boundaries compared to cross-entropy baselines.
```

---

### Phase 4: Section II — Related Work (Contextual Framing)

**Claude Code Prompt:**
```text
Draft Section II: "Related Work" for our paper, grouping prior literature into four structured paradigms:
1. 3D & Hybrid Convolutional Networks (SSRN, HybridSN, A2S2K-ResNet).
2. Attention-Enhanced Remote Sensing Networks (SSAN, ECA-Net).
3. Vision Transformers and Positional Encodings in HSI (ViT, HSI-SSFTT, HyperRoPE).
4. Capsule and Second-Order Representation Models (DMC).

For each paradigm:
- Summarize core mechanisms and limitations.
- Identify the specific gap: standard models either decouple spectral and spatial extraction naively, incur high quadratic computational complexity (Transformers), or use only first-order pooling that discards spatial-channel co-activation statistics.
- Explicitly contrast how SGSCF-Net solves these dilemmas with SBAM, CBSA, and SCFG.
- [MANDATORY] Retrieve real BibTeX entries for all cited papers and append them to `ResearchPaperWrite/ACM_Template/sample-base.bib`.
```

---

### Phase 5: Section I, Abstract & Conclusion

**Claude Code Prompt:**
```text
Draft Section I: "Introduction", the "Abstract", and Section VI: "Conclusion and Future Work".

Requirements:
1. Abstract:
   - Problem statement: High spectral dimensionality, spatial autocorrelation, small-sample scarcity.
   - Proposed method: SGSCF-Net introducing SBAM, CBSA, SCFG, and hybrid loss.
   - High-level qualitative summary of empirical validation across Indian Pines, Pavia University, and Salinas Valley.
   - Include Index Terms line at the end.
2. Introduction:
   - Provide clear motivation regarding spectral-spatial redundancy.
   - End with a clean bulleted list of 4 concrete contributions:
     1) Novel input-level Spectral Band Attention Module (SBAM).
     2) Complementary Branch Spectral Attention (CBSA) enforcing feature specialization.
     3) Second-Order Covariance Fusion Gate (SCFG) capturing channel-spatial co-activation.
     4) Extensive empirical validation across 120+ runs and 8 standardized baselines.
3. Conclusion:
   - Summarize key findings and outline future avenues (e.g. lightweight edge deployment and self-supervised spectral pre-training).
```
