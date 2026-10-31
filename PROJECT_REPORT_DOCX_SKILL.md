# Skill: Academic Project Report DOCX Generator (`academic-docx-project-report-generator`)
**Target Document Standard:** University / NIT Raipur Project Report (B.Tech / M.Tech / Minor / Major)
**Reference Prototype:** `C:\Users\LENOVO\Downloads\organized.pdf` (National Institute of Technology Raipur, Dept. of Information Technology)
**Output Format:** Fully Formatted Microsoft Word Document (`.docx`) via `python-docx`
**Repository Context:** `MinorProject_Final` (SGSCF-Net Hyperspectral Image Classification)

---

## 1. Skill Metadata & Operational Capabilities

This skill equips Claude (or Claude Code / Antigravity) to act as a **Lead Academic Author & Technical Document Architect**. It automates the end-to-end production of publication-grade academic project reports formatted as `.docx` files that match the institutional standards of NIT Raipur.

### Key Capabilities:
1. **Institutional Front-Matter Architecture**:
   - Generates the official NIT Raipur title page with university emblem (`nit_raipur_logo.jpeg`), project title, side-by-side student and supervisor metadata, branch, degree, and semester.
   - Generates Certificate of Approval, Candidate Declaration, Acknowledgements, and structured Abstract.
2. **Dynamic Navigation & Indexing**:
   - Generates Contents (TOC) with dotted tab leaders (`. . . . .`) and page numbers, List of Figures, and List of Tables.
3. **Rigorous Chapter & Section Hierarchy**:
   - Produces an 8-chapter body matching the depth and rigor of `organized.pdf` (Introduction, Related Work, System Model, Algorithmic Design, Experimental Setup, Results & Ablation, Discussion & Interpretability, Limitations & Future Work).
4. **Data-Grounded Tables & Booktabs Styling**:
   - Formats academic tables with custom Booktabs borders (thick top/bottom rules, mid-header rule, zero vertical lines), shaded header bands, alternating row highlights, and strict statistical reporting ($\text{mean} \pm \text{std}$, $p$-values, parameter counts, FLOPs, runtimes).
5. **High-Resolution Figures & Graphics Pipeline**:
   - Integrates existing repository assets (classification maps, ground truths, architecture diagrams, loss curves).
   - Generates high-DPI (300 DPI) Matplotlib/Seaborn charts for accuracy comparisons, training convergence, grid search heatmaps, and data scarcity stress testing.
   - Pairs every graphic with an academic multi-line caption (`Figure X: [Title]. [Detailed narrative describing methodology, observations, and implications]`).
6. **Mathematical Formulations & Equations**:
   - Typesets formal mathematical equations using clean Unicode/LaTeX-style representation, formatted in borderless 2-column tables with right-aligned equation numbering `(1)`, `(2)`, etc.
7. **BibTeX-Synchronized IEEE Citation Engine**:
   - Parses authentic `.bib` databases (e.g., `sample-base.bib`), maps citations sequentially as `[1]`, `[2]`, ..., and builds a complete, numbered IEEE bibliography with full volume, issue, page, month, and year data.
8. **Anti-AI Detection & Academic Burstiness**:
   - Enforces an anti-cliché lexicon (bans words like *delve, testament, pivotal, game-changer, tapestry, beacon, revolutionize*).
   - Enforces sentence variation (burstiness) and empirical grounding directly on experimental truths.

---

## 2. Document Styling & Layout Specification (Based on `organized.pdf`)

| Parameter | Specification | Implementation in `python-docx` |
| :--- | :--- | :--- |
| **Page Size** | A4 ($210 \times 297\text{ mm}$ / $8.27 \times 11.69\text{ in}$) | `section.page_width = Inches(8.27)`, `section.page_height = Inches(11.69)` |
| **Margins** | Exactly $1.0\text{ inch}$ ($72\text{ pt}$ / $25.4\text{ mm}$) all sides | `top_margin = bottom_margin = left_margin = right_margin = Inches(1.0)` |
| **Body Font** | Times New Roman, 12 pt, Regular | Font name: `'Times New Roman'`, size: `Pt(12)` |
| **Line Spacing** | 1.15 to 1.3 line pitch, 6 pt after paragraph | `line_spacing = 1.15`, `space_after = Pt(6)` |
| **Text Alignment** | Justified | `alignment = WD_ALIGN_PARAGRAPH.JUSTIFY` |
| **Cover Title** | 18–19 pt Bold, Centered | `size = Pt(18)`, `bold = True`, `alignment = CENTER` |
| **Chapter Headings (H1)** | 16–18 pt Bold, 12 pt before, 6 pt after | `size = Pt(16)`, `bold = True`, `space_before = Pt(12)`, `space_after = Pt(6)` |
| **Section Headings (H2)** | 13–14 pt Bold, 10 pt before, 4 pt after | `size = Pt(13.5)`, `bold = True`, `space_before = Pt(10)`, `space_after = Pt(4)` |
| **Subsection Headings (H3)**| 12 pt Bold / Semi-bold Italic, 6 pt before | `size = Pt(12)`, `bold = True`, `italic = True`, `space_before = Pt(6)` |
| **Captions** | 10.5 pt, Bold Prefix (`Figure 1:`, `Table 1:`) | `size = Pt(10.5)`, label bold, paragraph centered |
| **Headers & Footers** | Running head on top; Centered Arabic Page Number | Center-aligned footer with Word `PAGE` dynamic XML field |
| **Table Layout** | Booktabs style: 1.5 pt top/bottom, 0.75 pt header | XML `<w:tcBorders>` and `<w:shd w:fill="F1F5F9"/>` |
| **Equation Layout** | Centered math + Right-aligned `(X)` | 1-row, 2-column borderless table ($85\%$ width + $15\%$ width) |

---

## 3. Structural Report Anatomy

The generated report must strictly mirror the 23-page structure of `organized.pdf`:

```text
├── COVER PAGE (Page 1)
│   ├── University: National Institute of Technology Raipur (18 pt Bold)
│   ├── Department: Department of Information Technology (16 pt)
│   ├── Institution Emblem: nit_raipur_logo.jpeg (Centered, ~1.9 in width)
│   ├── Type: Project Report on (16 pt)
│   ├── Project Title: [Title in 18 pt Bold]
│   └── Metadata Grid: Submitted by (Students + Roll Nos) | Guided by (Supervisor), Branch, Course, Semester
│
├── FRONT MATTER
│   ├── Certificate of Approval (Supervisor & HOD signatures)
│   ├── Candidate Declaration (Plagiarism & originality statement)
│   ├── Acknowledgements (Mentor, Department, Technical resources)
│   ├── Abstract & Keywords (~300 words with quantitative results)
│   ├── Table of Contents (Contents with dotted leaders and page numbers)
│   ├── List of Figures (Figure numbers, captions, page numbers)
│   └── List of Tables (Table numbers, titles, page numbers)
│
├── CHAPTER 1: INTRODUCTION
│   ├── 1.1 Remote Sensing & Hyperspectral Imaging Challenges
│   ├── 1.2 Spectral-Spatial Redundancy & Small-Sample Scarcity
│   ├── 1.3 Problem Statement & Research Objectives
│   └── 1.4 Outline of the Report
│
├── CHAPTER 2: LITERATURE REVIEW & RELATED WORK
│   ├── 2.1 Survey of Existing Paradigms (3D CNNs, Attention, Transformers, Covariance/Capsule)
│   ├── 2.2 Table 1: Comprehensive Comparative Analysis of Prior Approaches
│   └── 2.3 Critical Research Gaps & Proposed Solution Nexus
│
├── CHAPTER 3: SYSTEM MODEL & PROBLEM FORMULATION
│   ├── 3.1 Mathematical Formulation of HSI Data Cube & Patch Tensors
│   ├── 3.2 Dual-Branch 3D Feature Space & Operator Notation
│   └── 3.3 Theoretical Problem Definition & Optimization Objectives
│
├── CHAPTER 4: SGSCF-NET ARCHITECTURE & MODULE DESIGN
│   ├── 4.1 System Overview & High-Level Block Flowchart (Figure 1)
│   ├── 4.2 Spectral Band Attention Module (SBAM) (Formulation & Equations)
│   ├── 4.3 Complementary Branch Spectral Attention (CBSA) (Formulation & Equations)
│   ├── 4.4 Second-Order Covariance Fusion Gate (SCFG) (Tensor Outer Product Equations)
│   ├── 4.5 Minority-Aware Hybrid Loss Formulation (Spread Loss + Focal Loss)
│   └── 4.6 Architectural Parameter & Hyperparameter Specifications (Table 2)
│
├── CHAPTER 5: EXPERIMENTAL SETUP & METHODOLOGY
│   ├── 5.1 Benchmark Hyperspectral Datasets (Indian Pines, Pavia University, Salinas Valley)
│   ├── 5.2 Baseline Benchmark Models (ViT, HSI-SSFTT, SSRN, SSAN, A2S2K-ResNet, CRV-MHDL, HyperRoPE-HiT, DMC)
│   ├── 5.3 Implementation Hardware, Software & Training Protocol
│   └── 5.4 Quantitative Evaluation Metrics (OA, AA, Kappa, Runtime, Parameter Efficiency)
│
├── CHAPTER 6: RESULTS, COMPARATIVE ANALYSIS & ABLATION STUDIES
│   ├── 6.1 Benchmark Performance Comparison on Indian Pines (Table 3 & Figures 2, 3)
│   ├── 6.2 Spatial Patch & Batch Size Sensitivity Analysis (60-Run Grid Search) (Figures 4, 5)
│   ├── 6.3 Data Scarcity Stress Testing (1%, 2%, 5%, 10% Training Splits) (Table 4 & Figure 6)
│   └── 6.4 Qualitative Classification Maps & Per-Class Minority Accuracy Analysis
│
├── CHAPTER 7: IN-DEPTH DISCUSSION & INTERPRETABILITY
│   ├── 7.1 Interpreting the Learned Covariance Gating Dynamic
│   ├── 7.2 Failure Modes of Prior Baselines & Architectural Lessons
│   └── 7.3 Computational Complexity vs. Parameter Efficiency Trade-offs
│
├── CHAPTER 8: LIMITATIONS & FUTURE WORK
│   ├── 8.1 Current Theoretical & Practical Constraints
│   ├── 8.2 Real-Time Edge & Airborne Deployment Avenues
│   └── 8.3 Self-Supervised Spectral Foundation Models
│
└── REFERENCES
    └── Complete IEEE-Standard Numbered Citations [1]–[X] mapped to sample-base.bib
```

---

## 4. Reusable Python-Docx Architectural Engine Patterns

When executing this skill, Claude writes and runs an automated Python script (e.g., `generate_project_report_docx.py`) utilizing the following tested primitives:

### A. Document Setup & Page Geometry
```python
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

doc = docx.Document()
for section in doc.sections:
    section.page_width = Inches(8.27)    # A4
    section.page_height = Inches(11.69)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
```

### B. Dynamic Page Numbering in Footers
```python
def add_footer_page_number(run):
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'PAGE')
    run._r.append(fldSimple)

for section in doc.sections:
    footer = section.footer
    f_p = footer.paragraphs[0]
    f_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    f_run = f_p.add_run()
    f_run.font.name = 'Times New Roman'
    f_run.font.size = Pt(10)
    f_run.font.color.rgb = RGBColor(100, 100, 100)
    add_footer_page_number(f_run)
```

### C. Booktabs Academic Table Styler
```python
def style_booktabs_table(table, col_widths=None, header_bg="F1F5F9"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Remove default grid borders, apply top, bottom, and header underline
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="1A202C"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="1A202C"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    # Style header row
    header_tr = table.rows[0]._tr
    header_borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="1A202C"/>\n'
        f'</w:tcBorders>'
    )
    for cell in table.rows[0].cells:
        cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{header_bg}"/>'))
        cell._tc.get_or_add_tcPr().append(header_borders)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10)
                r.font.bold = True
                r.font.color.rgb = RGBColor(20, 20, 20)

    # Style body cells
    for row_idx, row in enumerate(table.rows[1:], start=1):
        bg_color = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for cell in row.cells:
            cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>'))
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(9.5)

    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
```

### D. Numbered Equation Helper
```python
def add_numbered_equation(doc, equation_text, eq_number_str):
    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Make borders completely invisible
    tblPr = tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)
    
    # Left cell: equation
    c0 = tbl.cell(0, 0)
    c0.width = Inches(5.4)
    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.paragraph_format.space_before = Pt(4)
    p0.paragraph_format.space_after = Pt(4)
    r0 = p0.add_run(equation_text)
    r0.font.name = 'Times New Roman'
    r0.font.size = Pt(11)
    r0.font.italic = True
    
    # Right cell: equation number
    c1 = tbl.cell(0, 1)
    c1.width = Inches(0.8)
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p1.paragraph_format.space_before = Pt(4)
    p1.paragraph_format.space_after = Pt(4)
    r1 = p1.add_run(f"({eq_number_str})")
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(11)
    r1.font.bold = True
```

### E. Centered High-Resolution Figures with Academic Captions
```python
def add_academic_figure(doc, image_path, fig_num, title, description, width_inches=5.8):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(10)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(image_path, width=Inches(width_inches))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    p_cap.paragraph_format.left_indent = Inches(0.3)
    p_cap.paragraph_format.right_indent = Inches(0.3)
    
    r_lbl = p_cap.add_run(f"Figure {fig_num}: {title}. ")
    r_lbl.font.name = 'Times New Roman'
    r_lbl.font.size = Pt(10)
    r_lbl.font.bold = True
    
    r_desc = p_cap.add_run(description)
    r_desc.font.name = 'Times New Roman'
    r_desc.font.size = Pt(10)
    r_desc.font.italic = False
```

---

## 5. BibTeX Citation Extraction & Mapping Workflow

1. **BibTeX Parser**:
   Extract all `@article` and `@inproceedings` from `ResearchPaperWrite/ACM_Template/sample-base.bib`.
2. **Sequential Citation Matching**:
   As sections are written, link statements (e.g., `ViT [1]`, `SSRN [2]`, `A2S2K-ResNet [3]`, `DMC [4]`).
3. **IEEE Bibliography Generation**:
   At the document conclusion, generate a formatted IEEE bibliography:
   - `[1] Z. Zhong, J. Li, Z. Luo, and M. Chapman, "Spectral-spatial residual network for hyperspectral image classification: A 3-D deep learning framework," IEEE Trans. Geosci. Remote Sens., vol. 56, no. 2, pp. 847–858, Feb. 2018.`

---

## 6. Execution Protocol for Claude

When tasked with generating the project report:
1. **Source Ingestion**:
   - Inspect `SGSCF_Net.py` for exact mathematical equations, dimensions, and layer names.
   - Read `BASELINE_MODELS_MASTER_README.html` for 8 baseline metrics (OA, AA, Kappa, Training Time, Parameters).
   - Read `MASTER_README_120+runs_SGSCF_Net.html` for 60-run grid search and 1%-10% data scarcity stress tests.
   - Read `sample-base.bib` for peer-reviewed citations.
2. **Chart Synthesis**:
   - Write a standalone Python script (`generate_report_charts.py`) using `matplotlib` to render clean, publication-grade figures:
     * Bar chart comparing SGSCF-Net vs 8 baselines on Indian Pines.
     * Convergence loss & accuracy curves.
     * Heatmap of Patch Size vs Batch Size grid search.
     * Line plot of OA degradation across 1%, 2%, 5%, 10% data scarcity.
3. **Docx Document Assembly**:
   - Write `build_project_report_docx.py` using the reusable patterns above.
   - Insert NIT Raipur logo (`nit_raipur_logo.jpeg`), Cover page, Front matter, 8 Chapters, Tables, Equations, Figures, and References.
   - Execute the script to generate `SGSCF_Net_Minor_Project_Report.docx`.
4. **Quality & Formatting Verification**:
   - Ensure zero missing citations or placeholder text.
   - Verify all numbers match the master truth files exactly.
