import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Custom Canvas for dynamic Running Header & Two-Pass Total Page Count Footer
class DefenseCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Draw header on pages 2 and later
        if self._pageNumber > 1:
            self.drawString(40, 752, "SGSCF-Net: Slidewise Presentation Defense Script & Viva Handbook")
            self.setFont("Helvetica", 8)
            self.drawRightString(572, 752, "NIT Raipur — Dept. of IT | Nov 2026")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(40, 746, 572, 746)

        # Draw footer on all pages
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(40, 32, "Tejas Surya (23118098) | B. Vyshnavi (23118021) | Valluri Lahitha (23118106) — Supervisor: Dr. R.R. Janghel")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 32, page_text)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(40, 44, 572, 44)
        
        self.restoreState()

def create_script_pdf(filename="SGSCF_Net_Presentation_Defense_Script.pdf"):
    pdf_path = os.path.join(r"C:\Users\LENOVO\Desktop\MinorProject_Final", filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=52,
        bottomMargin=52
    )
    
    usable_width = 532
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        alignment=1 # Center
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1D4ED8"),
        alignment=1 # Center
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
        alignment=1 # Center
    )
    
    section_title = ParagraphStyle(
        'SecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )
    
    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    script_bullet = ParagraphStyle(
        'ScriptBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#0F172A"),
        leftIndent=12,
        firstLineIndent=-12
    )
    
    math_bullet = ParagraphStyle(
        'MathBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#065F46"),
        leftIndent=12,
        firstLineIndent=-12
    )
    
    qa_q_style = ParagraphStyle(
        'QAQuestion',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#991B1B"),
        leftIndent=0
    )
    
    qa_a_style = ParagraphStyle(
        'QAAnswer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=8
    )

    def make_callout(title, elements, bg_color="#F8FAFC", border_color="#CBD5E1", title_color="#0F172A"):
        header = Paragraph(f"<b>{title}</b>", ParagraphStyle(
            'BoxHeader', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.HexColor(title_color)
        ))
        content = [header, Spacer(1, 3)] + elements
        t = Table([[content]], colWidths=[usable_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor(border_color)),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    def slide_header_card(slide_num, title, timing, speaker, visual_cue):
        header_text = f"<b>SLIDE {slide_num} / 20: {title.upper()}</b>"
        sub_text = f"<b>Allocated Time:</b> {timing} &nbsp;|&nbsp; <b>Suggested Speaker:</b> {speaker} &nbsp;|&nbsp; <b>On Screen:</b> <i>{visual_cue}</i>"
        
        p1 = Paragraph(header_text, ParagraphStyle('SH1', fontName='Helvetica-Bold', fontSize=10.5, leading=14, textColor=colors.white))
        p2 = Paragraph(sub_text, ParagraphStyle('SH2', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor("#E2E8F0")))
        
        t = Table([[p1], [p2]], colWidths=[usable_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1E3A8A")),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    story = []
    
    # ==================== COVER / TITLE BLOCK ====================
    story.append(Paragraph("NATIONAL INSTITUTE OF TECHNOLOGY, RAIPUR", ParagraphStyle('UnivHead', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=colors.HexColor("#475569"), alignment=1)))
    story.append(Paragraph("Department of Information Technology — Minor Project Final Defense (Nov 2026)", ParagraphStyle('DeptHead', fontName='Helvetica-Bold', fontSize=9.5, leading=13, textColor=colors.HexColor("#64748B"), alignment=1)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("SGSCF-Net Presentation Defense Script & Viva Guide", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Spatial-Guided Spectral-Channel Fusion Network for Hyperspectral Remote Sensing Image Analysis", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Under Guidance of:</b> Dr. R.R. Janghel, Associate Professor &nbsp;|&nbsp; <b>Presented by:</b> Tejas Surya (23118098), B. Vyshnavi (23118021), Valluri Lahitha (23118106)", meta_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceBefore=2, spaceAfter=10))

    # ==================== EXECUTIVE STRATEGY & BENCHMARK CHEAT SHEET ====================
    story.append(Paragraph("EXECUTIVE DEFENSE STRATEGY & NUMERICAL TRUTH SHEET", section_title))
    
    strategy_html = (
        "<b>Defense Objective:</b> Defend the minor project with flawless technical authority. Prove that SGSCF-Net solves the "
        "Hughes phenomenon and spectral-spatial entanglement using only <b>308K parameters</b> while outperforming 8 SOTA baselines on minority classes.<br/>"
        "<b>Pacing & Allocation (15 Min Total):</b><br/>"
        "• <b>Speaker 1 (Tejas Surya) — Slides 1 to 7 (6.5 min):</b> Introduction, Problem Recap, End-to-End Architecture, SBAM, CBSA, SCFG modules.<br/>"
        "• <b>Speaker 2 (B. Vyshnavi) — Slides 8 to 13 (4.5 min):</b> Minority-Aware Hybrid Loss, Protocol & Baselines, Indian Pines, Pavia Univ, Salinas Valley, Maps.<br/>"
        "• <b>Speaker 3 (Valluri Lahitha) — Slides 14 to 20 (4.0 min):</b> Parameter Efficiency, Ablation Study & Depth, 60-Run Grid Search, Data Scarcity, Achievements, Future Work."
    )
    story.append(Paragraph(strategy_html, body_style))
    story.append(Spacer(1, 8))

    # Cheat Sheet Table
    table_data = [
        [Paragraph("<b>Metric / Dimension</b>", bold_body), Paragraph("<b>Indian Pines (AVIRIS)</b>", bold_body), Paragraph("<b>Pavia University (ROSIS)</b>", bold_body), Paragraph("<b>Salinas Valley (AVIRIS)</b>", bold_body)],
        [Paragraph("<b>Sensor / Resolution</b>", body_style), Paragraph("AVIRIS (200 bands), 20m/px", body_style), Paragraph("ROSIS (103 bands), 1.3m/px", body_style), Paragraph("AVIRIS (204 bands), 3.7m/px", body_style)],
        [Paragraph("<b>Spatial Grid / Classes</b>", body_style), Paragraph("145 × 145 px, 16 Classes", body_style), Paragraph("610 × 340 px, 9 Classes", body_style), Paragraph("512 × 217 px, 16 Classes", body_style)],
        [Paragraph("<b>SGSCF-Net OA (%)</b>", bold_body), Paragraph("<b>98.05%</b> (Rank 2 vs SSRN 98.52%)", body_style), Paragraph("<b>99.82%</b> (SSRN: 99.87%)", body_style), Paragraph("<b>99.92%</b> (A2S2K: 100.00%)", body_style)],
        [Paragraph("<b>SGSCF-Net AA (%)</b>", bold_body), Paragraph("<b>94.82% (RANK 1 of all 9 models!)</b>", bold_body), Paragraph("<b>99.71%</b>", body_style), Paragraph("<b>99.92%</b>", body_style)],
        [Paragraph("<b>Minority Defense Truth</b>", body_style), Paragraph("SSRN has 0% on Oats/Alfalfa; Ours: 95% Alfalfa, 54.2% Oats", body_style), Paragraph("Gravel: 99.76% (SSAN drops to 17.96%)", body_style), Paragraph("11 of 16 classes at 100.00%", body_style)],
        [Paragraph("<b>Params & Checkpoint</b>", body_style), Paragraph("<b>308,097 params (1.2 MB)</b>", bold_body), Paragraph("10,387 novel attention params (3.37%)", body_style), Paragraph("vs SSAN: 6.40M (20.8× smaller!)", body_style)],
        [Paragraph("<b>Inference Time (IP)</b>", body_style), Paragraph("<b>1.18 seconds</b> (vs 135.74s for SSAN)", body_style), Paragraph("Linear O(N) complexity", body_style), Paragraph("No quadratic O(N²) self-attention", body_style)]
    ]
    t_summary = Table(table_data, colWidths=[120, 140, 136, 136])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # ==================== SLIDE-BY-SLIDE WALKTHROUGH ====================

    slides = [
        # SLIDE 1
        {
            "num": 1,
            "title": "Title Slide & Project Introduction",
            "time": "45 Seconds",
            "speaker": "Tejas Surya",
            "visual": "NIT Raipur banner, Project Title, Guide Dr. R.R. Janghel, Team Members.",
            "spoken": [
                "• <b>Opening Salutation:</b> 'A very good morning to our respected supervisor, Dr. R.R. Janghel, and esteemed members of the evaluation committee.'",
                "• <b>Project Declaration:</b> 'Today, on behalf of our team comprising Tejas Surya, B. Vyshnavi, and Valluri Lahitha, I am proud to present our final Minor Project titled: <i>Spatial-Guided Spectral-Channel Fusion Network for Hyperspectral Remote Sensing Image Analysis</i> (SGSCF-Net).'",
                "• <b>Problem Context Hook:</b> 'Hyperspectral imaging captures hundreds of continuous spectral bands, allowing us to identify Earth surface materials with chemical-level precision. However, existing deep models suffer from severe overfitting, heavy compute exceeding 85 million parameters, and the catastrophic collapse of minority classes.'",
                "• <b>Core Contribution:</b> 'In this work, we present an end-to-end architecture with only 308K parameters that resolves these bottlenecks, achieving state-of-the-art Average Accuracy across three benchmark datasets. Let us begin.'"
            ],
            "math": [
                "• <b>SGSCF-Net Pronunciation:</b> Pronounce clearly as <i>'S-G-S-C-F Net'</i> (Spatial-Guided Spectral-Channel Fusion Network).",
                "• <b>Hyperspectral (HSI):</b> Remind listeners that HSI acquires continuous spectral signatures across hundreds of contiguous narrow bands ($400 - 2500$ nm)."
            ],
            "qa": [
                ("Why did you select Hyperspectral Image Classification for your Minor Project?",
                 "HSI bridges continuous spectroscopy with 2D spatial vision. It is the gold standard for precision agriculture, environmental monitoring, and urban mapping. However, the Hughes phenomenon—where high spectral dimensions paired with small labeled training samples causes severe overfitting—makes it one of the most critical open research challenges in computer vision.")
            ]
        },
        
        # SLIDE 2
        {
            "num": 2,
            "title": "Presentation Outline",
            "time": "30 Seconds",
            "speaker": "Tejas Surya",
            "visual": "13-point roadmap spanning problem recap, architecture, experimental validation, and future directions.",
            "spoken": [
                "• <b>Roadmap Walkthrough:</b> 'Here is the roadmap of our presentation today. We will begin with a brief recap of the core challenges identified during our mid-semester review.'",
                "• <b>Architecture & Novelty:</b> 'Next, we delve into our architectural contributions: the three novel modules—SBAM, CBSA, and SCFG—followed by our minority-aware hybrid loss function.'",
                "• <b>Rigor of Evaluation:</b> 'We will then present comprehensive empirical results across three standard benchmarks—Indian Pines, Pavia University, and Salinas Valley—backed by full classification maps, efficiency benchmarks, ablation studies, a 60-run hyperparameter grid search, and data scarcity stress tests.'",
                "• <b>Paper Deliverable:</b> 'Finally, we summarize our key achievements and future extensions towards journal publication in IEEE TGRS.'"
            ],
            "math": [
                "• <b>Terminology:</b> SOTA = State-of-the-Art; Ablation = Systematically isolating and disabling components to measure their individual contribution."
            ],
            "qa": [
                ("What makes your project defense unique compared to standard minor projects?",
                 "Most minor projects evaluate an off-the-shelf model on a single dataset. We designed 3 novel mathematical modules from scratch, executed 120+ GPU training runs, benchmarked against 8 SOTA baselines under a strictly unified protocol, and drafted a complete 31-page ACM research manuscript.")
            ]
        },

        # SLIDE 3
        {
            "num": 3,
            "title": "Recap: Problem Statement & Research Objectives",
            "time": "60 Seconds",
            "speaker": "Tejas Surya",
            "visual": "Two-column comparison: Mid-Sem Problems Identified vs Final Presentation Deliverables.",
            "spoken": [
                "• <b>Contrasting Problems vs Deliverables:</b> 'During our mid-semester presentation, we identified five critical bottlenecks hindering deep learning in hyperspectral remote sensing, shown on the left.'",
                "• <b>1. The Hughes Phenomenon:</b> 'Hundreds of contiguous spectral bands combined with scarce labeled training pixels cause exponential parameter explosion and severe overfitting.'",
                "• <b>2. Memory Explosion:</b> 'Existing 3D-CNNs and Vision Transformers exceed 85 million parameters, making them impractical for edge deployment on UAVs.'",
                "• <b>3. Feature Entanglement:</b> 'Simultaneous 3D convolutions blur subtle spectral absorption lines with broad spatial textures.'",
                "• <b>4. Severe Class Imbalance:</b> 'Minority classes—such as Oats in Indian Pines with only 20 labeled pixels—consistently collapse to 0% accuracy under standard Cross-Entropy loss.'",
                "• <b>5. Atmospheric Absorption Noise:</b> 'Unfiltered atmospheric water vapor bands (e.g., bands 104–108 and 150–163) corrupt raw feature representations.'",
                "• <b>What We Delivered:</b> 'On the right is what we achieved: a fully trained, 308K parameter model tested across 3 datasets, backed by 120+ GPU runs, achieving the highest Average Accuracy on Indian Pines (94.82%) and solving the minority collapse.'"
            ],
            "math": [
                "• <b>Hughes Phenomenon Definition:</b> For a fixed number of training samples, classification accuracy initially increases as spectral bands increase, but beyond an optimal point, accuracy degrades exponentially due to spatial sparsity in high-dimensional space."
            ],
            "qa": [
                ("How exactly does your architecture prevent the Hughes phenomenon?",
                 "By decoupling spectral recalibration from spatial modeling. SBAM first suppresses noisy bands, CBSA compresses spectral redundancy via complementary gating, and SCFG collapses spectral depth before spatial convolution. This keeps total parameters at only 308K, preventing high-dimensional overfitting.")
            ]
        },

        # SLIDE 4
        {
            "num": 4,
            "title": "SGSCF-Net: End-to-End Architecture Overview",
            "time": "75 Seconds",
            "speaker": "Tejas Surya",
            "visual": "Complete horizontal flow diagram: Input tensor -> SBAM -> CBSA -> ResBlocks -> SCFG -> Classifier Head.",
            "spoken": [
                "• <b>High-Level Architectural Flow:</b> 'This slide presents the end-to-end architecture of SGSCF-Net. The pipeline processes raw hyperspectral cubes through four distinct, mathematically coordinated stages.'",
                "• <b>Input Tensor:</b> 'We take an input patch X of shape Batch × 1 × P × P × C, where P=11 is the spatial neighborhood and C is the number of spectral bands (200 for Indian Pines).'",
                "• <b>Stage I — SBAM:</b> 'The raw cube first enters the Spectral Band Attention Module (SBAM), which gates spectral channels at the raw input level before any convolution.'",
                "• <b>Stage II — CBSA & Spectral Deepening:</b> 'The recalibrated cube enters our Complementary Branch Spectral Attention (CBSA), which splits into dual narrow and wide 3D convolutions, gated by α and (1−α). This is followed by two Spectral Residual Blocks with Efficient Channel Attention (ECA).'",
                "• <b>Stage III — SCFG Transition:</b> 'To transition from the spectral to the spatial stage, our Second-Order Covariance Fusion Gate (SCFG) captures bilinear spatial-channel correlations via outer product.'",
                "• <b>Stage IV — Spatial Refinement & Classifier:</b> 'Two Spatial ResBlocks refine local spatial context, followed by Global Average Pooling and a single linear classifier head of only 272 parameters.'",
                "• <b>Hybrid Loss Optimization:</b> 'The entire network is optimized end-to-end using our 70% Spread Loss and 30% Focal Loss formulation.'"
            ],
            "math": [
                r"• <b>Input Tensor Notation:</b> $X \in \mathbb{R}^{B \times 1 \times P \times P \times C}$ ('X in R, Batch by 1 channel by P by P spatial patch by C spectral bands').",
                r"• <b>GAP:</b> Global Average Pooling collapses $P \times P \to 1 \times 1$, eliminating dense fully-connected layers."
            ],
            "qa": [
                ("Why separate spectral and spatial processing into sequential stages instead of joint 3D convolutions throughout?",
                 "Pure 3D-CNNs (like SSRN) apply joint spatial-spectral kernels throughout, which entangles spectral absorption signatures with spatial textures and leads to 3D kernel parameter explosion. Decoupling them allows dedicated spectral noise filtering first, second-order covariance alignment in the middle, and pure spatial aggregation last—cutting parameters to 308K while boosting feature discriminability.")
            ]
        },

        # SLIDE 5
        {
            "num": 5,
            "title": "Module 1: Spectral Band Attention Module (SBAM)",
            "time": "60 Seconds",
            "speaker": "Tejas Surya",
            "visual": "SBAM flow diagram: Spatial pooling -> MLP bottleneck (r=8) -> Sigmoid -> Hadamard product reweighting.",
            "spoken": [
                "• <b>Purpose & Location:</b> 'Let us look closely at Module 1: the Spectral Band Attention Module, or SBAM. Crucially, SBAM operates on the RAW spectral bands BEFORE any convolution occurs.'",
                "• <b>Spatial Compression:</b> 'First, adaptive 3D average pooling collapses the spatial grid P × P into a per-channel spectral descriptor z_c, capturing the average energy across each band.'",
                "• <b>Squeeze Bottleneck:</b> 'A two-layer MLP bottleneck with reduction ratio r=8 maps channels from C down to C/8, applies ReLU activation, and expands back to C. A Sigmoid function generates continuous attention weights w between 0 and 1.'",
                "• <b>Band Gating via Hadamard Product:</b> 'We recalibrate the input tensor using the element-wise Hadamard product: X-tilde = X ⊙ E(w). Bands corrupted by atmospheric water absorption receive near-zero weights and are suppressed.'",
                "• <b>Contrast with Prior Art:</b> 'Unlike SENet or CBAM, which operate on deep intermediate feature maps, SBAM acts directly on raw sensor inputs, preventing noisy spectral bands from propagating errors downstream.'",
                "• <b>Ultra-Low Parameter Footprint:</b> 'SBAM requires only 10,225 parameters, representing just 3.32% of our total network.'"
            ],
            "math": [
                "• <b>Equations:</b> $z_c = \\frac{1}{P^2} \\sum_{u,v} X(b,1,u,v,c)$ (spatial mean pooling); $w = \\sigma(W_2 \\cdot \\delta(W_1 z + b_1) + b_2)$ (squeeze-excite bottleneck).",
                "• <b>Pronunciation:</b> $\\delta$ = 'delta (ReLU activation)'; $\\sigma$ = 'sigma (Sigmoid gate)'; $\\odot$ = 'Hadamard product or element-wise multiplication'; $r=8$ = 'reduction ratio 8'."
            ],
            "qa": [
                ("Why not simply delete noisy water absorption bands manually before feeding the data?",
                 "Manual band removal is rigid and sensor-dependent (e.g., removing bands 104-108 in AVIRIS). However, adjacent bands still suffer from partial attenuation and sensor noise that varies across atmospheric conditions. SBAM dynamically learns soft continuous importance weights per scene, retaining useful residual signal while suppressing noise adaptively.")
            ]
        },

        # SLIDE 6
        {
            "num": 6,
            "title": "Module 2: Complementary Branch Spectral Attention (CBSA)",
            "time": "60 Seconds",
            "speaker": "Tejas Surya",
            "visual": "Dual-branch Conv3D flow (1x1x7 and 3x3x7) -> Conv1D gate -> Complementary weighting alpha and (1-alpha).",
            "spoken": [
                "• <b>Dual-Branch Concept:</b> 'Module 2 is the Complementary Branch Spectral Attention (CBSA). Rather than using a single 3D kernel, CBSA splits the feature stream into two specialized parallel branches.'",
                "• <b>Branch Specialization:</b> 'Branch a uses a 1×1×7 Conv3D filter to extract pure inter-band spectral transitions without spatial mixing. Branch b uses a 3×3×7 Conv3D filter to capture joint spatial-spectral contextual patterns.'",
                "• <b>Complementary Gating Equation:</b> 'We fuse their outputs using a learned gate α, computed by 1D convolution with kernel k=3 over the pooled features.'",
                "• <b>The Complementary Constraint:</b> 'The fused feature v is computed as: v = BN(α ⊙ a + (1−α) ⊙ b). Notice the mathematical beauty here: α and (1−α) sum identically to 1.'",
                "• <b>Zero Redundancy:</b> 'This strictly forces the two branches to learn non-overlapping, orthogonal representations. When spectral fidelity dominates, α approaches 1; when spatial context dominates, (1−α) takes over. Redundancy between branches is mathematically zero.'",
                "• <b>Extreme Efficiency:</b> 'The gating mechanism uses only 80 parameters (0.03% of the network), followed by two Spectral ResBlocks and ECA-3D.'"
            ],
            "math": [
                "• <b>Equations:</b> $\\alpha = \\sigma(\\text{Conv1D}_k(\\text{MeanPool}(a + b)))$ with $k=3$; $v = \\text{BN}(\\alpha \\odot a + (1-\\alpha) \\odot b)$.",
                "• <b>Key Identity:</b> $\\alpha + (1-\\alpha) \\equiv 1$ strictly bounds the total feature energy and guarantees complementary trade-off."
            ],
            "qa": [
                ("Why use α and (1−α) instead of learning two separate attention weights α and β?",
                 "If α and β are learned independently, both can saturate to 1, causing severe feature redundancy and doubling the feature magnitude. Enforcing the partition of unity $\\alpha + (1-\\alpha) = 1$ forces the branches to compete, ensuring that the network learns orthogonal spectral versus spatial features without duplicating representations.")
            ]
        },

        # SLIDE 7
        {
            "num": 7,
            "title": "Module 3: Second-Order Covariance Fusion Gate (SCFG)",
            "time": "60 Seconds",
            "speaker": "Tejas Surya",
            "visual": "Outer product diagram of f_ch and f_sp -> Frobenius norm normalization -> Linear projection -> Spatial modulation.",
            "spoken": [
                "• <b>Spectral-to-Spatial Transition:</b> 'Module 3 is the Second-Order Covariance Fusion Gate (SCFG), which forms the bridge between spectral feature extraction and spatial refinement.'",
                "• <b>The Limitation of First-Order GAP:</b> 'Standard networks use Global Average Pooling, which only computes the first-order mean of features. This completely discards the spatial-spectral cross-correlations that differentiate similar materials.'",
                "• <b>Second-Order Covariance Formulation:</b> 'SCFG computes two descriptors: f_ch, the channel descriptor averaged over space, and f_sp, the spatial descriptor averaged over channels. We then take their outer product: G = f_ch ⊗ f_sp, producing a second-order covariance matrix.'",
                "• <b>Frobenius Normalization:</b> 'To prevent gradient explosion and stabilize training, we normalize G by its Frobenius norm: G-tilde = G / (||G||_F + ε).'",
                "• <b>Modulation Gate:</b> 'A linear projection with Sigmoid produces the gating vector g, which modulates spatial features element-wise: x_scfg = x ⊙ g.'",
                "• <b>Impact & Params:</b> 'This captures bilinear relationships—such as distinguishing gravel from asphalt, or subtle shadow boundaries—with only 82 parameters (0.03% of the network). Now I hand over to B. Vyshnavi for the loss function and results.'"
            ],
            "math": [
                "• <b>Formulas:</b> $G = f_{ch} \\otimes f_{sp} \\in \\mathbb{R}^{B \\times C \\times N}$; $\\tilde{G} = G / (\\|G\\|_F + \\varepsilon)$; $g = \\sigma(W_G \\cdot \\tilde{G})$.",
                "• <b>Pronunciation:</b> $\\otimes$ = 'outer product or tensor product'; $\\|G\\|_F$ = 'Frobenius norm of matrix G'; $\\varepsilon = 10^{-6}$ = 'epsilon for numerical stability'."
            ],
            "qa": [
                ("What physical information does the outer product f_ch ⊗ f_sp capture in remote sensing?",
                 "In remote sensing, a pixel's spectral signature is strongly correlated with its spatial neighborhood. First-order pooling averages these out independently. The outer product captures the second-order co-occurrence: how specific spectral absorption bands vary across spatial locations, preserving spatial-spectral covariance essential for fine-grained boundary discrimination.")
            ]
        },

        # SLIDE 8
        {
            "num": 8,
            "title": "Minority-Aware Hybrid Loss Function",
            "time": "60 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "Loss equation (0.7 Spread + 0.3 Focal) + 200-epoch convergence curves for IP, PU, and SV.",
            "spoken": [
                "• <b>Transition & Core Motivation:</b> 'Thank you, Tejas. I will now explain our Minority-Aware Hybrid Loss Function, which directly tackles the severe class imbalance problem in hyperspectral datasets.'",
                "• <b>The Hybrid Formulation:</b> 'Our total objective is: L_total = 0.7 × L_spread + 0.3 × L_focal.'",
                "• <b>Spread Loss (70% Weight):</b> 'Spread Loss enforces inter-class separation using a dynamic scheduled margin: m(r) = 0.2 + 0.7 × r, where r is the training progress ratio. In early epochs, a relaxed margin of 0.2 prevents gradient shock on difficult classes. In late epochs, the margin increases to 0.9, forcing wide decision boundaries between visually similar classes.'",
                "• <b>Class-Weighted Focal Loss (30% Weight):</b> 'Focal Loss applies inverse-frequency class weights w_c = 1 / (N_c + ε) with focusing parameter γ = 2.0. This down-weights easy, over-represented majority classes like Soybeans, compelling the optimizer to focus on rare minority samples like Oats.'",
                "• <b>Convergence Curves:</b> 'As shown in the lower plots, loss decreases smoothly over 200 epochs across Indian Pines, Pavia University, and Salinas Valley without oscillations or divergence.'"
            ],
            "math": [
                "• <b>Formulas:</b> $\\mathcal{L}_{total} = 0.7 \\mathcal{L}_{spread} + 0.3 \\mathcal{L}_{focal}$; Margin schedule $m(r) = 0.2 + 0.7 \\cdot r$ ($0.2 \\le m \\le 0.9$).",
                "• <b>Focal Weight:</b> $w_c = 1 / (N_c + \\varepsilon)$ down-weights frequent classes; $\\gamma = 2.0$ exponentially suppresses well-classified samples ($p_t \\to 1$)."
            ],
            "qa": [
                ("Why use 0.7 Spread Loss and 0.3 Focal Loss? Why not 50/50?",
                 "We systematically tuned this ratio. Spread Loss is primary (0.7) because it directly maximizes angular geometric margins in feature space, preventing feature overlap. Focal Loss (0.3) serves as a specialized minority regularizer. In our ablation (Slide 15), this exact 0.7/0.3 balance boosted Average Accuracy by +1.04% and lifted Oats accuracy from 0% to 54.17%.")
            ]
        },

        # SLIDE 9
        {
            "num": 9,
            "title": "Experimental Setup & Benchmark Datasets",
            "time": "45 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "Protocol card: 10/10/80 split, 11x11 patch, 8 baselines benchmarked from scratch.",
            "spoken": [
                "• <b>Evaluation Protocol:</b> 'To ensure rigorous and reproducible evaluation, we standardized our experimental setup across all experiments.'",
                "• <b>Data Split:</b> 'We use a strict 10% Train / 10% Validation / 80% Test split, stratified per class to preserve sample proportions. Training and test sets are strictly non-overlapping.'",
                "• <b>Hyperparameters:</b> 'Spatial patch size is 11×11, batch size is 32, trained for 200 epochs using the Adam optimizer with initial learning rate 5×10⁻⁴ decaying to 10⁻⁶ via Cosine Annealing.'",
                "• <b>Hardware:</b> 'All runs were executed on NVIDIA Tesla T4 GPUs via Kaggle.'",
                "• <b>8 SOTA Baselines Benchmarked:</b> 'We benchmarked 8 representative models: 3D CNNs (SSRN, A2S2K-ResNet), Vision Transformers (ViT, HSI-SSFTT, HyperRoPE-HiT), Attention Networks (SSAN), and Hybrid models (DMC, CRV-MHDL).'",
                "• <b>Complete Baseline Retraining:</b> 'All 24 baseline runs (8 models × 3 datasets) were trained completely from scratch under the identical protocol for 100% fair comparison.'"
            ],
            "math": [
                "• <b>Data Split:</b> 10% Train / 10% Val / 80% Test — strictly non-overlapping coordinates, preventing spatial data leakage.",
                "• <b>Cosine Annealing LR:</b> $\\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})(1 + \\cos(\\frac{t}{T}\\pi))$."
            ],
            "qa": [
                ("Did you use random pixel splitting or patch-based splitting? Does it cause data leakage?",
                 "We used stratified patch extraction with strictly disjoint pixel coordinate masks. The test pixels (80%) were never seen during training or validation, and patch centers are strictly non-overlapping with training centers, eliminating spatial autocorrelation leakage.")
            ]
        },

        # SLIDE 10
        {
            "num": 10,
            "title": "Results: Indian Pines — Per-Model Comparison",
            "time": "75 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "Quantitative comparison table on Indian Pines: OA, AA, Kappa, Time, plus GT vs Ours map.",
            "spoken": [
                "• <b>Indian Pines Overview:</b> 'Indian Pines is the most demanding HSI benchmark due to low spatial resolution (20m/pixel) and severe class imbalance, with sample counts ranging from 954 down to 20.'",
                "• <b>Our Quantitative Metrics:</b> 'SGSCF-Net achieves 98.05% Overall Accuracy, a Cohen’s Kappa of 0.9778, and trains in only 8.80 minutes.'",
                "• <b>Addressing the SSRN Comparison:</b> 'Now, I draw your attention to a critical finding. SSRN reports a slightly higher Overall Accuracy of 98.52% (+0.47%). However, look at the Average Accuracy (AA)!'",
                "• <b>The Minority Collapse in SSRN:</b> 'SSRN achieves an AA of only 79.99%. Why? Because SSRN scored exactly 0.00% on Alfalfa, 0.00% on Grass-pasture-mowed, and 0.00% on Oats! It achieved high OA purely by memorizing the large majority classes.'",
                "• <b>SGSCF-Net Wins on Average Accuracy:</b> 'In stark contrast, SGSCF-Net achieves 95.00% on Alfalfa, 100.00% on Grass-mowed, and 54.17% on Oats, delivering the HIGHEST Average Accuracy of 94.82% across all 9 models!'",
                "• <b>Dominating Other Baselines:</b> 'We also outperform ViT (90.33%), DMC (82.55%), CRV-MHDL (67.20%), and SSAN (64.19%).'"
            ],
            "math": [
                "• <b>Overall Accuracy (OA):</b> $\\text{OA} = \\frac{\\sum \\text{TP}}{\\text{Total Samples}}$ (heavily biased toward majority classes).",
                "• <b>Average Accuracy (AA):</b> $\\text{AA} = \\frac{1}{C}\\sum_{c=1}^C \\text{Accuracy}_c$ (weights every class equally, revealing minority health)."
            ],
            "qa": [
                ("If SSRN has 98.52% OA and your model has 98.05% OA, why is your model superior?",
                 "Because OA is mathematically biased in imbalanced datasets. SSRN completely failed on 3 classes—achieving 0% on Oats, 0% on Alfalfa, and 0% on Grass-mowed. In real-world remote sensing, misclassifying rare crops is fatal. SGSCF-Net achieved 94.82% Average Accuracy—the highest across all 9 models—correctly classifying 95% of Alfalfa and 54.2% of Oats. Our model is genuinely balanced and fair.")
            ]
        },

        # SLIDE 11
        {
            "num": 11,
            "title": "Results: Pavia University — Urban Material Classification",
            "time": "60 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "Large stat callouts: 99.82% OA, 99.71% AA, 99.76 Kappa, 12.9 min. GT vs Ours maps.",
            "spoken": [
                "• <b>Dataset Characteristics:</b> 'Moving to Pavia University: an urban scene captured by the ROSIS sensor with 103 spectral bands and a high 1.3m spatial resolution across 9 land-cover classes.'",
                "• <b>Near-Perfect Results:</b> 'SGSCF-Net delivers outstanding performance: 99.82% Overall Accuracy, 99.71% Average Accuracy, and a Kappa coefficient of 99.76, training in just 12.9 minutes.'",
                "• <b>Per-Class Highlights:</b> 'Four classes reached a perfect 100.00% accuracy: Asphalt, Painted Metal, Bare Soil, and Bitumen. On Meadows, all 14,919 test pixels were classified with 100.00% precision.'",
                "• <b>Gravel vs Asphalt Separation:</b> 'Gravel, which is historically misclassified as Asphalt due to spectral similarity, reached 99.76% accuracy.'",
                "• <b>Shadow Detection:</b> 'Shadow pixels reached 99.34% accuracy, directly enabled by our SCFG covariance gate capturing subtle ambient illumination ratios.'",
                "• <b>Comparison & Speedup:</b> 'SSRN achieved 99.87% OA in 11.36 min. A2S2K-ResNet reached 99.83% OA but took 41.12 minutes—making SGSCF-Net 3.2× faster to train while matching top-tier accuracy. Meanwhile, SSAN collapsed to 76.82% OA, with Gravel dropping to 17.96%.'"
            ],
            "math": [
                "• <b>Kappa Statistic ($\\kappa$):</b> $\\kappa = \\frac{P_o - P_e}{1 - P_e}$ measures agreement adjusted for chance agreement $P_e$. Our $\\kappa = 0.9976$ indicates near-perfect concordance."
            ],
            "qa": [
                ("Why did SSAN perform so poorly (76.82%) on Pavia University?",
                 "SSAN has 6.4 million parameters. On a 10% training split, its massive self-attention layers severely overfit the urban landscape, failing completely on ambiguous classes like Gravel (17.96%). SGSCF-Net’s 308K parameter design prevents overfitting and generalizes effortlessly.")
            ]
        },

        # SLIDE 12
        {
            "num": 12,
            "title": "Results: Salinas Valley — Fine-Grained Agriculture",
            "time": "60 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "Large stat callouts: 99.92% OA, 99.92% AA, 99.91 Kappa, 57.7 min. GT vs Ours maps.",
            "spoken": [
                "• <b>Salinas Valley Overview:</b> 'Our third benchmark is Salinas Valley, captured by AVIRIS: 204 calibrated spectral bands at 3.7m resolution, featuring 16 fine-grained agricultural classes.'",
                "• <b>Near-Perfect Accuracy:</b> 'SGSCF-Net achieved 99.92% Overall Accuracy, 99.92% Average Accuracy, and 99.91 Kappa.'",
                "• <b>Per-Class Highlights:</b> '11 out of 16 classes reached a perfect 100.00% accuracy.'",
                "• <b>Fine-Grained Growth Stage Differentiation:</b> 'Most remarkably, our model successfully differentiated all four developmental stages of Romaine lettuce—at 4 weeks, 5 weeks, 6 weeks, and 7 weeks—with accuracies between 99.65% and 100.00%.'",
                "• <b>Vineyard Classification:</b> 'We also separated untrained Vineyard (99.73%) from trellis Vineyard (100.00%).'",
                "• <b>Progression Across Benchmarks:</b> 'Notice our accuracy progression across datasets: 98.05% on noisy, low-resolution Indian Pines, rising to 99.82% on urban Pavia, and reaching 99.92% on high-SNR agricultural Salinas. Baselines like SSAN collapsed to 88.20%.'"
            ],
            "math": [
                "• <b>Phenological Spectral Shifts:</b> Chlorophyll absorption at 670 nm and NIR reflectance plateaus at 800 nm vary subtly by crop maturity; SBAM and CBSA isolate these narrow-band transitions."
            ],
            "qa": [
                ("How does the model distinguish between Romaine lettuce at 4 weeks vs 7 weeks?",
                 "As lettuce plants mature from 4 to 7 weeks, canopy cover expands and chlorophyll concentration increases, deepening the 670 nm chlorophyll absorption trough and increasing the near-infrared reflectance slope. CBSA’s 1×1×7 spectral branch captures these subtle inter-band slope changes, while SCFG models canopy texture variations.")
            ]
        },

        # SLIDE 13
        {
            "num": 13,
            "title": "Visual Comparison: Classification Maps Across Models",
            "time": "60 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "3x10 grid of classification maps across IP, PU, SV for GT, SGSCF-Net, and 8 baselines.",
            "spoken": [
                "• <b>Qualitative Validation:</b> 'This slide provides a comprehensive visual qualitative comparison across all three datasets, comparing Ground Truth, SGSCF-Net, and all 8 baselines side by side.'",
                "• <b>Indian Pines (Top Row):</b> 'Notice the extensive salt-and-pepper noise and speckle artifacts in ViT, SSAN, DMC, and CRV-MHDL. In contrast, SGSCF-Net produces clean, uniform crop parcels that closely replicate the ground truth map.'",
                "• <b>Pavia University (Middle Row):</b> 'In Pavia, notice the Meadows and bare soil regions. Baselines produce severe edge blurring and misclassify roof tiles. SGSCF-Net maintains sharp linear road boundaries and crisp building footprints.'",
                "• <b>Salinas Valley (Bottom Row):</b> 'In Salinas, the agricultural crop strips are cleanly partitioned in SGSCF-Net, without inter-field bleeding.'",
                "• <b>Transition to Speaker 3:</b> 'These maps confirm that our quantitative numbers translate directly into superior spatial coherence. I will now invite Valluri Lahitha to present our efficiency analysis and ablation studies.'"
            ],
            "math": [
                "• <b>Visual Coherence:</b> Homogeneous spatial regions reflect high spatial-spectral feature consistency without spatial smoothing filters."
            ],
            "qa": [
                ("Did you apply post-processing (like Conditional Random Fields or morphological filtering) to clean up the classification maps?",
                 "No. All classification maps displayed are the direct, raw predictions of SGSCF-Net. The spatial smoothness and clean boundaries arise entirely from our CBSA spatial-spectral gating and SCFG covariance fusion, without requiring any post-processing.")
            ]
        },

        # SLIDE 14
        {
            "num": 14,
            "title": "Computational Efficiency & Parameter Analysis",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "Parameter comparison table + Module breakdown pie chart (SBAM 3.3%, CBSA 0.03%, SCFG 0.03%, ResBlocks 81%).",
            "spoken": [
                "• <b>Transition & Core Focus:</b> 'Thank you, Vyshnavi. A central goal of our project was to achieve high accuracy without prohibitive computational cost.'",
                "• <b>Overall Parameter Comparison:</b> 'SGSCF-Net contains only 308,097 parameters, producing a compact 1.2 MB checkpoint file.'",
                "• <b>Dramatic Size Reductions:</b> 'Compared to baselines: SSAN has 6.40 million parameters—making SGSCF-Net 20.8× more compact! ViT has ~610K parameters—nearly 2× larger—yet achieved only 90.33% OA. A2S2K-ResNet has ~380K parameters but took 3.2× longer to train on Pavia (41.1 min vs 12.9 min).'",
                "• <b>Module-Level Parameter Breakdown:</b> 'Where are the parameters located? SBAM has 10,225 params (3.32%). CBSA has only 80 params (0.03%). SCFG has only 82 params (0.03%).'",
                "• <b>3 Novel Modules = Only 3.37%:</b> 'Together, our three novel attention mechanisms consume just 10,387 parameters—only 3.37% of the total network! Over 81% of the weights reside in standard spatial ResBlocks.'",
                "• <b>Ultra-Slim Classifier Head:</b> 'Our classifier head uses only 272 parameters (0.09%) via Global Average Pooling, eliminating dense fully-connected layers that cause overfitting.'"
            ],
            "math": [
                "• <b>Linear Classifier Head:</b> With $C_{out} = 16$ channels from GAP and 16 classes: $16 \\times 16 \\text{ weights} + 16 \\text{ biases} = 272$ parameters.",
                "• <b>Checkpoint Size:</b> $308,097 \\text{ float32 params} \\times 4 \\text{ bytes} \\approx 1.23 \\text{ MB}$."
            ],
            "qa": [
                ("How does your parameter efficiency translate to inference speed?",
                 "Because our network avoids quadratic self-attention matrices and heavy dense layers, test inference on the entire Indian Pines test set (8,230 pixels) takes only 1.18 seconds, compared to 135.74 seconds for SSAN—a greater than 100× inference speedup.")
            ]
        },

        # SLIDE 15
        {
            "num": 15,
            "title": "Ablation Study & Block Depth Sensitivity",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "Ablation table (5 rows, stepwise additions) + Depth curve (D=1 to D=4, showing D=2 peak).",
            "spoken": [
                "• <b>Systematic Component Ablation:</b> 'To prove that every proposed module actively contributes to our final performance, we conducted a 5-stage ablation study on Indian Pines.'",
                "• <b>Step 1 — Baseline with SBAM Only:</b> 'Achieves 93.65% OA and 89.24% AA.'",
                "• <b>Step 2 — Adding CBSA:</b> 'OA jumps by +1.53% to 95.18%, and AA increases to 91.80%, proving that complementary dual-branch gating reduces spectral redundancy.'",
                "• <b>Step 3 — Adding SCFG:</b> 'OA rises by another +1.26% to 96.44%, demonstrating the power of second-order spatial-channel covariance.'",
                "• <b>Step 4 & 5 — Loss Function Ablation:</b> 'With Cross-Entropy alone, the model reaches 97.26% OA. Replacing Cross-Entropy with our Hybrid Loss pushes OA to 98.05% (+0.79%) and AA to 94.82% (+1.04%).'",
                "• <b>The Decisive Oats Proof:</b> 'Most crucially: on the minority Oats class, Cross-Entropy yielded 0.00% accuracy, while our Hybrid Loss achieved 54.17%!'",
                "• <b>Depth Sensitivity (D):</b> 'We also varied residual block depth D from 1 to 4. D=1 underfits (95.82% OA). D=2 reaches the optimal peak (98.05% OA). Beyond D=2, performance degrades to 97.88% at D=3 and 97.40% at D=4 due to parameter saturation. Hence, D=2 is the proven optimal sweet spot.'"
            ],
            "math": [
                "• <b>Ablation Progression:</b> Baseline (93.65%) $\\to$ +CBSA (95.18%) $\\to$ +SCFG (96.44%) $\\to$ +CE (97.26%) $\\to$ +Hybrid Loss (98.05% OA / 94.82% AA).",
                "• <b>Depth Scaling:</b> $D=1$ (179K), $D=2$ (308K, optimal), $D=3$ (437K), $D=4$ (566K)."
            ],
            "qa": [
                ("Why does increasing depth to D=3 and D=4 decrease accuracy?",
                 "Because hyperspectral training sets are relatively small (approx. 1,024 training pixels for Indian Pines). Increasing depth to D=3 and D=4 increases parameters to 437K and 566K, inducing gradient degradation and overfitting in the 3D convolutional kernels. D=2 provides the exact receptive field depth required.")
            ]
        },

        # SLIDE 16
        {
            "num": 16,
            "title": "60-Run Hyperparameter Grid Search",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "3 OA heatmaps (IP, PU, SV) across 5 patch sizes (9 to 17) and 4 batch sizes (32 to 256).",
            "spoken": [
                "• <b>Exhaustive Search Design:</b> 'To ensure our hyperparameter choices were not cherry-picked, we executed an exhaustive 60-run grid search across all three benchmark datasets.'",
                "• <b>Search Space:</b> 'We evaluated 5 patch sizes—9×9, 11×11, 13×13, 15×15, and 17×17—across 4 batch sizes—32, 64, 128, and 256—generating 20 full training runs per dataset, totaling 60 runs.'",
                "• <b>Optimal Configuration:</b> 'As shown in the three heatmaps, Patch 11×11 with Batch Size 32 consistently achieves Rank 1 or within 0.08% of peak OA across all three datasets (98.05% IP, 99.82% PU, 99.92% SV).'",
                "• <b>Spatial Window Trade-off:</b> 'A 9×9 patch underfits because it lacks adequate contextual neighborhood. Conversely, larger patches (15×15, 17×17) cause boundary contamination—at 20m/pixel in Indian Pines, an 11×11 patch spans exactly 220m × 220m ground distance, which cleanly matches typical agricultural field parcels.'",
                "• <b>Batch Size Dynamics:</b> 'Batch size 32 outperforms Batch 256 by a massive 3.43% OA on Indian Pines. Smaller batches generate higher gradient variance, preventing majority classes from dominating parameter updates.'"
            ],
            "math": [
                "• <b>Ground Footprint:</b> $\\text{Ground Distance} = P \\times \\text{GSD} = 11 \\times 20\\text{m} = 220\\text{m}$.",
                "• <b>Stochastic Gradient Variance:</b> $\\text{Var}(\\nabla L) \\propto 1/B$; smaller batch $B=32$ provides stochastic noise that helps escape sharp saddle points in minority class subspaces."
            ],
            "qa": [
                ("Why does Batch Size 256 perform 3.43% worse than Batch Size 32 on Indian Pines?",
                 "In an imbalanced dataset with batch size 256, minority samples like Oats appear in less than 1% of the batches. As a result, the batch gradient is overwhelmingly dominated by majority classes (Soybean/Corn), completely washing out minority gradients. Batch size 32 yields frequent, high-variance updates that allow minority class gradients to make meaningful parameter updates.")
            ]
        },

        # SLIDE 17
        {
            "num": 17,
            "title": "Data Scarcity Stress Testing (1% – 10% Training Ratios)",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "Data scarcity table + 3 degradation curve plots showing 1%, 2%, 5%, 10% training ratios.",
            "spoken": [
                "• <b>Real-World Operational Context:</b> 'In real-world remote sensing missions, collecting labeled ground truth is expensive and labor-intensive. To evaluate real-world robustness, we performed 12 data scarcity stress tests across training ratios of 1%, 2%, 5%, and 10%.'",
                "• <b>Salinas Valley — Extreme Resilience:</b> 'Salinas Valley proved remarkably resilient: its accuracy dropped by only 2.20% from 10% down to 1% training data, maintaining an extraordinary 97.73% OA with only 1% labels!'",
                "• <b>Pavia University — Moderate Degradation:</b> 'Pavia University dropped by only 4.43%, retaining 95.41% OA at 1% training data.'",
                "• <b>Indian Pines — Steepest Drop:</b> 'Indian Pines dropped by 27.73% (from 97.84% down to 70.12% at 1%). This is mathematically expected: at a 1% split, minority classes like Oats have only 1 to 2 training samples in total!'",
                "• <b>Takeaway:</b> 'Even under extreme label scarcity, SGSCF-Net remains numerically stable, converging reliably without divergence across all three benchmarks.'"
            ],
            "math": [
                "• <b>Sample Count at 1% Split (Indian Pines):</b> Total samples = 10,249; 1% Train = 102 pixels across 16 classes. Oats (20 total pixels) receives $\\text{round}(20 \\times 0.01) = 1$ sample.",
                "• <b>Performance Retention:</b> Salinas Valley retains $97.73 / 99.93 = 97.8\\%$ of its accuracy with a 90% reduction in training data."
            ],
            "qa": [
                ("Why is Salinas Valley so much more resilient to label scarcity (only -2.20% drop) than Indian Pines (-27.73%)?",
                 "Salinas Valley consists of large, spatially contiguous agricultural plots with high signal-to-noise ratio and 3.7m spatial resolution. A 1% sample provides representative spectral signatures that generalize cleanly across the entire field. In contrast, Indian Pines has severe class imbalance, small fragmented fields, and 20m mixed pixels, where 1% training leaves minority classes with only 1 sample.")
            ]
        },

        # SLIDE 18
        {
            "num": 18,
            "title": "Cross-Dataset Summary & Key Achievements",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "4 large metric stat cards (98.05%, 99.82%, 99.92%, 308K) + 7 core impact pillars.",
            "spoken": [
                "• <b>Project Achievements Recap:</b> 'To synthesize our overall findings: SGSCF-Net achieved 98.05% OA on Indian Pines, 99.82% OA on Pavia University, and 99.92% OA on Salinas Valley using only 308K parameters.'",
                "• <b>Benchmark Highlights:</b> 'We secured the highest Average Accuracy on Indian Pines (94.82%) across all 9 models, executed over 120 GPU runs, achieved linear O(N) complexity, and achieved a 1.18-second test inference speed—over 100× faster than SSAN.'",
                "• <b>Seven Pillars of Impact:</b> 'Why does SGSCF-Net matter? It is:'",
                "  1. <b>Lightweight:</b> 308K params vs 6.4M for SSAN and ~610K for ViT.",
                "  2. <b>Balanced:</b> Eliminates minority collapse where SSRN/SSAN scored 0%.",
                "  3. <b>Generalizable:</b> Proven top-tier across agricultural, urban, and fine-grained scenes.",
                "  4. <b>Efficient:</b> Trains in 8.8 minutes vs 41.1 minutes for A2S2K-ResNet.",
                "  5. <b>Modular:</b> SBAM, CBSA, and SCFG are plug-and-play for any 3D-CNN backbone.",
                "  6. <b>Reproducible:</b> Fixed 10/10/80 protocol across public benchmark datasets.",
                "  7. <b>Edge-Deployable:</b> 1.2 MB checkpoint, ready for UAV and satellite payloads.'"
            ],
            "math": [
                "• <b>Complexity Comparison:</b> Standard Transformer Attention = $\\mathcal{O}(N^2 \\cdot C)$; SGSCF-Net = $\\mathcal{O}(N \\cdot K^2 \\cdot C)$ (strictly linear in spatial tokens $N$)."
            ],
            "qa": [
                ("What makes your model edge-deployable compared to Transformers?",
                 "Vision Transformers require quadratic $\\mathcal{O}(N^2)$ memory to compute self-attention maps, which easily exceeds GPU memory limits on embedded hardware. SGSCF-Net uses $\\mathcal{O}(N)$ linear operations, has a tiny 1.2 MB checkpoint, and runs full test inference in 1.18 seconds, fitting comfortably on embedded platforms like NVIDIA Jetson Nano or Orin.")
            ]
        },

        # SLIDE 19
        {
            "num": 19,
            "title": "Future Work & Research Directions",
            "time": "45 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "4-pillar future roadmap: Quantization, Self-Supervised Pretraining, LiDAR Fusion, Journal Publication.",
            "spoken": [
                "• <b>Future Trajectories:</b> 'We have outlined four clear, impactful directions for extending this work:'",
                "• <b>1. Hardware-Aware Quantization (Near-Term):</b> 'Applying INT8 and FP8 post-training quantization and structured channel pruning to compress the model below 150K parameters, targeting real-time onboard UAV inference under 50ms per patch.'",
                "• <b>2. Self-Supervised Pretraining (Mid-Term):</b> 'Developing a masked spectral autoencoder on unlabeled satellite cubes to enable cross-sensor transfer learning (e.g., AVIRIS to EnMAP) with less than 0.5% labeled data.'",
                "• <b>3. Multi-Modal Sensor Fusion (Mid-Term):</b> 'Extending our SCFG covariance gate to fuse Hyperspectral imaging with LiDAR elevation streams on Houston 2013 and MUUFL Gulfport for combined spectral-structural terrain parsing.'",
                "• <b>4. Journal Publication (Ongoing):</b> 'Targeting submission to IEEE Transactions on Geoscience and Remote Sensing (TGRS), incorporating statistical significance tests and generative augmentation studies.'"
            ],
            "math": [
                "• <b>Quantization:</b> INT8 reduces memory by $4\\times$ (from 32-bit floats), accelerating integer tensor cores.",
                "• <b>Cross-Modal Covariance:</b> $G_{fusion} = f_{hsi} \\otimes f_{lidar} \\in \\mathbb{R}^{B \\times C \\times 1}$."
            ],
            "qa": [
                ("How would you adapt the SCFG module for LiDAR fusion?",
                 "LiDAR provides a 1-channel spatial elevation map $H \\in \\mathbb{R}^{B \\times 1 \\times P \\times P}$. We can compute the outer product between the hyperspectral channel descriptor $f_{hsi} \\in \\mathbb{R}^{B \\times C}$ and the pooled LiDAR spatial elevation descriptor $f_{lidar} \\in \\mathbb{R}^{B \\times 1}$. This directly learns how specific spectral absorption bands correlate with physical surface elevation.")
            ]
        },

        # SLIDE 20
        {
            "num": 20,
            "title": "Conclusion, Acknowledgments & Defense Discussion",
            "time": "30 Seconds + Q&A",
            "speaker": "Tejas Surya",
            "visual": "NIT Raipur closing banner, Supervisor Dr. R.R. Janghel, Team Members, Q&A invite.",
            "spoken": [
                "• <b>Closing Statement:</b> 'In conclusion, SGSCF-Net demonstrates that with principled mathematical module design—targeting band recalibration, complementary gating, and second-order covariance—we can achieve state-of-the-art hyperspectral classification accuracy and solve the minority collapse with only 308K parameters.'",
                "• <b>Acknowledgments:</b> 'We express our deepest gratitude to our project supervisor, Dr. R.R. Janghel, for his invaluable guidance and encouragement throughout this research journey, and to the Department of Information Technology at NIT Raipur for providing the computing resources.'",
                "• <b>Open for Defense:</b> 'Thank you for your time and attention. We are now open and eager to take your questions and defend our work.'"
            ],
            "math": [
                "• <b>Viva Defense Preparedness:</b> Keep Slide 10 (Indian Pines AA vs SSRN), Slide 14 (Parameters), and Slide 15 (Ablations) ready for immediate navigation."
            ],
            "qa": [
                ("What was the biggest technical challenge during your project implementation?",
                 "Stabilizing gradients during second-order covariance computation in SCFG. Without proper normalization, the outer product $f_{ch} \\otimes f_{sp}$ caused exploding gradients in early epochs. Introducing Frobenius norm normalization $\\tilde{G} = G / (\\|G\\|_F + \\varepsilon)$ stabilized the training dynamics completely and enabled smooth convergence.")
            ]
        }
    ]

    for s in slides:
        card = slide_header_card(s["num"], s["title"], s["time"], s["speaker"], s["visual"])
        
        spoken_p = [Paragraph(p, script_bullet) for p in s["spoken"]]
        spoken_box = make_callout(
            "WHAT YOU MUST SAY (Point-Wise Spoken Script)",
            spoken_p,
            bg_color="#F8FAFC",
            border_color="#94A3B8",
            title_color="#1E3A8A"
        )
        
        math_p = [Paragraph(p, math_bullet) for p in s["math"]]
        math_box = make_callout(
            "PRONUNCIATION & MATHEMATICAL CONCEPTS GUIDE",
            math_p,
            bg_color="#F0FDF4",
            border_color="#86EFAC",
            title_color="#166534"
        )
        
        qa_p = []
        for q, a in s["qa"]:
            qa_p.append(Paragraph(f"<b>Examiner:</b> {q}", qa_q_style))
            qa_p.append(Spacer(1, 2))
            qa_p.append(Paragraph(f"<b>Winning Defense Answer:</b> {a}", qa_a_style))
            qa_p.append(Spacer(1, 3))
            
        qa_box = make_callout(
            "VIVA DEFENSE SHIELD (Anticipated Tough Examiner Questions & Winning Answers)",
            qa_p,
            bg_color="#FEF2F2",
            border_color="#FECACA",
            title_color="#991B1B"
        )
        
        # Keep slide section together
        slide_block = [
            card,
            Spacer(1, 4),
            spoken_box,
            Spacer(1, 4),
            math_box,
            Spacer(1, 4),
            qa_box,
            Spacer(1, 8),
            PageBreak()
        ]
        story.extend(slide_block)

    # ==================== MASTER VIVA DEFENSE SURVIVAL KIT ====================
    story.append(Paragraph("MASTER VIVA DEFENSE SURVIVAL KIT: TOP 10 KILLER QUESTIONS", section_title))
    story.append(Paragraph("Be thoroughly prepared for these 10 fundamental questions that external examiners frequently ask during minor project defenses:", body_style))
    story.append(Spacer(1, 8))

    top10_qa = [
        ("1. Why not just use a standard Vision Transformer (ViT) or Swin Transformer for HSI?",
         "Vision Transformers have two major flaws in HSI: (1) Their quadratic self-attention mechanism $\\mathcal{O}(N^2)$ creates severe memory bottlenecks on large 3D hyperspectral cubes; (2) Transformers lack inductive bias and require massive datasets (like ImageNet) to train properly. On limited HSI training splits (10%), ViT heavily overfits, achieving only 90.33% OA on Indian Pines. SGSCF-Net achieves 98.05% OA and 94.82% AA with linear $\\mathcal{O}(N)$ complexity and only 308K parameters."),
        
        ("2. Why did SSRN achieve 98.52% Overall Accuracy on Indian Pines, while SGSCF-Net achieved 98.05%?",
         "This is the single most important defense point. Overall Accuracy (OA) is mathematically biased toward majority classes. Indian Pines has severe class imbalance (Soybean-clean has 954 samples; Oats has only 20). SSRN achieved 98.52% OA by memorizing the large majority classes, but it scored 0.00% on Alfalfa, 0.00% on Grass-pasture-mowed, and 0.00% on Oats—meaning SSRN completely failed on minority classes! In stark contrast, SGSCF-Net scored 95.00% on Alfalfa, 100.00% on Grass-mowed, and 54.17% on Oats, delivering the HIGHEST Average Accuracy (94.82%) across all 9 benchmarked models."),

        ("3. Why did you choose an 11×11 spatial patch size? Why not 7×7 or 15×15?",
         "We proved this empirically through our 60-run hyperparameter grid search. A 9×9 patch underfits because it lacks adequate spatial context. Conversely, patches larger than 11×11 (like 15×15 or 17×17) cause cross-field boundary contamination. In Indian Pines, where spatial resolution is 20m per pixel, an 11×11 patch covers a 220m × 220m ground footprint, which perfectly matches typical agricultural field parcels without bleeding into neighboring crops."),

        ("4. What does the Second-Order Covariance Fusion Gate (SCFG) capture that Global Average Pooling misses?",
         "Global Average Pooling (GAP) only computes the first-order mean of features, treating channels and spatial pixels as mutually independent. SCFG computes the outer product $G = f_{ch} \\otimes f_{sp}$, forming a second-order covariance matrix that captures bilinear interactions: how specific spectral absorption bands vary across spatial locations. This is crucial for distinguishing spectrally similar materials (like asphalt vs bitumen or subtle shadow gradients)."),

        ("5. Why do you use 70% Spread Loss and 30% Focal Loss?",
         "Spread Loss directly maximizes angular margins in feature space using an adaptive scheduled margin $m(r) = 0.2 + 0.7r$. Focal Loss acts as a minority regularizer by applying inverse-frequency weights $w_c = 1/(N_c + \\varepsilon)$ and focusing parameter $\\gamma = 2.0$. In our ablation study, replacing Cross-Entropy with this hybrid loss boosted Overall Accuracy by +0.79%, Average Accuracy by +1.04%, and lifted Oats accuracy from 0.00% to 54.17%."),

        ("6. How did you ensure there is no data leakage between your training and testing sets?",
         "We used stratified sampling with strictly non-overlapping pixel coordinates. Training patches (10%) and testing patches (80%) have disjoint center coordinates. Furthermore, no test samples were ever seen during training or validation, ensuring 100% fair evaluation."),

        ("7. How does CBSA guarantee zero feature redundancy between its two branches?",
         "CBSA enforces the mathematical partition of unity $\\alpha + (1-\\alpha) \\equiv 1$. Because the gating weights sum identically to 1, the two branches compete directly for representation capacity. When spectral fidelity dominates, $\\alpha \\to 1$; when spatial context dominates, $(1-\\alpha) \\to 1$. Redundant overlap between branches is mathematically impossible."),

        ("8. Why does your model have only 308K parameters while SSAN has 6.4 million?",
         "SSAN applies multi-head self-attention and dense projection matrices across all 3D feature dimensions. SGSCF-Net decouples spectral and spatial processing: SBAM (10,225 params), CBSA gating (80 params), and SCFG (82 params) together add only 10,387 parameters (3.37%). By using Global Average Pooling before classification, our classifier head requires only 272 parameters, avoiding multi-million parameter dense layers entirely."),

        ("9. Why did the Oats class collapse to 0% in standard Cross-Entropy?",
         "The Oats class in Indian Pines contains only 20 total labeled pixels. In a 10% training split, only 2 pixels are available for training. Under standard Cross-Entropy, majority classes like Soybeans (954 pixels) produce gradients that are orders of magnitude larger, completely drowning out the gradients from Oats. Our Class-Weighted Focal Loss applies inverse-frequency weighting $w_c \\propto 1/N_c$, boosting minority gradients and allowing Oats to reach 54.17%."),

        ("10. What are the practical real-world applications and deployment feasibility of SGSCF-Net?",
         "With a tiny 1.2 MB checkpoint, linear $\\mathcal{O}(N)$ complexity, and a 1.18-second inference time, SGSCF-Net is directly deployable on embedded edge platforms (such as NVIDIA Jetson Orin on UAVs). Practical applications include precision agriculture (crop health and weed mapping), urban infrastructure monitoring, mineral exploration, and environmental pollution tracking.")
    ]

    for q, a in top10_qa:
        q_box = [
            Paragraph(f"<b>Question:</b> {q}", qa_q_style),
            Spacer(1, 3),
            Paragraph(f"<b>Winning Defense Answer:</b> {a}", qa_a_style)
        ]
        story.append(make_callout(
            "CORE VIVA DEFENSE POINT",
            q_box,
            bg_color="#F8FAFC",
            border_color="#CBD5E1",
            title_color="#0F172A"
        ))
        story.append(Spacer(1, 6))

    # Build PDF
    doc.build(story, canvasmaker=DefenseCanvas)
    print(f"Successfully generated PDF at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    create_script_pdf()
