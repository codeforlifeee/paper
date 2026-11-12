import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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
            self.drawString(40, 752, "SGSCF-Net: Minor Project Presentation Defense Script & Visual Slide Guide")
            self.setFont("Helvetica", 8)
            self.drawRightString(572, 752, "NIT Raipur — Dept. of IT | Sept-Nov 2026")
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

def build_pdf(filename="SGSCF_Net_Minor_Project_Progress_Defense_Script.pdf"):
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
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=18, leading=22,
        textColor=colors.HexColor("#0F172A"), alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10.5, leading=14,
        textColor=colors.HexColor("#1D4ED8"), alignment=1
    )
    meta_style = ParagraphStyle(
        'DocMeta', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=colors.HexColor("#475569"), alignment=1
    )
    section_title = ParagraphStyle(
        'SecTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=colors.HexColor("#0F172A"), spaceBefore=6, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyDark', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=colors.HexColor("#1E293B")
    )
    bold_body = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold')
    script_bullet = ParagraphStyle(
        'ScriptBullet', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=11.5,
        textColor=colors.HexColor("#0F172A"), leftIndent=10, firstLineIndent=-10
    )
    math_bullet = ParagraphStyle(
        'MathBullet', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=11,
        textColor=colors.HexColor("#065F46"), leftIndent=10, firstLineIndent=-10
    )
    qa_q_style = ParagraphStyle(
        'QAQuestion', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=11.5,
        textColor=colors.HexColor("#991B1B")
    )
    qa_a_style = ParagraphStyle(
        'QAAnswer', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=11.5,
        textColor=colors.HexColor("#1E293B"), leftIndent=6
    )

    def make_callout(title, elements, bg_color="#F8FAFC", border_color="#CBD5E1", title_color="#0F172A"):
        header = Paragraph(f"<b>{title}</b>", ParagraphStyle(
            'BoxHeader', fontName='Helvetica-Bold', fontSize=8.5, leading=11.5, textColor=colors.HexColor(title_color)
        ))
        content = [header, Spacer(1, 2)] + elements
        t = Table([[content]], colWidths=[usable_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
            ('BOX', (0,0), (-1,-1), 0.7, colors.HexColor(border_color)),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    def slide_header_card(slide_num, title, timing, speaker, visual_cue):
        header_text = f"<b>SLIDE {slide_num} / 21: {title.upper()}</b>"
        sub_text = f"<b>Time:</b> {timing} &nbsp;|&nbsp; <b>Speaker:</b> {speaker} &nbsp;|&nbsp; <b>Visual Focus:</b> <i>{visual_cue}</i>"
        p1 = Paragraph(header_text, ParagraphStyle('SH1', fontName='Helvetica-Bold', fontSize=9.5, leading=13, textColor=colors.white))
        p2 = Paragraph(sub_text, ParagraphStyle('SH2', fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor("#E2E8F0")))
        t = Table([[p1], [p2]], colWidths=[usable_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1E3A8A")),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    story = []

    # ==================== COVER / TITLE BLOCK ====================
    story.append(Paragraph("NATIONAL INSTITUTE OF TECHNOLOGY, RAIPUR", ParagraphStyle('UnivHead', fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=colors.HexColor("#475569"), alignment=1)))
    story.append(Paragraph("Department of Information Technology — Minor Project Defense & Visual Guide", ParagraphStyle('DeptHead', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.HexColor("#64748B"), alignment=1)))
    story.append(Spacer(1, 6))
    story.append(Paragraph("SGSCF-Net Presentation Defense Script with Slide Visuals", title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Spatial-Guided Spectral-Channel Fusion Network for Hyperspectral Remote Sensing Image Analysis", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Supervisor:</b> Dr. R.R. Janghel, Associate Professor &nbsp;|&nbsp; <b>Presenters:</b> Tejas Surya (23118098), B. Vyshnavi (23118021), Valluri Lahitha (23118106)", meta_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#1E3A8A"), spaceBefore=2, spaceAfter=6))

    # Executive Summary & Table
    story.append(Paragraph("EXECUTIVE PRESENTATION STRATEGY & NUMERICAL TRUTH SHEET", section_title))
    strat_text = (
        "<b>Presentation Structure & Team Division (15 Minutes Total):</b><br/>"
        "• <b>Speaker 1 (Tejas Surya) — Slides 1 to 7 (5.0 min):</b> Title, Outline, Remote Sensing Fundamentals, HSI Principles, Imaging Comparison, Applications.<br/>"
        "• <b>Speaker 2 (B. Vyshnavi) — Slides 8 to 13 (5.0 min):</b> Benchmark Datasets, Feature Extraction Workflow, Deep Learning Limitations, Literature Review (Parts 1 & 2), Research Gaps.<br/>"
        "• <b>Speaker 3 (Valluri Lahitha) — Slides 14 to 21 (5.0 min):</b> Empirical Flaws & Bottlenecks, Research Objectives, Techniques Employed, Proposed Architecture Workflow, Phase-wise Research Plan, Experimental Sweeps, References, Conclusion."
    )
    story.append(Paragraph(strat_text, body_style))
    story.append(Spacer(1, 6))

    table_data = [
        [Paragraph("<b>Metric / Benchmark</b>", bold_body), Paragraph("<b>Indian Pines (AVIRIS)</b>", bold_body), Paragraph("<b>Pavia University (ROSIS)</b>", bold_body), Paragraph("<b>Salinas Valley (AVIRIS)</b>", bold_body)],
        [Paragraph("<b>Spatial Grid / GSD</b>", body_style), Paragraph("145 × 145 px, 20m/pixel", body_style), Paragraph("610 × 340 px, 1.3m/pixel", body_style), Paragraph("512 × 217 px, 3.7m/pixel", body_style)],
        [Paragraph("<b>Spectral Channels</b>", body_style), Paragraph("200 calibrated bands", body_style), Paragraph("103 spectral channels", body_style), Paragraph("204 calibrated bands", body_style)],
        [Paragraph("<b>SGSCF-Net OA (%)</b>", bold_body), Paragraph("<b>98.05%</b>", bold_body), Paragraph("<b>99.82%</b>", bold_body), Paragraph("<b>99.92%</b>", bold_body)],
        [Paragraph("<b>SGSCF-Net AA (%)</b>", bold_body), Paragraph("<b>94.82% (RANK 1 SOTA)</b>", bold_body), Paragraph("<b>99.71%</b>", bold_body), Paragraph("<b>99.92%</b>", bold_body)],
        [Paragraph("<b>Minority Accuracy</b>", body_style), Paragraph("Oats: 54.17% (SSRN: 0.0%)", body_style), Paragraph("Gravel: 99.76% (SSAN: 17.9%)", body_style), Paragraph("11 of 16 classes at 100.0%", body_style)],
        [Paragraph("<b>Parameters & Checkpoint</b>", bold_body), Paragraph("<b>308,097 params (1.2 MB)</b>", bold_body), Paragraph("3 Attention modules = 3.37%", body_style), Paragraph("vs SSAN: 6.40M (20.8× smaller)", body_style)]
    ]
    t_summary = Table(table_data, colWidths=[120, 140, 136, 136])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 8))
    story.append(PageBreak())

    # ==================== SLIDE DEFINITIONS ====================
    img_dir = r"C:\Users\LENOVO\Desktop\MinorProject_Final\extracted_slides_tejas"

    slides = [
        # SLIDE 1
        {
            "num": 1,
            "title": "Title Slide & Project Overview",
            "time": "45 Seconds",
            "speaker": "Tejas Surya",
            "visual": "NIT Raipur bilingual banner, Project title, Dr. R.R. Janghel, Student roll numbers.",
            "spoken": [
                "• <b>Opening Greeting:</b> 'Respected supervisor Dr. R.R. Janghel sir, and distinguished members of the evaluation committee, a very good morning.'",
                "• <b>Project Declaration:</b> 'Today, on behalf of our team comprising Tejas Surya, B. Vyshnavi, and Valluri Lahitha, I present our Minor Project: <i>Spatial-Guided Spectral-Channel Fusion Network for Hyperspectral Remote Sensing Image Analysis</i> (SGSCF-Net).'",
                "• <b>Core Motivation:</b> 'Hyperspectral imaging captures hundreds of narrow, continuous spectral bands, enabling material identification with spectroscopic accuracy. However, existing deep models suffer from severe overfitting, heavy compute exceeding 85 million parameters, and minority class collapse.'",
                "• <b>Project Goal:</b> 'In this presentation, we outline the foundational theory, systematic literature analysis, architectural bottlenecks, and our proposed SGSCF-Net design of only 308K parameters that resolves these challenges.'"
            ],
            "math": [
                "• <b>SGSCF-Net Pronunciation:</b> 'S-G-S-C-F Net' (Spatial-Guided Spectral-Channel Fusion Network).",
                "• <b>Hyperspectral (HSI):</b> Imaging across contiguous narrow wavelengths ($400 - 2500$ nm)."
            ],
            "qa": [
                ("What is the primary motivation for selecting Hyperspectral Image Classification?",
                 "HSI bridges 2D visual imagery with laboratory spectroscopy. Unlike RGB imagery which only captures 3 broad bands, HSI captures 100 to 200+ narrow continuous bands, enabling identification of crop species, mineral compositions, and urban materials that appear identical to the naked eye. However, high dimensionality combined with limited labeled data makes it an important research challenge.")
            ]
        },

        # SLIDE 2
        {
            "num": 2,
            "title": "Presentation Outline",
            "time": "30 Seconds",
            "speaker": "Tejas Surya",
            "visual": "9-point structured presentation agenda.",
            "spoken": [
                "• <b>Agenda Walkthrough:</b> 'Here is the roadmap for our presentation today.'",
                "• <b>Foundations to Review:</b> 'We begin with an introduction to remote sensing and hyperspectral principles, followed by a comprehensive review of 8 state-of-the-art literature methods.'",
                "• <b>Bottlenecks to Architecture:</b> 'Next, we examine the empirical flaws and research gaps identified, followed by our proposed techniques: SBAM, CBSA, SCFG, and our Spread-Focal loss.'",
                "• <b>Research Execution:</b> 'Finally, we detail our proposed architecture workflow, phase-wise research milestones, experimental sensitivity plan, and references.'"
            ],
            "math": [
                "• <b>Terminology:</b> SOTA = State-of-the-Art; Empirical Bottlenecks = Concrete failure modes observed in existing benchmarked codebases."
            ],
            "qa": [
                ("Why is a thorough literature review essential before proposing an architecture?",
                 "To avoid incremental guesswork. By systematically categorizing 3D-CNNs, Transformers, State Space Models, and Capsule networks, we isolate the exact mathematical causes of failure—such as hardware locks, static tensor crashes, and minority collapse—and design specific mathematical modules to solve each.")
            ]
        },

        # SLIDE 3
        {
            "num": 3,
            "title": "Introduction - Remote Sensing",
            "time": "45 Seconds",
            "speaker": "Tejas Surya",
            "visual": "Figure 1: Energy radiation, material interaction, sensor capture, algorithm processing.",
            "spoken": [
                "• <b>Definition:</b> 'Remote sensing is the science of acquiring physical and chemical information about the Earth’s surface from a distance, without direct physical contact.'",
                "• <b>Sensor Platforms:</b> 'Data is collected across multiple platforms: low-Earth orbit satellites, airborne aircraft, and uncrewed aerial vehicles (UAVs or drones).'",
                "• <b>Four-Step Physical Process:</b> 'The process follows four stages: (1) An energy source—typically the Sun—emits electromagnetic radiation; (2) The radiation interacts with Earth surface materials, undergoing wavelength-specific absorption and reflection; (3) Optical sensors capture the backscattered radiance; (4) Deep learning algorithms classify the pixels into land-cover maps.'",
                "• <b>Significance:</b> 'Remote sensing enables macro-scale monitoring of agriculture, forestry, disaster management, and urban expansion.'"
            ],
            "math": [
                "• <b>Passive vs. Active Sensing:</b> Passive sensors measure reflected solar radiance ($L_\\lambda$); active sensors (like LiDAR/Radar) emit their own coherent pulses."
            ],
            "qa": [
                ("What type of remote sensing is hyperspectral imaging?",
                 "HSI is primarily passive optical remote sensing, operating in the Solar Reflective region from the visible (400 nm) through Near-Infrared (NIR, 700-1000 nm) to Short-Wave Infrared (SWIR, 1000-2500 nm).")
            ]
        },

        # SLIDE 4
        {
            "num": 4,
            "title": "Hyperspectral Imaging (HSI)",
            "time": "50 Seconds",
            "speaker": "Tejas Surya",
            "visual": "Figure 2: Hypercube (X, Y, Lambda) and continuous spectral reflection curves.",
            "spoken": [
                "• <b>Core Definition:</b> 'Hyperspectral Imaging (HSI) captures electromagnetic energy across hundreds of narrow, contiguous spectral bands, typically between 100 and 300+ channels.'",
                "• <b>The 3D Data Cube:</b> 'The sensor output is a 3D volume or data cube of dimensions (X, Y, λ), where X and Y are spatial coordinates, and λ represents the continuous spectral wavelength dimension.'",
                "• <b>The Spectral Fingerprint:</b> 'Every spatial pixel in this cube contains a continuous spectral reflectance curve. Because different materials have distinct chemical bonds, their absorption features act as unique spectral fingerprints.'",
                "• <b>Key Advantage:</b> 'This allows us to separate materials that look visually identical in conventional RGB photography—such as distinguishing different tree species, plastics, or crop maturity stages.'"
            ],
            "math": [
                "• <b>Data Cube Coordinates:</b> $X(x, y, \\lambda)$ where $(x,y)$ are spatial 2D coordinates and $\\lambda \\in [\\lambda_{min}, \\lambda_{max}]$ is the wavelength coordinate."
            ],
            "qa": [
                ("What is a spectral signature and what causes it?",
                 "A spectral signature is the reflectance percentage of an object as a function of wavelength. It is governed by molecular vibrational overtones and electronic transitions in chemical bonds (e.g., chlorophyll absorption in the red band, water absorption in the SWIR).")
            ]
        },

        # SLIDE 5
        {
            "num": 5,
            "title": "Working Principles of HSI",
            "time": "50 Seconds",
            "speaker": "Tejas Surya",
            "visual": "Prism optical dispersion diagram, band splitting, and 3D volume stacking.",
            "spoken": [
                "• <b>Physical Dispersion:</b> 'How does an HSI sensor physically generate this data cube? Incident light reflected from a target enters an imaging spectrometer.'",
                "• <b>Optical Splitting:</b> 'An internal dispersing element—such as a diffraction grating or prism—splits the incoming broadband light into discrete, narrow wavelengths.'",
                "• <b>2D Detector Array:</b> 'A linear slit projects one spatial line across the detector while the orthogonal axis captures the dispersed spectral bands, scanning row-by-row in a pushbroom manner.'",
                "• <b>Hypercube Assembly:</b> 'These stacked 2D spatial frames at contiguous wavelengths form the 3D hypercube. X and Y provide spatial context, while the Z-axis provides rich spectroscopic depth.'"
            ],
            "math": [
                "• <b>Pushbroom Scanning:</b> Captures an entire spatial line simultaneously, dispersing wavelengths across a 2D focal plane array (FPA)."
            ],
            "qa": [
                ("Why does the 3D structure of an HSI cube create challenges for standard 2D convolutions?",
                 "Standard 2D convolutions collapse all spectral channels into a single depth sum, treating them like arbitrary color channels. This obliterates continuous spectral curvature and inter-band absorption slopes. True spectral preservation requires 3D convolutions or targeted spectral attention.")
            ]
        },

        # SLIDE 6
        {
            "num": 6,
            "title": "Imaging Techniques Comparison",
            "time": "60 Seconds",
            "speaker": "Tejas Surya",
            "visual": "Comparative table: RGB vs Multispectral (MSI) vs Hyperspectral (HSI) vs Ultraspectral.",
            "spoken": [
                "• <b>Comparative Spectrum:</b> 'This slide provides a clear comparison between RGB imaging, Multispectral Imaging (MSI), and Hyperspectral Imaging (HSI).'",
                "• <b>Number of Bands:</b> 'RGB has only 3 broad bands (Red, Green, Blue). MSI captures 4 to 20 discrete bands. HSI captures 100 to 300+ contiguous bands.'",
                "• <b>Spectral Resolution:</b> 'RGB has broad resolution (~100 nm). MSI has medium resolution (~30–50 nm) with wide unmeasured gaps. HSI has high resolution (~1–10 nm) contiguous sampling.'",
                "• <b>Data Complexity:</b> 'RGB is a simple 2D array; MSI is moderate; HSI is an extremely dense 3D volume requiring high-performance processing.'",
                "• <b>Use Case Distinction:</b> 'RGB is limited to visual inspection. MSI is suitable for broad vegetation indices like NDVI. HSI is required for precision agriculture, mineralogy, and chemical differentiation.'"
            ],
            "math": [
                "• <b>Spectral Sampling Interval:</b> $\\Delta \\lambda \\le 10\\text{ nm}$ in HSI ensures continuous absorption line capture."
            ],
            "qa": [
                ("Why can't Multispectral Imaging (like Sentinel-2) identify subtle crop diseases?",
                 "Multispectral sensors have wide spectral gaps between bands (e.g., 50 to 100 nm). Early crop stress causes subtle shifts in the 'red edge' slope (690–740 nm) and water absorption depth at 970 nm, which fall completely between MSI bands. HSI samples continuously at 10 nm intervals, detecting stress weeks earlier.")
            ]
        },

        # SLIDE 7
        {
            "num": 7,
            "title": "Applications of Hyperspectral Imaging",
            "time": "45 Seconds",
            "speaker": "Tejas Surya",
            "visual": "Application chart: Agriculture, Medical, Environment, Recycling, Food, Pharmaceuticals.",
            "spoken": [
                "• <b>Multidisciplinary Scope:</b> 'Hyperspectral imaging has revolutionized diverse scientific and industrial fields, shown here in six key sectors.'",
                "• <b>Precision Agriculture:</b> 'In agriculture, HSI detects nitrogen deficiency, soil salinity, moisture stress, and weed infestations at the single-plant level.'",
                "• <b>Environmental & Mineralogy:</b> 'It enables automated mapping of mineral deposits, tracking oil spill dispersion, and quantifying greenhouse gas plumes.'",
                "• <b>Medicine & Industry:</b> 'In medicine, HSI assists in non-invasive cancer tissue margin detection; in manufacturing, it inspects pharmaceutical tablet uniformity and sorts recyclable plastics by polymer type.'",
                "• <b>Our Project Focus:</b> 'Our work focuses on precision agriculture and urban scene classification, validated across standard international benchmarks.'"
            ],
            "math": [
                "• <b>Biochemical Indicators:</b> Chlorophyll absorption ($670\\text{ nm}$), Carotenoids ($450\\text{ nm}$), Water absorption ($970, 1200, 1450\\text{ nm}$)."
            ],
            "qa": [
                ("Which specific datasets in your study correspond to these applications?",
                 "Indian Pines and Salinas Valley represent precision agriculture (crop type and maturity stage classification). Pavia University represents urban infrastructure and environmental mapping (differentiating asphalt, gravel, bitumen, and vegetation).")
            ]
        },

        # SLIDE 8
        {
            "num": 8,
            "title": "Datasets & Ground Truth Specification",
            "time": "60 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "Dataset table: Indian Pines, Pavia University, Salinas Scene parameters.",
            "spoken": [
                "• <b>Dataset Overview:</b> 'Thank you, Tejas. I will now present our benchmark datasets and experimental workflows.'",
                "• <b>Indian Pines (AVIRIS):</b> 'Captured by AVIRIS over Northwest Indiana. Spatial resolution is 20m/pixel across a 145×145 grid. It has 200 calibrated spectral bands (20 atmospheric bands removed), 16 agricultural classes, and 10,249 labeled samples. It features extreme class imbalance, from 954 samples down to only 20 samples for Oats.'",
                "• <b>Pavia University (ROSIS):</b> 'Captured by ROSIS over Pavia, Italy. Spatial resolution is high at 1.3m/pixel across a 610×340 grid, with 103 spectral channels, 9 urban classes, and 42,776 labeled samples.'",
                "• <b>Salinas Valley (AVIRIS):</b> 'Captured by AVIRIS over Salinas Valley, California. Spatial resolution is 3.7m/pixel across a 512×217 grid, with 204 calibrated bands, 16 agricultural classes, and 54,129 labeled samples, including fine-grained crop maturity stages.'"
            ],
            "math": [
                "• <b>Atmospheric Water Absorption Bands:</b> AVIRIS bands 104–108 ($1.4\\,\\mu\\text{m}$), 150–163 ($1.9\\,\\mu\\text{m}$), and band 220 are omitted due to near-zero atmospheric transmission."
            ],
            "qa": [
                ("Why is Indian Pines considered the most difficult HSI benchmark?",
                 "Indian Pines combines low spatial resolution (20m/pixel, resulting in severe mixed-pixel effects) with extreme class imbalance: the largest class has 954 samples, while the Oats class has only 20 samples. Most models collapse to 0% on Oats.")
            ]
        },

        # SLIDE 9
        {
            "num": 9,
            "title": "HSI Feature Extraction Workflow",
            "time": "45 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "5-step chevron diagram: Acquisition -> Preprocessing -> Feature Extraction -> Training -> Optimization.",
            "spoken": [
                "• <b>Standard Pipeline:</b> 'This diagram illustrates the classical HSI processing workflow from raw sensor acquisition to final thematic classification.'",
                "• <b>Stages 1 & 2 — Acquisition & Preprocessing:</b> 'Raw sensor radiances are converted to surface reflectance via atmospheric correction. Noisy water absorption bands are removed.'",
                "• <b>Stages 3 & 4 — Feature Extraction & Training:</b> 'Spatial and spectral neighborhood patches are extracted and passed to deep learning models to learn discriminative material representations.'",
                "• <b>Stage 5 — Optimization & Loss:</b> 'Model weights are iteratively optimized using loss gradients to minimize classification error across all classes.'",
                "• <b>Limitations of Standard Flow:</b> 'Traditional pipelines often use linear PCA dimensionality reduction, which discards valuable non-linear spectral signatures. In SGSCF-Net, we replace manual PCA with end-to-end attention.'"
            ],
            "math": [
                "• <b>PCA Formulation:</b> Projects $X \\in \\mathbb{R}^{N \\times C}$ onto principal eigenvectors of covariance $\\Sigma = \\frac{1}{N}X^TX$, retaining top $k$ variance components."
            ],
            "qa": [
                ("Why does PCA dimensionality reduction sometimes degrade classification?",
                 "PCA is an unsupervised linear projection that maximizes overall variance. However, high-variance bands often correspond to broad illumination changes rather than material differences. Subtle absorption features of rare minority classes have low overall variance and are discarded during PCA truncation.")
            ]
        },

        # SLIDE 10
        {
            "num": 10,
            "title": "Limitations of Deep Learning in HSI",
            "time": "60 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "4 limitation cards: Hughes Phenomenon, Memory Explosion, Feature Entanglement, Class Imbalance.",
            "spoken": [
                "• <b>Four Major Bottlenecks:</b> 'Applying standard deep learning directly to hyperspectral data encounters four fundamental bottlenecks.'",
                "• <b>1. The Hughes Phenomenon:</b> 'High spectral dimensionality combined with very few labeled training pixels causes exponential parameter explosion and severe overfitting.'",
                "• <b>2. Memory Explosion:</b> 'Standard 3D-CNNs and Vision Transformers require tens of millions of parameters (exceeding 85M+ in models like ViT), exhausting GPU VRAM.'",
                "• <b>3. Feature Entanglement:</b> 'Simultaneous 3D convolutions mix spatial and spectral features indiscriminately, blurring fine absorption signatures with spatial textures.'",
                "• <b>4. Severe Class Imbalance:</b> 'Standard Cross-Entropy loss is dominated by frequent classes, leading to the total collapse of minority classes (0% accuracy on rare crops).'"
            ],
            "math": [
                "• <b>Hughes Phenomenon Curve:</b> Accuracy peaks at optimal dimension $d^*$, then degrades: $\\lim_{d \\to \\infty} \\text{Accuracy} = \\text{Random Guess}$ for fixed $N$ samples."
            ],
            "qa": [
                ("What causes feature entanglement in standard 3D-CNNs?",
                 "Standard 3D convolutions apply kernels of size $K_h \\times K_w \\times K_d$ that compute weighted sums across both spatial coordinates and spectral channels simultaneously in every layer. This prevents the network from isolating pure spectral chemical signatures independently of local spatial textures.")
            ]
        },

        # SLIDE 11
        {
            "num": 11,
            "title": "Literature Review (Part 1: Papers 1 to 4)",
            "time": "75 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "Table: SSRN (2018), SSAN (2021), ViT (2021), A2S2K-ResNet (2021) with results & research gaps.",
            "spoken": [
                "• <b>Systematic Literature Analysis:</b> 'We conducted a comprehensive literature review of representative state-of-the-art models, beginning with four foundational works.'",
                "• <b>1. SSRN (Zhong et al., 2018, IEEE TGRS):</b> 'SSRN introduced consecutive 3D and 2D residual blocks. However, heavy 3D convolutions cause massive computational overhead and severe overfitting on small sample sizes (<1%), with zero dynamic noise filtering.'",
                "• <b>2. SSAN (Sun et al., 2021, IEEE TGRS):</b> 'SSAN used RNNs and CNNs with attention modules. However, RNN recurrence causes high inference latency, sequential attention misses joint correlations, and minority classes collapse.'",
                "• <b>3. ViT (Dosovitskiy et al., 2021, ICLR):</b> 'Vision Transformers use global self-attention. But with ~85M parameters, ViT suffers from data starvation and severe overfitting on limited HSI training samples.'",
                "• <b>4. A2S2K-ResNet (Roy et al., 2021, IEEE TGRS):</b> 'A2S2K-ResNet introduced adaptive 3D kernels. However, kernel generation causes high memory spikes, and branches duplicate features due to lack of complementary gating.'"
            ],
            "math": [
                "• <b>SSRN Receptive Field:</b> Uses $3 \\times 3 \\times 7$ kernels in 3D residual blocks, yielding high parameter count $\\mathcal{O}(K_h K_w K_d \\cdot C_{in} C_{out})$."
            ],
            "qa": [
                ("Why do Transformers like ViT struggle on hyperspectral datasets?",
                 "Transformers lack inductive biases (such as translation equivariance and local neighborhood locality) inherent to convolutions. They require millions of training images (like ImageNet-21k) to learn spatial relationships from scratch. HSI datasets typically have only 1,000 to 5,000 labeled pixels, causing severe overfitting.")
            ]
        },

        # SLIDE 12
        {
            "num": 12,
            "title": "Literature Review (Part 2: Papers 5 to 8)",
            "time": "75 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "Table: DMC (2021), SSFTT (2022), MambaHSI (2024), CSCANet (2025) with results & research gaps.",
            "spoken": [
                "• <b>Continuing the Review:</b> 'Continuing to recent advances from 2021 to 2025:'",
                "• <b>5. Deep Matrix Capsules (Paoletti et al., 2021, IEEE TGRS):</b> 'DMC uses capsule routing with EM algorithms. However, iterative EM routing is exceptionally slow, rendering real-time inference impossible.'",
                "• <b>6. SSFTT (Sun et al., 2022, IEEE TGRS):</b> 'SSFTT introduced feature tokenization before a Transformer encoder. While effective, tokenization consumes heavy GPU VRAM, restricting batch sizes.'",
                "• <b>7. MambaHSI (Li et al., 2024, IEEE TGRS):</b> 'MambaHSI applied Selective State Space Models for linear scaling. However, it requires custom, non-portable C++ CUDA extensions (mamba_ssm), locking it out of edge hardware.'",
                "• <b>8. CSCANet (Wang et al., 2025, IEEE TIP):</b> 'CSCANet introduced cross-attention between spatial and channel maps. However, it hardcodes tensor dimensions (such as BATCH_SIZE=64), crashing on remainder validation batches.'",
                "• <b>Summary:</b> 'None of these architectures achieve the desired balance of ultra-low parameter footprint, minority robustness, and hardware portability.'"
            ],
            "math": [
                "• <b>Selective State Space (Mamba):</b> $h'(t) = A h(t) + B x(t); \\quad y(t) = C h(t)$ discretized with input-dependent $B, C, \\Delta$."
            ],
            "qa": [
                ("What is the primary architectural takeaway from reviewing these 8 papers?",
                 "The literature demonstrates that complex attention matrices, capsule routing, and non-standard CUDA kernels introduce severe practical drawbacks. The optimal solution is a mathematically principled, pure PyTorch 3D-CNN that uses lightweight complementary gating and second-order covariance—achieving SOTA accuracy with minimal parameters.")
            ]
        },

        # SLIDE 13
        {
            "num": 13,
            "title": "Research Gaps Identified",
            "time": "50 Seconds",
            "speaker": "B. Vyshnavi",
            "visual": "5 synthesized research gap cards.",
            "spoken": [
                "• <b>Synthesis of Gaps:</b> 'From our literature review, we distilled five fundamental open research gaps:'",
                "• <b>Gap 1:</b> 'Heavy dependence on pre-segmentation or superpixels, whose boundary quality severely bottlenecks downstream classification.'",
                "• <b>Gap 2:</b> 'Isolated processing: models focus on either spectral or spatial features independently, missing joint cross-domain relationships.'",
                "• <b>Gap 3:</b> 'Ignoring long-range spatial context in favor of purely local $3 \\times 3$ patches.'",
                "• <b>Gap 4:</b> 'Mixed pixels and spectral heterogeneity within parcels causing boundary feature distortion.'",
                "• <b>Gap 5:</b> 'Total collapse of supervised models under small-sample, label-scarce regimes (<1% training data).'"
            ],
            "math": [
                "• <b>Mixed Pixels:</b> $y = \\sum_{i=1}^M a_i s_i + n$ where $a_i$ is fractional abundance and $s_i$ is endmember signature."
            ],
            "qa": [
                ("How does your project address Gap 2 (joint cross-domain relationships)?",
                 "Through our Second-Order Covariance Fusion Gate (SCFG), which computes the outer product between the channel descriptor and spatial descriptor, capturing joint bilinear covariance rather than processing them in isolation.")
            ]
        },

        # SLIDE 14
        {
            "num": 14,
            "title": "Empirical Flaws & Architectural Bottlenecks",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "5 empirical flaw cards: Hardware Locks, Static Tensor Crashes, Unfiltered Noise, Branch Redundancy, Minority Collapse.",
            "spoken": [
                "• <b>Concrete Engineering Failures:</b> 'Thank you, Vyshnavi. Beyond theoretical gaps, our hands-on implementation and benchmarking identified five severe empirical flaws in existing codebases.'",
                "• <b>1. Hardware Locks:</b> 'Models like MambaHSI fail on standard laptops and edge devices due to non-portable C++ CUDA extensions.'",
                "• <b>2. Static Tensor Crashes:</b> 'Architectures like CSCANet hardcode `BATCH_SIZE=64`, crashing during validation when test batch sizes don’t divide evenly.'",
                "• <b>3. Unfiltered Noise:</b> 'Feeding raw bands directly into dense filters allows atmospheric water noise to propagate through all layers.'",
                "• <b>4. Branch Redundancy:</b> 'Multi-stream networks learn duplicated features because their parallel branches lack complementary constraints.'",
                "• <b>5. Minority Collapse:</b> 'Standard models yield 0.0% accuracy on rare classes like Oats under small training ratios.'",
                "• <b>Takeaway:</b> 'Our proposed SGSCF-Net architecture is systematically engineered to eliminate each of these five empirical flaws.'"
            ],
            "math": [
                "• <b>Static Batch Crash Cause:</b> Dimension mismatch when dataset size $N \\not\\equiv 0 \\pmod B$ and code uses static reshape `view(64, ...)`."
            ],
            "qa": [
                ("How does SGSCF-Net avoid 'Hardware Locks'?",
                 "SGSCF-Net is implemented in 100% native PyTorch using standard 3D and 1D convolutions, average pooling, and matrix operations. It requires zero custom C++ CUDA extensions, ensuring out-of-the-box portability across any GPU, CPU, or edge NPU.")
            ]
        },

        # SLIDE 15
        {
            "num": 15,
            "title": "Research Objectives",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "Primary Objective banner + Objective 1 (Noise/Redundancy) & Objective 2 (Calibration/Balance).",
            "spoken": [
                "• <b>Primary Objective:</b> 'Our primary objective is to construct SGSCF-Net: an ultra-lightweight architecture of only 308K parameters that outperforms SOTA baselines while resolving minority class collapse.'",
                "• <b>Objective 1 — Noise & Redundancy Reduction:</b> 'Deploy the Spectral Band Attention Module (SBAM) for adaptive band weighting to filter atmospheric noise before convolution, paired with Complementary Branch Spectral Attention (CBSA) using α-probabilities to prevent branch duplication.'",
                "• <b>Objective 2 — Calibration & Class Balance:</b> 'Integrate the Second-Order Covariance Fusion Gate (SCFG) for dynamic channel calibration via outer products, paired with a Joint Spread-Focal Loss to guarantee accuracy on minority classes.'"
            ],
            "math": [
                "• <b>Target Parameters:</b> $\\approx 308\\text{K parameters}$ ($\approx 1.2\\text{ MB}$ checkpoint), representing a $20.8\\times$ reduction compared to SSAN (6.40M)."
            ],
            "qa": [
                ("What is the core research novelty of your proposed objectives?",
                 "The novelty lies in three mathematical mechanisms: (1) Input-level band recalibration via SBAM; (2) The complementary constraint $\\alpha + (1-\\alpha) = 1$ in CBSA ensuring zero redundancy; (3) Second-order covariance outer product in SCFG replacing heavy fully-connected layers.")
            ]
        },

        # SLIDE 16
        {
            "num": 16,
            "title": "Techniques Employed in SGSCF-Net",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "4 technique cards: SBAM, CBSA, SCFG, Spread-Focal Loss.",
            "spoken": [
                "• <b>Four Core Techniques:</b> 'This slide presents the four foundational techniques powering SGSCF-Net.'",
                "• <b>1. SBAM:</b> 'Attenuates atmospheric water absorption noise using spectral-band attention on raw input cubes before any convolution.'",
                "• <b>2. CBSA:</b> 'Fuses complementary narrow (1×1×7) and wide (3×3×7) 3D branches to learn non-overlapping spectral features, strictly constrained by α and (1−α).'",
                "• <b>3. SCFG:</b> 'Uses second-order covariance-based calibration via outer products, eliminating dense fully-connected layers and reducing classifier parameters by over 85%.'",
                "• <b>4. Spread-Focal Loss:</b> 'Combines dynamic margin scheduling with inverse-frequency class weighting, preventing minority class feature collapse.'"
            ],
            "math": [
                "• <b>Technique Parameters:</b> SBAM (10,225 params), CBSA (80 params), SCFG (82 params) $\\to$ 3 novel modules total only 10,387 params (3.37%)."
            ],
            "qa": [
                ("Why does CBSA use two different kernel sizes (1×1×7 and 3×3×7)?",
                 "The $1 \\times 1 \\times 7$ kernel spans 7 contiguous spectral bands with zero spatial context, capturing pure spectral transitions. The $3 \\times 3 \\times 7$ kernel captures joint spatial texture and spectral patterns. The complementary gate $\\alpha$ balances between spectral fidelity and spatial context.")
            ]
        },

        # SLIDE 17
        {
            "num": 17,
            "title": "Proposed Architecture Workflow",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "Complete horizontal workflow: Patch X -> Stage 1 SBAM -> Stage 2 CBSA -> Stage 3 SCFG -> Stage 4 Head & Loss.",
            "spoken": [
                "• <b>Workflow Overview:</b> 'Here we see the complete workflow of our proposed SGSCF-Net architecture.'",
                "• <b>Stage 1 — Input & SBAM:</b> 'The input patch X enters Stage 1, where SBAM removes atmospheric and sensor noise via spatial average pooling and MLP bottleneck.'",
                "• <b>Stage 2 — Dual-Branch CBSA:</b> 'Stage 2 extracts complementary representations: the spatial branch captures context, while the spectral branch captures absorption lines, fused via α-gating and deepened via Spectral ResBlocks with ECA.'",
                "• <b>Stage 3 — SCFG Covariance Transition:</b> 'Stage 3 computes the outer product $G = f_{ch} \\otimes f_{sp}$, normalized by the Frobenius norm, calibrating features for the spatial ResBlocks.'",
                "• <b>Stage 4 — Classification Head:</b> 'Global Average Pooling collapses the spatial grid, followed by a linear layer trained under Joint Spread-Focal Loss.'"
            ],
            "math": [
                "• <b>Tensor Evolution:</b> $X \\in \\mathbb{R}^{B \\times 1 \\times P \\times P \\times C} \\xrightarrow{\\text{SBAM}} \\tilde{X} \\xrightarrow{\\text{CBSA}} v \\xrightarrow{\\text{ResSpec}} X_{spec} \\xrightarrow{\\text{SCFG}} X_{spat} \\xrightarrow{\\text{GAP+FC}} \\hat{y} \\in \\mathbb{R}^{B \\times C_{cls}}$."
            ],
            "qa": [
                ("How does Stage 4 prevent overfitting in the classifier head?",
                 "Standard models use 2 or 3 dense fully-connected layers (e.g., $1024 \\times 512 \\times 16$), creating millions of weights. We apply Global Average Pooling across the final spatial cube, reducing the feature map to a single 16-element vector. The classifier head requires only $16 \\times 16 + 16 = 272$ parameters.")
            ]
        },

        # SLIDE 18
        {
            "num": 18,
            "title": "Phase-Wise Research Plan",
            "time": "45 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "4-phase execution timeline: Preprocessing, Architecture Implementation, Sweeps & Tuning, Validation.",
            "spoken": [
                "• <b>Four Execution Phases:</b> 'Our project followed a structured, phase-wise research plan.'",
                "• <b>Phase 1 — Preprocessing:</b> 'Dataset extraction, atmospheric band filtering, and stratified spatial patch extraction with non-overlapping coordinates.'",
                "• <b>Phase 2 — Architecture Implementation:</b> 'Custom PyTorch design of SBAM, CBSA dual-stream fusion, SCFG covariance gating, and the Spread-Focal loss function.'",
                "• <b>Phase 3 — Sweeps & Tuning:</b> 'Systematic sensitivity sweeps across patch sizes (9 to 17), batch sizes (32 to 256), and data scarcity stress testing (0.2% to 10%).'",
                "• <b>Phase 4 — Validation & Documentation:</b> 'Benchmarking against 8 SOTA baselines from scratch, minority class breakdown, defense preparation, and paper writing.'"
            ],
            "math": [
                "• <b>Milestone Progress:</b> All 4 phases successfully completed with 120+ GPU training runs executed."
            ],
            "qa": [
                ("What was the most challenging part of Phase 2?",
                 "Implementing the second-order covariance in SCFG with proper gradient backpropagation. Raw outer products caused exploding gradients until we formulated the Frobenius norm normalization $\\tilde{G} = G / (\\|G\\|_F + \\varepsilon)$, which stabilized training.")
            ]
        },

        # SLIDE 19
        {
            "num": 19,
            "title": "Experimental Plan & Sensitivity Sweeps",
            "time": "60 Seconds",
            "speaker": "Valluri Lahitha",
            "visual": "Experimental plan card: Environment, Patch sweep, Batch sweep, Scarcity testing, Minority tracking.",
            "spoken": [
                "• <b>Experimental Design:</b> 'This slide presents our rigorous experimental evaluation design.'",
                "• <b>Environment:</b> 'Implemented in PyTorch, trained on NVIDIA GPUs for 200 epochs using the Adam optimizer with Cosine Annealing learning rate schedule.'",
                "• <b>Patch Size Sweep:</b> 'Evaluating 9×9, 11×11, 13×13, 15×15, and 17×17 spatial windows to balance spatial context against boundary noise.'",
                "• <b>Batch Size Sweep:</b> 'Testing batch sizes 32, 64, 128, and 256 to evaluate gradient variance and regularization.'",
                "• <b>Data Scarcity Stress Testing:</b> 'Evaluating resilience under extreme label scarcity (0.2%, 1%, 5%, and 10% training ratios).'",
                "• <b>Minority Validation:</b> 'Strict monitoring of ultra-minority classes: Oats (20 samples) and Mowed-Grass (28 samples).'"
            ],
            "math": [
                "• <b>Cosine Annealing Schedule:</b> $\\eta_t = 10^{-6} + \\frac{1}{2}(5\\times 10^{-4} - 10^{-6})(1 + \\cos(\\frac{t}{200}\\pi))$."
            ],
            "qa": [
                ("What was the result of the patch size sweep?",
                 "Patch 11×11 emerged as globally optimal across all 3 datasets. A 9×9 patch lacks sufficient spatial context, while patches larger than 11×11 cause cross-field boundary contamination (at 20m/pixel in Indian Pines, 11×11 spans 220m × 220m ground footprint).")
            ]
        },

        # SLIDE 20
        {
            "num": 20,
            "title": "References",
            "time": "30 Seconds",
            "speaker": "Tejas Surya",
            "visual": "8 seminal IEEE/ICLR citations (SSRN, SSAN, ViT, A2S2K-ResNet, DMC, SSFTT, MambaHSI, CSCANet).",
            "spoken": [
                "• <b>Scholarly Foundation:</b> 'Thank you, Lahitha. This slide lists the eight primary benchmark references from top IEEE journals and conferences that grounded our research.'",
                "• <b>Literature Pedigree:</b> 'Our baselines include Zhong et al. for SSRN (2018), Sun et al. for SSAN (2021), Dosovitskiy et al. for ViT (2021), Roy et al. for A2S2K-ResNet (2021), Paoletti et al. for DMC (2021), Sun et al. for SSFTT (2022), Li et al. for MambaHSI (2024), and Wang et al. for CSCANet (2025).'",
                "• <b>Fairness:</b> 'All benchmark comparison metrics cited in our work were obtained by retraining these exact architectures under our standardized 10/10/80 protocol.'"
            ],
            "math": [
                "• <b>All citations:</b> Published in IEEE TGRS, IEEE TIP, and ICLR."
            ],
            "qa": [
                ("Did you rely on numbers reported in the original papers or rerun them?",
                 "We retrained all 8 baselines completely from scratch under identical 10% training, 10% validation, and 80% testing splits. Reported paper numbers often use overlapping patches or different splits, making direct comparison unfair.")
            ]
        },

        # SLIDE 21
        {
            "num": 21,
            "title": "Conclusion & Thank You",
            "time": "30 Seconds + Q&A",
            "speaker": "Tejas Surya",
            "visual": "NIT Raipur closing slide, Dr. R.R. Janghel, Student team, Q&A invitation.",
            "spoken": [
                "• <b>Concluding Synthesis:</b> 'In conclusion, SGSCF-Net demonstrates that with principled mathematical module design—targeting band recalibration, complementary gating, and second-order covariance—we can achieve state-of-the-art hyperspectral classification accuracy and solve the minority collapse with only 308K parameters.'",
                "• <b>Acknowledgments:</b> 'We express our deepest gratitude to our supervisor, Dr. R.R. Janghel, for his continuous guidance and encouragement, and to the Department of Information Technology at NIT Raipur.'",
                "• <b>Open for Questions:</b> 'Thank you very much. We are now open for questions and discussion.'"
            ],
            "math": [
                "• <b>Defense Preparedness:</b> Master numerical sheet and ablation breakdowns ready for immediate examiner reference."
            ],
            "qa": [
                ("What is the next step for this project?",
                 "We have completed drafting a 31-page research manuscript formatted for IEEE Transactions on Geoscience and Remote Sensing (TGRS). We plan to submit the paper after adding INT8 quantization experiments for edge UAV deployment.")
            ]
        }
    ]

    for s in slides:
        card = slide_header_card(s["num"], s["title"], s["time"], s["speaker"], s["visual"])
        
        # Load screenshot image
        img_name = f"slide_{s['num']:02d}.png"
        img_path = os.path.join(img_dir, img_name)
        img_element = None
        if os.path.exists(img_path):
            img_element = Image(img_path, width=320, height=180)
            img_element.hAlign = 'CENTER'

        spoken_p = [Paragraph(p, script_bullet) for p in s["spoken"]]
        spoken_box = make_callout(
            "WHAT YOU MUST SAY (Point-Wise Spoken Script)",
            spoken_p,
            bg_color="#F8FAFC", border_color="#94A3B8", title_color="#1E3A8A"
        )
        
        math_p = [Paragraph(p, math_bullet) for p in s["math"]]
        math_box = make_callout(
            "KEY CONCEPTS & PRONUNCIATION GUIDE",
            math_p,
            bg_color="#F0FDF4", border_color="#86EFAC", title_color="#166534"
        )
        
        qa_p = []
        for q, a in s["qa"]:
            qa_p.append(Paragraph(f"<b>Examiner:</b> {q}", qa_q_style))
            qa_p.append(Spacer(1, 1))
            qa_p.append(Paragraph(f"<b>Winning Defense Answer:</b> {a}", qa_a_style))
            qa_p.append(Spacer(1, 2))
            
        qa_box = make_callout(
            "VIVA DEFENSE SHIELD (Anticipated Examiner Questions & Winning Answers)",
            qa_p,
            bg_color="#FEF2F2", border_color="#FECACA", title_color="#991B1B"
        )
        
        slide_elements = [card, Spacer(1, 4)]
        if img_element:
            slide_elements.extend([img_element, Spacer(1, 4)])
        slide_elements.extend([
            spoken_box, Spacer(1, 4),
            math_box, Spacer(1, 4),
            qa_box, Spacer(1, 6),
            PageBreak()
        ])
        story.extend(slide_elements)

    # ==================== MASTER VIVA DEFENSE SURVIVAL KIT ====================
    story.append(Paragraph("MASTER VIVA DEFENSE SURVIVAL KIT: TOP 10 QUESTIONS", section_title))
    story.append(Paragraph("Be thoroughly prepared for these 10 fundamental questions that examiners frequently ask during minor project defenses:", body_style))
    story.append(Spacer(1, 6))

    top10_qa = [
        ("1. What is the fundamental difference between Multispectral and Hyperspectral imaging?",
         "Multispectral imaging (MSI) captures 4 to 20 discrete, wide bands with gaps between them (~30–50 nm resolution). Hyperspectral imaging (HSI) captures hundreds of contiguous, narrow bands (~1–10 nm resolution), providing a continuous spectral reflectance curve that acts as a chemical fingerprint to identify visually identical materials."),
        
        ("2. What is the Hughes phenomenon, and how does SGSCF-Net resolve it?",
         "The Hughes phenomenon is the curse of dimensionality where classification accuracy degrades as spectral bands increase for a fixed small sample size. SGSCF-Net resolves it by decoupling spectral and spatial processing: SBAM filters noisy bands upfront, CBSA compresses spectral redundancy via complementary gating, and SCFG collapses spectral depth before spatial ResBlocks, keeping total parameters at only 308K."),

        ("3. Why did SSRN fail on minority classes while SGSCF-Net succeeded?",
         "SSRN uses standard Cross-Entropy loss without class weighting. On Indian Pines, majority classes like Soybeans (954 samples) generate gradients orders of magnitude larger than Oats (20 samples), causing SSRN to score 0.00% on Oats, Alfalfa, and Grass-mowed. SGSCF-Net uses a class-weighted Focal Loss ($w_c \\propto 1/N_c$) with dynamic Spread Loss, lifting Oats accuracy to 54.17% and achieving the highest Average Accuracy (94.82%)."),

        ("4. What is the Second-Order Covariance Fusion Gate (SCFG), and why is it novel?",
         "Standard models use first-order Global Average Pooling (GAP), which only calculates channel means and discards spatial-channel co-dependencies. SCFG calculates the outer product $G = f_{ch} \\otimes f_{sp}$, forming a second-order covariance matrix that captures bilinear interactions between spectral bands and spatial coordinates, stabilized via Frobenius normalization using only 82 parameters."),

        ("5. How does CBSA guarantee zero feature redundancy between branches?",
         "CBSA enforces the mathematical partition of unity $\\alpha + (1-\\alpha) \\equiv 1$. Because the gating weights sum to 1, the two branches compete directly for representation capacity. When spectral fidelity dominates, $\\alpha \\to 1$; when spatial texture dominates, $(1-\\alpha) \\to 1$. Redundant feature overlap between branches is mathematically impossible."),

        ("6. Why not just use Vision Transformers (ViT) or Mamba for HSI?",
         "Vision Transformers have $\\mathcal{O}(N^2)$ quadratic complexity, exceed 85M parameters, and suffer extreme data starvation on small HSI splits (achieving only 90.33% OA on Indian Pines). Mamba requires non-portable C++ CUDA extensions (`mamba_ssm`), locking it out of edge hardware. SGSCF-Net uses native PyTorch operations with linear $\\mathcal{O}(N)$ complexity and only 308K parameters."),

        ("7. Why did you choose patch size 11×11?",
         "Through our 60-run hyperparameter grid search, 11×11 proved globally optimal. A 9×9 patch lacks sufficient spatial context, while patches larger than 11×11 cause cross-field boundary contamination. At 20m/pixel in Indian Pines, an 11×11 patch covers a 220m × 220m ground footprint, perfectly matching agricultural parcels."),

        ("8. Why does Batch Size 32 outperform Batch Size 256?",
         "In an imbalanced dataset with batch size 256, minority samples appear in less than 1% of the batches, so the averaged gradient is completely dominated by majority classes. Batch size 32 provides frequent, high-variance updates that allow minority class gradients to make meaningful parameter updates (+3.43% OA on Indian Pines)."),

        ("9. How did you prevent data leakage during training?",
         "We used stratified sampling with strictly non-overlapping pixel coordinates. Training patches (10%) and testing patches (80%) have disjoint center coordinates, and test pixels were never seen during training or validation, ensuring zero spatial autocorrelation leakage."),

        ("10. What are the edge deployment advantages of SGSCF-Net?",
         "With a 1.2 MB checkpoint, linear $\\mathcal{O}(N)$ complexity, and a 1.18-second test inference time, SGSCF-Net can be deployed directly on embedded edge platforms (such as NVIDIA Jetson Orin on UAVs) for real-time onboard crop health monitoring and disaster assessment.")
    ]

    for q, a in top10_qa:
        q_box = [
            Paragraph(f"<b>Question:</b> {q}", qa_q_style),
            Spacer(1, 2),
            Paragraph(f"<b>Winning Defense Answer:</b> {a}", qa_a_style)
        ]
        story.append(make_callout(
            "CORE VIVA DEFENSE POINT",
            q_box,
            bg_color="#F8FAFC", border_color="#CBD5E1", title_color="#0F172A"
        ))
        story.append(Spacer(1, 5))

    doc.build(story, canvasmaker=DefenseCanvas)
    print(f"Successfully generated PDF at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    build_pdf()
