import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class ThreePageCanvas(canvas.Canvas):
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
        self.setFillColor(colors.HexColor("#334155"))
        
        # Header on all pages
        self.drawString(36, 755, "SGSCF-Net: The Three Novel Architectural Modules (Manuscript Reference)")
        self.setFont("Helvetica", 8)
        self.drawRightString(576, 755, "NIT Raipur — Dept. of IT | ACM Manuscript Ref")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 749, 576, 749)

        # Footer on all pages
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 28, "Tejas Surya (23118098) | B. Vyshnavi (23118021) | Valluri Lahitha (23118106) — Guide: Dr. R.R. Janghel")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 28, page_text)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 38, 576, 38)
        
        self.restoreState()

def build_pdf(filename="SGSCF_Net_3_Novel_Modules_Explanation.pdf"):
    pdf_path = os.path.join(r"C:\Users\LENOVO\Desktop\MinorProject_Final", filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=48,
        bottomMargin=45
    )
    
    usable_width = 540
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    mod_title_style = ParagraphStyle(
        'ModTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=15,
        textColor=colors.white
    )
    mod_meta_style = ParagraphStyle(
        'ModMeta', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=11,
        textColor=colors.HexColor("#E2E8F0")
    )
    section_h_style = ParagraphStyle(
        'SecHeader', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8.5, leading=11,
        textColor=colors.HexColor("#0F172A")
    )
    body_style = ParagraphStyle(
        'BodyTextCustom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.8, leading=10.8,
        textColor=colors.HexColor("#1E293B")
    )
    bullet_style = ParagraphStyle(
        'BulletCustom', parent=body_style,
        leftIndent=8, firstLineIndent=-8
    )
    math_style = ParagraphStyle(
        'MathCustom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.6, leading=10.5,
        textColor=colors.HexColor("#065F46"),
        leftIndent=8, firstLineIndent=-8
    )
    qa_q_style = ParagraphStyle(
        'QAQuestion', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=7.8, leading=10.5,
        textColor=colors.HexColor("#991B1B")
    )
    qa_a_style = ParagraphStyle(
        'QAAnswer', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.8, leading=10.5,
        textColor=colors.HexColor("#1E293B"), leftIndent=6
    )

    def header_banner(mod_num, title, location, params_str, color_hex="#1E3A8A"):
        t1 = Paragraph(f"<b>MODULE {mod_num}: {title.upper()}</b>", mod_title_style)
        t2 = Paragraph(f"<b>Network Stage:</b> {location} &nbsp;|&nbsp; <b>Parameter Footprint:</b> {params_str}", mod_meta_style)
        t = Table([[t1], [t2]], colWidths=[usable_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(color_hex)),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    def make_box(title, paragraphs, bg_color="#F8FAFC", border_color="#CBD5E1", title_color="#0F172A"):
        header = Paragraph(f"<b>{title}</b>", ParagraphStyle('BoxH', fontName='Helvetica-Bold', fontSize=8.2, leading=10.5, textColor=colors.HexColor(title_color)))
        content = [header, Spacer(1, 1.5)] + paragraphs
        t = Table([[content]], colWidths=[usable_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
            ('BOX', (0,0), (-1,-1), 0.6, colors.HexColor(border_color)),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    fig_dir = r"C:\Users\LENOVO\Desktop\MinorProject_Final\extracted_manuscript_figures"
    story = []

    # ==================== PAGE 1: MODULE 1 (SBAM) ====================
    sbam_banner = header_banner(
        1, "Spectral Band Attention Module (SBAM)",
        "Stage I — Input Recalibration (Pre-Convolution)",
        "10,225 params (3.32% of SGSCF-Net)",
        "#1E3A8A"
    )
    story.append(sbam_banner)
    story.append(Spacer(1, 3))

    # Figure
    sbam_img_path = os.path.join(fig_dir, "fig_sbam.png")
    if os.path.exists(sbam_img_path):
        img_sbam = Image(sbam_img_path, width=320, height=174)
        img_sbam.hAlign = 'CENTER'
        story.append(img_sbam)
        story.append(Spacer(1, 3))

    # Box 1: Problem & Architectural Purpose
    box1_p = [
        Paragraph("• <b>Atmospheric Noise & Degradation:</b> Airborne hyperspectral sensors (AVIRIS/ROSIS) capture 100–200+ contiguous bands. Critical regions suffer from severe <b>atmospheric water vapor absorption</b> (bands 104–108 at 1.4 µm, bands 150–163 at 1.9 µm). If raw cubes enter 3D convolutions directly, noisy zero-transmission bands corrupt representations, propagating errors into deep layers.", bullet_style),
        Paragraph("• <b>Pre-Convolution Solution:</b> Unlike SENet or CBAM that operate strictly on deep latent feature maps, SBAM operates <b>directly on raw spectral bands before any convolution occurs</b>. It dynamically learns continuous soft attention weights per scene, shielding downstream representation learning.", bullet_style)
    ]
    story.append(make_box("1. PROBLEM STATEMENT & ARCHITECTURAL MOTIVATION", box1_p, "#F8FAFC", "#94A3B8", "#1E3A8A"))
    story.append(Spacer(1, 3))

    # Box 2: Mathematical Formulation & Step-by-Step Dataflow
    box2_p = [
        Paragraph("• <b>Step 1 (Spatial Compression):</b> For volumetric patch X ∈ ℝ^(B×1×P×P×C), adaptive 3D average pooling compresses spatial grid P×P to extract global spectral energy descriptor z ∈ ℝ^(B×C):<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>z_c = (1 / P²) ∑_{u=1}^P ∑_{v=1}^P X(b, 1, u, v, c)</b>, &nbsp; for each band c ∈ {1, ..., C}.", math_style),
        Paragraph("• <b>Step 2 (MLP Bottleneck with Reduction r=8):</b> z passes through two linear projections with reduction ratio r=8 (bottleneck channels C_mid = max(⌊C/8⌋, 8)), applying ReLU δ and Sigmoid σ:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>w = σ(W₂ · δ(W₁ z + b₁) + b₂) ∈ (0, 1)^(B×C)</b>, &nbsp; where W₁ ∈ ℝ^(C_mid×C) and W₂ ∈ ℝ^(C×C_mid).", math_style),
        Paragraph("• <b>Step 3 (Channel-Wise Hadamard Modulation):</b> Attention vector w is broadcast across spatial dimensions ℰ(w) and multiplied element-wise with input cube X:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>X̃ = X ⊙ ℰ(w) ∈ ℝ^(B×1×P×P×C)</b>. Noisy water absorption bands receive near-zero weights and are attenuated at the physical boundary.", math_style)
    ]
    story.append(make_box("2. MATHEMATICAL FORMULATION & TENSOR TRANSFORMATIONS", box2_p, "#F0FDF4", "#86EFAC", "#166534"))
    story.append(Spacer(1, 3))

    # Box 3: Novelty & Viva Defense Shield
    box3_p = [
        Paragraph("<b>Examiner: Why not simply delete noisy water absorption bands manually during data preprocessing?</b>", qa_q_style),
        Paragraph("<b>Winning Defense Answer:</b> Manual band removal relies on rigid, hardcoded indices (e.g. deleting 20 bands in AVIRIS). However, atmospheric attenuation, sensor drift, and noise vary continuously across different flight lines and times. SBAM dynamically learns continuous, data-driven attention weights per scene, preserving useful residual spectral signal while filtering corrupted channels adaptively with only 10,225 parameters.", qa_a_style)
    ]
    story.append(make_box("3. VIVA DEFENSE SHIELD & NOVELTY HIGHLIGHTS", box3_p, "#FEF2F2", "#FECACA", "#991B1B"))
    story.append(PageBreak())

    # ==================== PAGE 2: MODULE 2 (CBSA) ====================
    cbsa_banner = header_banner(
        2, "Complementary Branch Spectral Attention (CBSA)",
        "Stage I — Dual-Stream Spectral Feature Entry",
        "80 params (0.03% of SGSCF-Net)",
        "#0D9488"
    )
    story.append(cbsa_banner)
    story.append(Spacer(1, 3))

    # Figure
    cbsa_img_path = os.path.join(fig_dir, "fig_cbsa.png")
    if os.path.exists(cbsa_img_path):
        img_cbsa = Image(cbsa_img_path, width=320, height=174)
        img_cbsa.hAlign = 'CENTER'
        story.append(img_cbsa)
        story.append(Spacer(1, 3))

    # Box 1: Problem & Architectural Purpose
    box1_cbsa = [
        Paragraph("• <b>Multi-Branch Redundancy Problem:</b> Dual-stream 3D networks (e.g. Double-Branch CNNs, A2S2K-ResNet) extract features using parallel branches. However, combining them via direct summation (a + b) or shared attention assigns identical importance to both streams, causing co-dependent, duplicated feature representations and severe parameter explosion.", bullet_style),
        Paragraph("• <b>Complementary Branch Solution:</b> CBSA decouples spectral and spatial-spectral modeling into two dedicated parallel 3D streams, fused via a learned gating scalar α strictly constrained by <b>α + (1 - α) ≡ 1</b>. This forces orthogonal specialization and zero branch redundancy.", bullet_style)
    ]
    story.append(make_box("1. PROBLEM STATEMENT & ARCHITECTURAL MOTIVATION", box1_cbsa, "#F8FAFC", "#94A3B8", "#0F766E"))
    story.append(Spacer(1, 3))

    # Box 2: Mathematical Formulation
    box2_cbsa = [
        Paragraph("• <b>Step 1 (Dual Receptive Fields):</b> Recalibrated patch X̃ enters two parallel 3D convolutions (K=16 filters, spectral stride=2, C₂ = ⌊(C-7)/2⌋ + 1):<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>Branch a (Pure Spectral):</b> a = ReLU(BN(Conv3D_{(1×1×7)}(X̃))) ∈ ℝ^(B×16×P×P×C₂) (captures narrow inter-band absorption lines).<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>Branch b (Spatial-Spectral):</b> b = ReLU(BN(Conv3D_{(3×3×7)}(X̃))) ∈ ℝ^(B×16×P×P×C₂) (captures joint spatial context).", math_style),
        Paragraph("• <b>Step 2 (Spatial Pooling & Depthwise Conv1D Gate):</b> Spatial context is pooled over fused features U = a + b:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>U_avg = (1 / P²) ∑_{u,v} (a + b)[:, :, u, v, :] ∈ ℝ^(B×16×C₂)</b>.<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;A depthwise 1D convolution (kernel k_s=3, grouped across 16 channels, only 48 params) models cross-band dependencies:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>α = σ(Conv1D_{k_s}(U_avg)) ∈ ℝ^(B×16×C₂)</b>.", math_style),
        Paragraph("• <b>Step 3 (Complementary Convex Fusion):</b> The streams are fused via complementary combination and 3D batch normalization:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>v = BN(α ⊙ a + (1 - α) ⊙ b) ∈ ℝ^(B×16×P×P×C₂)</b>.<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;Because α + (1 - α) ≡ 1, the branches compete for capacity: when spectral fidelity dominates, α → 1; when spatial texture dominates, (1 - α) → 1.", math_style)
    ]
    story.append(make_box("2. MATHEMATICAL FORMULATION & TENSOR TRANSFORMATIONS", box2_cbsa, "#F0FDF4", "#86EFAC", "#166534"))
    story.append(Spacer(1, 3))

    # Box 3: Novelty & Viva Defense Shield
    box3_cbsa = [
        Paragraph("<b>Examiner: Why enforce α and (1 - α) instead of learning two independent attention gates α and β?</b>", qa_q_style),
        Paragraph("<b>Winning Defense Answer:</b> If α and β are learned independently, both can saturate to 1, causing high feature redundancy and doubling the feature magnitude. Enforcing the partition of unity α + (1 - α) = 1 mathematically guarantees that the branches compete and specialize into orthogonal representations without duplicating features, using only 80 parameters.", qa_a_style)
    ]
    story.append(make_box("3. VIVA DEFENSE SHIELD & NOVELTY HIGHLIGHTS", box3_cbsa, "#FEF2F2", "#FECACA", "#991B1B"))
    story.append(PageBreak())

    # ==================== PAGE 3: MODULE 3 (SCFG) ====================
    scfg_banner = header_banner(
        3, "Second-Order Covariance Fusion Gate (SCFG)",
        "Stage I ──► Stage II Transition Bridge (Spectral to Spatial)",
        "82 params (0.03% of SGSCF-Net)",
        "#7C3AED"
    )
    story.append(scfg_banner)
    story.append(Spacer(1, 3))

    # Figure
    scfg_img_path = os.path.join(fig_dir, "fig_scfg.png")
    if os.path.exists(scfg_img_path):
        img_scfg = Image(scfg_img_path, width=320, height=174)
        img_scfg.hAlign = 'CENTER'
        story.append(img_scfg)
        story.append(Spacer(1, 3))

    # Box 1: Problem & Architectural Purpose
    box1_scfg = [
        Paragraph("• <b>The Flaw of First-Order GAP:</b> Standard spatial attention mechanisms and global pooling compute only the <b>first-order spatial mean</b> μ_c = (1/N) ∑ x_{c,i}. In remote sensing, this discards spatial-channel co-activations (e.g. how specific spectral channels correlate with edges vs homogeneous interiors), failing to separate spectrally similar materials like asphalt and gravel, or subtle shadow gradients.", bullet_style),
        Paragraph("• <b>Second-Order Covariance Solution:</b> SCFG computes a <b>bilinear outer product G = f_ch ⊗ f_sp</b> between the channel descriptor and spatial position descriptor, normalized via the Frobenius norm. This models second-order statistics with zero iterative decompositions.", bullet_style)
    ]
    story.append(make_box("1. PROBLEM STATEMENT & ARCHITECTURAL MOTIVATION", box1_scfg, "#F8FAFC", "#94A3B8", "#6D28D9"))
    story.append(Spacer(1, 3))

    # Box 2: Mathematical Formulation
    box2_scfg = [
        Paragraph("• <b>Step 1 (Spectral Collapse & Channel Expansion):</b> Adaptive 3D average pooling collapses spectral dimension C₂ → 1; Conv3D_{(1×1×1)} expands channels 16 → C_mid = 128. Squeezing yields 2D spatial feature map x ∈ ℝ^(B×128×P×P) with N = P × P pixels.<br/>"
                  "• <b>Step 2 (Dual Descriptors):</b> Global channel descriptor f_ch and spatial position descriptor f_sp are extracted:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>f_ch(b, c) = (1 / N) ∑_{n=1}^N x(b, c, n) ∈ ℝ^(B×128)</b>, &nbsp;&nbsp; <b>f_sp(b, n) = (1 / 128) ∑_{c=1}^{128} x(b, c, n) ∈ ℝ^(B×N)</b>.", math_style),
        Paragraph("• <b>Step 3 (Bilinear Covariance Matrix via Outer Product):</b> Outer product captures cross-domain co-occurrences:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>G = f_ch ⊗ f_sp = f_ch f_sp^T ∈ ℝ^(B×128×N)</b>, &nbsp; where G(b, c, n) = f_ch(b, c) · f_sp(b, n).", math_style),
        Paragraph("• <b>Step 4 (Frobenius Normalization & Gating):</b> Normalized to stabilize gradient backpropagation without matrix square roots:<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;<b>G̃ = G / (‖G‖_F + ε)</b>, &nbsp; where ‖G‖_F = √(∑_c ∑_n G(b, c, n)²) and ε = 10⁻⁸.<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;Linear projection W_G ∈ ℝ^(1×N) produces channel gate: <b>g = σ(W_G G̃) ∈ ℝ^(B×128)</b>.<br/>"
                  "&nbsp;&nbsp;&nbsp;&nbsp;Spatial features are modulated element-wise: <b>x_scfg = x ⊙ g</b>.", math_style)
    ]
    story.append(make_box("2. MATHEMATICAL FORMULATION & TENSOR TRANSFORMATIONS", box2_scfg, "#F0FDF4", "#86EFAC", "#166534"))
    story.append(Spacer(1, 3))

    # Box 3: Novelty & Viva Defense Shield
    box3_scfg = [
        Paragraph("<b>Examiner: What physical information does the outer product f_ch ⊗ f_sp capture in remote sensing?</b>", qa_q_style),
        Paragraph("<b>Winning Defense Answer:</b> In remote sensing, a pixel's reflectance is strongly coupled to its spatial context. First-order pooling averages these out independently. The outer product captures the second-order co-occurrence: how specific spectral absorption bands vary across spatial locations. This enabled SGSCF-Net to achieve 99.34% accuracy on shadow pixels and 99.76% on gravel in Pavia University using only 82 parameters.", qa_a_style)
    ]
    story.append(make_box("3. VIVA DEFENSE SHIELD & NOVELTY HIGHLIGHTS", box3_scfg, "#FEF2F2", "#FECACA", "#991B1B"))

    doc.build(story, canvasmaker=ThreePageCanvas)
    print(f"Successfully generated 3-page PDF at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    build_pdf()
