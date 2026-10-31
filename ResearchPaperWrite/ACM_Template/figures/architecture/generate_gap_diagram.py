# -*- coding: utf-8 -*-
"""
generate_gap_diagram.py -- Generate the GAP (Global Average Pooling) and 
Classification Head architecture diagram matching SGSCF-Net's style.
Canvas: 660 x 360 px (matching fig_sbam, fig_cbsa, and fig_scfg).
"""
import os, sys, subprocess
import numpy as np

KIT = os.path.expanduser("~/.claude/skills/paper-architecture-diagrams/scripts")
sys.path.insert(0, KIT)
import drawio_kit
drawio_kit.FONT = "Times New Roman"
from drawio_kit import (Diagram, cuboid, stack, matrix, neurons, m, lighten, darken, cmap_hex)

HERE = os.path.dirname(os.path.abspath(__file__))

# Palette matching the paper's Stage II
GAP_C  = "#1E88E5"     # Vibrant royal blue for GAP
RES_C  = "#4A6FA5"     # Steel blue for spatial residual input
FC_C   = "#2D6CD1"     # Deep blue for linear projection
PRED_C = "#3E8E4E"     # Forest green for predictions
LOSS_C = "#C93C32"     # Crimson red for hybrid loss
GREY   = "#555555"
INK    = "#262626"
FS, FS_S, FS_X, FS_T = 16, 14, 13, 17
MOD_W, MOD_H = 660, 360

class Cub:
    def __init__(s, d, x, cy, w, h, dx, dy, **kw):
        s.d, s.w, s.h, s.dx, s.dy = d, w, h, dx, dy
        s.g = cuboid(w, h, dx, dy, **kw)
        s.x, s.y, s.cy = x, cy - (h + dy) / 2, cy
        s.id = d.glyph(s.x, s.y, s.g)

    @property
    def W(s): return s.w + s.dx
    @property
    def L(s): return (s.x, s.cy)
    @property
    def R(s): return (s.x + s.W, s.cy)

def arrow(d, a, b, pts=(), **kw):
    kw.setdefault("color", INK)
    kw.setdefault("sw", 1.3)
    return d.edge(a, b, pts=pts, **kw)

def sub(text):
    return f'<br><font style="font-size:{FS_X - 1}px;font-weight:normal;color:#555555">{text}</font>'

def build_gap_diagram():
    d = Diagram("fig_gap", MOD_W, MOD_H)
    d.shift(0, 0)
    
    # Outer panel enclosing the module
    d.panel(12, 12, 636, 336, "GAP & Classification Head", lighten(GAP_C, 0.94), GAP_C, fs=FS_T, title_align="right")
    
    Y = 125
    
    # -------------------------------------------------------------------------
    # 1. Input Spatial Feature Tensor from Spatial ResBlock: s in R^(K x P x P x 1)
    # -------------------------------------------------------------------------
    purples = cmap_hex("Blues", np.linspace(0.35, 0.85, 8))
    s_cub = Cub(d, 28, Y, 46, 46, 36, 24, front="#E3F2FD", stroke=RES_C, stripes=purples)
    
    d.text(s_cub.x - 14, s_cub.y - 28, 90, 20, m(r"\mathbf{s}\in\mathbb{R}^{K\times P\times P\times 1}"), fs=FS_S, bold=True)
    d.text(s_cub.x - 14, s_cub.y + s_cub.h + s_cub.dy + 4, 90, 18, "Spatial ResBlock", fs=FS_X, color=RES_C, italic=True)
    d.text(s_cub.x + 8, s_cub.y + s_cub.h + s_cub.dy - 12, 38, 16, m(r"P{\times}P"), fs=11, color=GREY)
    d.text(s_cub.x + s_cub.W - 16, s_cub.y + 4, 32, 16, m(r"K{=}16"), fs=11, color=GREY)

    # -------------------------------------------------------------------------
    # 2. Global Average Pooling (GAP) Operator Block
    # -------------------------------------------------------------------------
    gap_x, gap_y = 154, Y - 30
    gap_w, gap_h = 72, 60
    d.block(gap_x, gap_y, gap_w, gap_h, "GAP" + sub(m(r"\mathrm{Pool}_{3\mathrm{D}}")), GAP_C, fs=FS_S)
    arrow(d, s_cub.R, (gap_x, Y))
    
    # Mathematical spatial reduction formula above GAP
    d.rect(gap_x - 24, gap_y - 44, gap_w + 48, 32, 
           m(r"\mathbf{z}_k = \frac{1}{P^2}\sum_{i=1}^{P}\sum_{j=1}^{P}\mathbf{s}_{k,i,j}"),
           fill="#FFFFFF", stroke=lighten(GAP_C, 0.4), sw=1.0, fs=11)
    d.edge((gap_x + gap_w / 2, gap_y - 12), (gap_x + gap_w / 2, gap_y), color=GAP_C, sw=1.0, dashed=True)

    # -------------------------------------------------------------------------
    # 3. Latent Feature Vector: z in R^K (16-D compact representation)
    # -------------------------------------------------------------------------
    z_x = 270
    z_matrix = matrix(8, 1, 16, 10, [0.85, 0.42, 0.95, 0.18, 0.73, 0.61, 0.35, 0.88],
                      lo="#E3F2FD", hi=GAP_C, stroke="#1565C0", sw=0.7)
    z_y = Y - z_matrix.h / 2
    d.glyph(z_x, z_y, z_matrix)
    
    arrow(d, (gap_x + gap_w, Y), (z_x - 3, Y))
    
    d.text(z_x - 24, z_y - 24, z_matrix.w + 48, 18, "Latent Vector", fs=FS_X, color=GREY, italic=True)
    d.text(z_x - 20, z_y + z_matrix.h + 4, z_matrix.w + 40, 20, m(r"\mathbf{z}\in\mathbb{R}^{16}"), fs=FS_S, bold=True)

    # -------------------------------------------------------------------------
    # 4. Linear Fully Connected Layer: W_fc in R^(N_c x K) + b
    # -------------------------------------------------------------------------
    fc_x, fc_y = 338, Y - 30
    fc_w, fc_h = 76, 60
    d.block(fc_x, fc_y, fc_w, fc_h, "FC Head" + sub(m(r"16\to N_c")), FC_C, fs=FS_S)
    arrow(d, (z_x + z_matrix.w + 3, Y), (fc_x, Y))
    
    # Projection equation above FC
    d.rect(fc_x - 14, fc_y - 44, fc_w + 28, 32,
           m(r"\boldsymbol{\ell} = \mathbf{W}_{\mathrm{fc}}\mathbf{z} + \mathbf{b}"),
           fill="#FFFFFF", stroke=lighten(FC_C, 0.4), sw=1.0, fs=11)
    d.edge((fc_x + fc_w / 2, fc_y - 12), (fc_x + fc_w / 2, fc_y), color=FC_C, sw=1.0, dashed=True)

    # -------------------------------------------------------------------------
    # 5. Softmax & Class Probability Distribution: \hat{y} in [0, 1]^(N_c)
    # -------------------------------------------------------------------------
    sm_x, sm_y = 458, Y - 25
    sm_w, sm_h = 58, 50
    d.block(sm_x, sm_y, sm_w, sm_h, "Softmax", "#6B6B6B", fs=FS_S, bold=False)
    arrow(d, (fc_x + fc_w, Y), (sm_x, Y))
    d.text(fc_x + fc_w + 1, Y - 18, sm_x - (fc_x + fc_w) - 2, 16, m(r"\boldsymbol{\ell}\in\mathbb{R}^{N_c}"), fs=10, color=GREY)
    
    # Output class probabilities: vertical bar / matrix
    y_x = 558
    y_matrix = matrix(8, 1, 16, 10, [0.03, 0.01, 0.94, 0.01, 0.01, 0.02, 0.01, 0.01],
                      lo="#E8F5E9", hi=PRED_C, stroke="#2E7D32", sw=0.7)
    y_y = Y - y_matrix.h / 2
    d.glyph(y_x, y_y, y_matrix)
    
    arrow(d, (sm_x + sm_w, Y), (y_x - 3, Y))
    
    d.text(y_x - 20, y_y - 24, y_matrix.w + 40, 18, "Probabilities", fs=FS_X, color=PRED_C, italic=True)
    d.text(y_x - 16, y_y + y_matrix.h + 4, y_matrix.w + 32, 20, m(r"\hat{\mathbf{y}}\in[0,1]^{N_c}"), fs=FS_S, bold=True)
    d.text(y_x + y_matrix.w + 6, y_y + 12, 34, 20, m(r"\hat{c}"), fs=FS+1, color=PRED_C, bold=True)

    # -------------------------------------------------------------------------
    # 6. Training Supervision: Minority-Aware Hybrid Loss Box
    # -------------------------------------------------------------------------
    loss_box_x, loss_box_y = 106, 238
    loss_box_w, loss_box_h = 514, 82
    
    d.rect(loss_box_x, loss_box_y, loss_box_w, loss_box_h, "",
           fill="#FFFBFB", stroke=LOSS_C, dashed=True, sw=1.3)
    
    d.text(loss_box_x + 12, loss_box_y + 6, 240, 18, "Training Loss Supervision:", fs=FS_X, bold=True, color=LOSS_C)
    d.text(loss_box_x + 12, loss_box_y + 28, 480, 24,
           m(r"\mathcal{L}_{\mathrm{Total}} = 0.7\,\mathcal{L}_{\mathrm{Spread}}(m_r) + 0.3\,\mathcal{L}_{\mathrm{Focal}}(\gamma, \mathbf{w})"),
           fs=FS_S, color=INK)
    d.text(loss_box_x + 12, loss_box_y + 54, 480, 18,
           "Dynamic margin " + m(r"m_r\in[0.2, 0.9]") + " + Inverse-frequency class weights " + m(r"w_c \propto 1/N_c^{\alpha}"),
           fs=11, color=GREY)
    
    # Downward dashed flow from predictions to loss
    d.edge((y_x + y_matrix.w / 2, y_y + y_matrix.h + 24), 
           (loss_box_x + loss_box_w - 40, loss_box_y),
           pts=[(y_x + y_matrix.w / 2, loss_box_y - 12), (loss_box_x + loss_box_w - 40, loss_box_y - 12)],
           color=LOSS_C, dashed=True, sw=1.2)
    
    # Ground truth label badge feeding into loss
    d.rect(26, loss_box_y + 20, 64, 42, "Target<br>" + m(r"y\in\{1..N_c\}"),
           fill="#FFFFFF", stroke="#555555", sw=1.0, fs=11)
    d.edge((90, loss_box_y + 41), (loss_box_x, loss_box_y + 41), color="#555555", dashed=True, sw=1.2)

    return d

if __name__ == '__main__':
    print(f"Generating fig_gap in {HERE}...")
    d = build_gap_diagram()
    d.save(HERE)
    print("Saved fig_gap.drawio successfully!")

    # Render via render_drawio.py
    out_drawio = os.path.join(HERE, "fig_gap.drawio")
    render_script = os.path.join(KIT, "render_drawio.py")
    cmd = ["python", render_script, out_drawio, "--scale", "3"]
    print("Rendering fig_gap.drawio to PDF and PNG...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("RC:", res.returncode)
    print("STDOUT:", res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr[:300])
