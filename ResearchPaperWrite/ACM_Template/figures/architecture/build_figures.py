"""Build the four SGSCF-Net architecture figures (.drawio) and render them (.pdf/.png).

Every operator, kernel and tensor shape mirrors SGSCF-Net_Final_Implementation/SGSCF_Net.py.
Toolkit: ~/.claude/skills/paper-architecture-diagrams/scripts (drawio_kit.py, render_drawio.py)
usage: python build_figures.py [--no-render]

Print sizes (fonts land at ~7-8 pt):
  fig_sgscf_overall -> figure*, width=\\textwidth
  fig_sbam / fig_cbsa / fig_scfg (common 660 px canvas) -> figure*, width=0.75\\textwidth
"""
import os, sys
import numpy as np
import scipy.io as sio

KIT = os.path.expanduser("~/.claude/skills/paper-architecture-diagrams/scripts")
sys.path.insert(0, KIT)
import drawio_kit
drawio_kit.FONT = "Times New Roman"
from drawio_kit import (Diagram, cuboid, stack, circles, neurons, local_conv, matrix, m, mix, lighten,
                        png_b64, file_png_uri, cmap_hex, SLICES)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(FIGS, "..", "..", ".."))
DATA_DIRS = [os.path.join(ROOT, "data"), "C:/Users/LENOVO/Pictures/MinorProject/data"]

# ----------------------------------------------------------------------------- palette & type
SBAM_C, CBSA_C, SCFG_C = "#3E8E4E", "#D9822B", "#7A56B0"     # proposed modules
A_C, B_C = "#2D6CD1", "#C93C32"                               # 1x1x7 branch a / 3x3x7 branch b
INK, GREY, STD_C = "#262626", "#555555", "#6B6B6B"
RES_C = "#4A6FA5"
ST1 = ("#F3F7FC", "#86A9D8")
ST2 = ("#FCF5F2", "#D69C88")
FS, FS_S, FS_X, FS_T = 16, 14, 13, 17
MOD_W, MOD_H = 660, 360                                       # shared canvas for the module figures


# ----------------------------------------------------------------------------- real HSI data
def load_ip():
    for d in DATA_DIRS:
        p = os.path.join(d, "Indian_pines_corrected.mat")
        if os.path.exists(p):
            return sio.loadmat(p)["indian_pines_corrected"].astype(np.float32)
    raise FileNotFoundError("Indian_pines_corrected.mat")


def stretch(a, lo=2, hi=98):
    p1, p2 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1)


def band_face(M):
    """(n_pixels, C) spectra -> per-band normalised turbo colours (n_pixels, C, 3)."""
    import matplotlib
    N = (M - M.min(0)) / (M.max(0) - M.min(0) + 1e-6)
    return matplotlib.colormaps["turbo"](N)[..., :3]


IP = load_ip()
HH, WW, CC = IP.shape
FC = np.dstack([stretch(IP[:, :, b]) for b in (50, 27, 17)])           # classic IP false colour
PR, PC, HALF = 92, 76, 5                                                 # 11x11 patch on a field boundary
PATCH = FC[PR - HALF:PR + HALF + 1, PC - HALF:PC + HALF + 1]
SUB = slice(0, CC, 2)


def cube_faces(rows, cols):
    """top face = first row's spectra, side face = last column's spectra (back = high bands)."""
    top = band_face(IP[rows.start, cols, :][:, SUB])
    side = band_face(IP[rows, cols.stop - 1, :][:, SUB])
    return png_b64(np.flipud(top.transpose(1, 0, 2))), png_b64(side)


HSI_TOP, HSI_SIDE = cube_faces(slice(0, HH), slice(0, WW))
P_TOP, P_SIDE = cube_faces(slice(PR - HALF, PR + HALF + 1), slice(PC - HALF, PC + HALF + 1))
HSI_FRONT = png_b64(FC)
P_FRONT = png_b64(PATCH, upscale=8)

# Illustrative SBAM band weights: suppressed bands appear washed out on the re-weighted cube.
WTS = [0.95, 0.85, 0.30, 0.12, 0.70, 0.92, 0.55, 0.18, 0.80, 0.97, 0.40, 0.88]
TURBO12 = cmap_hex("turbo", np.linspace(0.06, 0.94, 12))
VEIL, VEIL_A = ["#FFFFFF"] * 12, [0.82 * (1 - w) for w in WTS]
PATCH_KW = dict(front_img=P_FRONT, top_img=P_TOP, side_img=P_SIDE, pixelated=True)


# ----------------------------------------------------------------------------- helpers
class Cub:
    """Cuboid placed by left x and bbox-centre y; L/R ports sit on that centre line, so chains of
    tensors connect with perfectly horizontal arrows."""

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

    @property
    def T(s): return (s.x + (s.w + s.dx) / 2, s.y + s.dy / 2)

    def Bx(s, f=0.5):
        return (s.x + f * s.w, s.y + s.dy + s.h)

    def dims(s, left=None, bottom=None, depth=None, fs=FS_X, color=GREY, bottom_f=0.5):
        d = s.d
        if left:
            d.text(s.x - 24, s.y + s.dy, 20, s.h, m(left), fs=fs, align="right", color=color)
        if bottom:
            d.text(s.x + bottom_f * s.w - 20, s.y + s.dy + s.h + 1, 40, 18, m(bottom), fs=fs, color=color)
        if depth:
            d.text(s.x + s.w + s.dx / 2 + 1, s.y + s.dy / 2 + s.h - 6, 44, 18, m(depth), fs=fs, align="left",
                   color=color)

    def title(s, tex, dy=-24, fs=FS, color=INK, w_extra=40):
        s.d.text(s.x - w_extra / 2, s.y + dy, s.W + w_extra, 20, tex, fs=fs, color=color)


def arrow(d, a, b, pts=(), **kw):
    kw.setdefault("color", INK)
    return d.edge(a, b, pts, **kw)


def sub(text):
    return f"<br><font style=\"font-size:{FS_X - 1}px;font-weight:normal\">{text}</font>"


# ============================================================================= FIG 1: overall
def fig_overall():
    d = Diagram("fig_sgscf_overall", 1080, 594)
    Y1, Y2 = 150, 418
    d.panel(196, 10, 874, 262, "Stage I · Spectral feature learning", *ST1, fs=FS_T)
    d.panel(196, 290, 874, 252, "Stage II · Spatial feature learning &amp; classification", *ST2, fs=FS_T,
            title_align="right", title_valign="bottom")

    # input HSI cube (real Indian Pines false colour + spectral faces) with patch cue
    s = 116 / WW
    Cub(d, 14, 112, 116, 116, 44, 34, front_img=HSI_FRONT, top_img=HSI_TOP, side_img=HSI_SIDE,
        highlight=((PC - HALF) * s, (PR - HALF) * s, 11 * s + 1, "#FF1A1A"))
    d.text(4, 14, 186, 22, "HSI cube " + m(r"H\times W\times C"), fs=FS)
    hx = 14 + (PC + HALF) * s + 4
    hy = (112 - (116 + 34) / 2) + 34 + (PR - HALF) * s + 4

    # ---------------- Stage I
    pt = Cub(d, 216, Y1, 56, 56, 32, 24, **PATCH_KW)
    pt.dims(bottom="P", depth="C")
    pt.title("Patch " + m("X"), dy=-26)
    arrow(d, (hx, hy), pt.L, pts=[(hx + 44, hy)], color="#E01B1B", sw=1.7, curved=True)
    d.text(30, 192, 160, 40, "neighbourhood<br>extraction", fs=FS_X, color="#B01515", italic=True)

    sb = d.block(318, Y1 - 31, 78, 62, "SBAM" + sub("band gate"), SBAM_C, fs=FS + 1)
    arrow(d, pt.R, (sb, 0, 0.5))
    xt = Cub(d, 422, Y1, 46, 46, 32, 22, stripes=VEIL, stripe_alpha=VEIL_A, **PATCH_KW)
    xt.title(m(r"\tilde{X}"), dy=-26)
    arrow(d, (sb, 1, 0.5), xt.L)

    fx = 526
    ca = Cub(d, 582, 66, 40, 40, 28, 18, front=lighten(A_C, 0.72), stroke=A_C, sw=1.2)
    cb = Cub(d, 582, 234, 40, 40, 28, 18, front=lighten(B_C, 0.72), stroke=B_C, sw=1.2)
    arrow(d, xt.R, ca.L, pts=[(fx, Y1), (fx, ca.cy)])
    arrow(d, xt.R, cb.L, pts=[(fx, Y1), (fx, cb.cy)])
    d.text(522, 38, 62, 22, m(r"1{\times}1{\times}7"), fs=FS_S, color=A_C)
    d.text(522, 238, 62, 22, m(r"3{\times}3{\times}7"), fs=FS_S, color=B_C)
    d.text(ca.x + ca.W, ca.y - 8, 22, 22, m(r"\boldsymbol{a}"), fs=FS + 1, color=A_C)
    d.text(cb.x + cb.W, cb.y + cb.h + cb.dy - 12, 22, 22, m(r"\boldsymbol{b}"), fs=FS + 1, color=B_C)

    cbs = d.block(678, Y1 - 34, 78, 68, "CBSA" + sub(m(r"\alpha\,\vert\,1{-}\alpha")), CBSA_C, fs=FS + 1)
    arrow(d, ca.R, (cbs, 0, 0.3), pts=[(658, ca.cy), (658, Y1 - 34 + 0.3 * 68)], color=A_C, sw=1.5)
    arrow(d, cb.R, (cbs, 0, 0.7), pts=[(658, cb.cy), (658, Y1 - 34 + 0.7 * 68)], color=B_C, sw=1.5)

    # spectral residual blocks: A2S2K-style feature stacks + dashed identity + ⊕
    prev, x = (cbs, 1, 0.5), 782
    for k in range(2):
        g = stack(6, 22, 40, 5, 4, colors=SLICES[k * 2:] + SLICES[:k * 2], thick=3)
        d.glyph(x, Y1 - g.h / 2, g)
        px = x + g.w + 22
        d.op(px, Y1, "add", r=8)
        arrow(d, prev, (x, Y1))
        arrow(d, (x + g.w, Y1), (px - 8, Y1))
        d.edge((x - 12, Y1), (px, Y1 - 8), pts=[(x - 12, Y1 - 50), (px, Y1 - 50)], color=A_C, dashed=True,
               sw=1.3)
        prev, x = (px + 8, Y1), px + 24
    d.edge((770, 198), (962, 198), color=RES_C, dashed=True, sw=1.1, arrow=None)
    d.text(744, 202, 236, 40, "Spectral ResBlocks ×2<br>" + m(r"1{\times}1{\times}7") + " Conv3D + ECA",
           fs=FS_S, color=RES_C)

    pe = d.block(976, Y1 - 38, 86, 76, "Spectral<br>AvgPool<br>Conv " + m(r"1{\times}1{\times}1"),
                 STD_C, fs=FS_X - 1, bold=False)
    arrow(d, prev, (pe, 0, 0.5))

    # hand-off Stage I -> Stage II
    d.fat_arrow(1004, 190, 30, 194, color="#8A8A8A", direction="south")
    d.text(880, 298, 118, 40, m(r"K{\to}128,\ \text{squeeze}") + "<br>" + m(r"128{\times}P{\times}P"),
           fs=FS_X, color=GREY, align="right")

    # ---------------- Stage II (flows right -> left, U-shaped layout)
    sc = d.block(979, Y2 - 32, 80, 64,"SCFG" + sub(m(r"f_{\mathrm{ch}}\!\otimes\! f_{\mathrm{sp}}")), SCFG_C,
                 fs=FS + 1)
    xs = Cub(d, 878, Y2, 40, 40, 30, 20, front=lighten(SCFG_C, 0.8), stroke=SCFG_C,
             stripes=cmap_hex("Purples", np.linspace(0.25, 0.75, 10)))
    xs.title(m(r"x_{\mathrm{scfg}}"), dy=-26)
    d.text(852, xs.y + xs.h + xs.dy + 2, 122, 18, m(r"128{\times}P{\times}P"), fs=FS_X, color=GREY)
    arrow(d, (sc, 0, 0.5), xs.R)

    se = d.block(758, Y2 - 30, 90, 60, "Conv3D<br>" + m(r"3{\times}3{\times}128"), STD_C, fs=FS_S, bold=False)
    arrow(d, xs.L, (se, 1, 0.5))

    cols2 = ["#C9DAF8", "#A4C2F4", "#6D9EEB", "#9FC5E8", "#CFE2F3", "#B4C7E7"]
    prev, x = (se, 0, 0.5), 734
    for k in range(2):
        g = stack(6, 30, 30, 5, 4, colors=cols2[k:] + cols2[:k], thick=3)
        gx = x - g.w
        d.glyph(gx, Y2 - g.h / 2, g)
        px = gx - 22
        d.op(px, Y2, "add", r=8)
        arrow(d, prev, (gx + g.w, Y2))
        arrow(d, (gx, Y2), (px + 8, Y2))
        d.edge((gx + g.w + 12, Y2), (px, Y2 - 8), pts=[(gx + g.w + 12, Y2 - 48), (px, Y2 - 48)], color=A_C,
               dashed=True, sw=1.3)
        prev, x = (px - 8, Y2), px - 24
    d.edge((528, 464), (742, 464), color=RES_C, dashed=True, sw=1.1, arrow=None)
    d.text(522, 468, 236, 40, "Spatial ResBlocks ×2<br>" + m(r"3{\times}3{\times}1") + " Conv3D + ECA",
           fs=FS_S, color=RES_C)

    gp = d.block(448, Y2 - 26, 58, 52, "GAP", STD_C, fs=FS_S, bold=False)
    arrow(d, prev, (gp, 1, 0.5))
    chain = stack(9, 10, 10, 5, 5, colors=SLICES)
    d.glyph(366, Y2 - chain.h / 2, chain)
    arrow(d, (gp, 0, 0.5), (366 + chain.w + 2, Y2))
    d.text(340, Y2 + chain.h / 2 + 2, 104, 20, "FC " + m(r"K{\to}N_c"), fs=FS_X, color=GREY)
    d.text(326, Y2 - 32, 40, 22, m(r"\hat{y}"), fs=FS + 1)

    ls = d.rect(212, 488, 252, 40, "Training: " + m(r"\mathcal{L}=0.7\,\mathcal{L}_{\mathrm{Spread}}"
                                                     r"+0.3\,\mathcal{L}_{\mathrm{Focal}}"),
                fill="#FFFFFF", stroke="#8C8C8C", dashed=True, fs=FS_X)
    d.edge((392, Y2 + chain.h / 2 + 24), (ls, 0.71, 0), color="#8C8C8C", dashed=True, sw=1.1)

    mp = d.image(28, 344, 132, 132, file_png_uri(os.path.join(FIGS, "ip_sgscfnet.png")), stroke="#333333")
    d.text(4, 480, 180, 22, "Classification map", fs=FS)
    arrow(d, (366, Y2), (mp, 1, (Y2 - 344) / 132), sw=1.6)
    d.text(222, Y2 - 28, 96, 20, "softmax", fs=FS_X, color=GREY, italic=True)

    d.legend(200, 572, [("add", "Element-wise sum"), ("dash", "Identity (residual)"),
                        ("swatch:" + SBAM_C, ""), ("swatch:" + CBSA_C, ""), ("swatch:" + SCFG_C, "Proposed modules"),
                        (None, "Batch dim. omitted;  " + m(r"K{=}16,\ C_2{=}\lfloor (C{-}7)/2\rfloor{+}1"))],
             fs=FS_X)
    return d


# ============================================================================= FIG 2: SBAM
def fig_sbam():
    d = Diagram("fig_sbam", MOD_W, MOD_H)
    d.shift(26, 6)
    Y = 140
    d.panel(156, 30, 296, 234, "SBAM", lighten(SBAM_C, 0.92), SBAM_C, fs=FS_T, title_align="right")
    x = Cub(d, 36, Y, 60, 60, 48, 32, **PATCH_KW)
    x.dims(left="P", bottom="P", depth="C", bottom_f=0.3)
    x.title(m("X") + " " + m(r"\in\mathbb{R}^{P\times P\times C}"), dy=-28, w_extra=60)

    nn, pos = neurons([7, 3, 7], 6.5, 52, 4, ["#CFE2F3", lighten(SBAM_C, 0.55), "#CFE2F3"], edge="#9DB89F")
    nx, ny = 206, Y - nn.h / 2
    d.glyph(nx, ny, nn)
    arrow(d, x.R, (nx - 1, Y))
    d.text(x.R[0] + 4, Y - 26, 60, 20, "GAP", fs=FS_S, color=GREY)
    for k, lab in enumerate([r"z\,{\in}\,\mathbb{R}^{C}", "C/r", "C"]):
        d.text(nx + pos[k][0][0] - 34, ny + nn.h + 4, 68, 20, m(lab), fs=FS_X, color=GREY)
    d.text(nx - 24, ny - 28, 160, 20, "FC" + m(r"\to") + "ReLU" + m(r"\to") + "FC", fs=FS_X, color=GREY)

    bx, cw, chh = 366, 15, 8.5
    for i, (c, w) in enumerate(zip(TURBO12, WTS)):
        d.rect(bx, Y - 51 + i * chh, cw, chh, fill=c, stroke="#333333", sw=0.6, rounded=False,
               extra=f"opacity={int(25 + 75 * w)};")
    arrow(d, (nx + nn.w + 1, Y), (bx, Y))
    d.text(nx + nn.w + 2, Y - 26, 34, 20, m(r"\sigma"), fs=FS)
    d.text(bx - 30, Y + 54, 76, 20, m(r"w\in(0,1)^{C}"), fs=FS_X)

    mul = d.op(420, Y, "dot", r=10)
    arrow(d, (bx + cw, Y), (409, Y))
    xt = Cub(d, 470, Y, 54, 54, 44, 28, stripes=VEIL, stripe_alpha=VEIL_A, **PATCH_KW)
    xt.dims(bottom="P", depth="C")
    xt.title(m(r"\tilde{X}"), dy=-28)
    arrow(d, (431, Y), xt.L)
    bp = x.Bx(0.8)
    d.edge(bp, (mul, 0.5, 1), pts=[(bp[0], 286), (420, 286)], color=INK, dashed=True, sw=1.3)
    d.text(150, 288, 160, 20, "identity", fs=FS_X, color=GREY, italic=True)
    d.legend(4, 318, [("dot", "Hadamard product; " + m(r"w") + " broadcast over " + m(r"P{\times}P")),
                      (None, "spatial pooling: GAP over " + m(r"P{\times}P"))], fs=FS_X)
    return d


# ============================================================================= FIG 3: CBSA
def fig_cbsa():
    d = Diagram("fig_cbsa", MOD_W, MOD_H)
    d.shift(14, 0)
    Y = 176
    d.panel(108, 42, 424, 262, "CBSA", lighten(CBSA_C, 0.93), CBSA_C, fs=FS_T, title_align="left",
            title_valign="bottom")
    a = Cub(d, 16, 96, 44, 44, 36, 22, front=lighten(A_C, 0.72), stroke=A_C, sw=1.2)
    b = Cub(d, 16, 256, 44, 44, 36, 22, front=lighten(B_C, 0.72), stroke=B_C, sw=1.2)
    d.text(0, a.y + a.h + a.dy + 2, 104, 18, m(r"1{\times}1{\times}7") + " branch", fs=FS_X, color=A_C)
    d.text(0, b.y - 22, 104, 18, m(r"3{\times}3{\times}7") + " branch", fs=FS_X, color=B_C)
    d.text(a.x - 16, a.y - 6, 18, 22, m(r"\boldsymbol{a}"), fs=FS + 2, color=A_C)
    d.text(b.x - 16, b.y + b.h + b.dy - 18, 18, 22, m(r"\boldsymbol{b}"), fs=FS + 2, color=B_C)

    s1 = d.op(138, Y, "add", r=9)
    arrow(d, a.R, (s1, 0.5, 0), pts=[(138, a.cy)])
    arrow(d, b.R, (s1, 0.5, 1), pts=[(138, b.cy)])
    u = Cub(d, 164, Y, 42, 42, 32, 20, front=lighten(mix(A_C, B_C, 0.5), 0.78), stroke=mix(A_C, B_C, 0.5))
    u.title(m(r"U=a+b"), dy=-26)
    arrow(d, (149, Y), u.L)

    lc = local_conv(7, 6, 54, 4, 3, "#F3E3CF", lighten(CBSA_C, 0.45),
                    ["#E06666", "#F6B26B", "#6AA84F", "#3D85C6", "#8E7CC3", "#C27BA0", "#76A5AF"], lw=1.1)
    lx = 306
    d.glyph(lx, Y - lc.h / 2, lc)
    arrow(d, u.R, (lx - 1, Y))
    d.text(u.R[0] + 2, Y - 24, 64, 20, m(r"\mathrm{mean}_{P\times P}"), fs=FS_X, color=GREY)
    d.text(lx + lc.w / 2 - 50, Y - lc.h / 2 - 44, 100, 40, "Depthwise<br>Conv1D, " + m(r"k{=}3"), fs=FS_X,
           color=GREY)
    d.text(lx - 30, Y + lc.h / 2 + 2, 56, 20, m(r"U_{\mathrm{avg}}"), fs=FS_X)

    fx, ya, yb, cx0, mx = 394, 108, 244, 412, 512
    ch_a = circles(5, 5.5, 3, [lighten(A_C, 0.25), lighten(A_C, 0.55), A_C, lighten(A_C, 0.4),
                               lighten(A_C, 0.7)], vertical=False)
    ch_b = circles(5, 5.5, 3, [lighten(B_C, 0.6), lighten(B_C, 0.35), lighten(B_C, 0.75), lighten(B_C, 0.45),
                               lighten(B_C, 0.2)], vertical=False)
    d.glyph(cx0, ya - ch_a.h / 2, ch_a)
    d.glyph(cx0, yb - ch_b.h / 2, ch_b)
    arrow(d, (lx + lc.w + 1, Y), (cx0 - 1, ya), pts=[(fx, Y), (fx, ya)], color=A_C)
    arrow(d, (lx + lc.w + 1, Y), (cx0 - 1, yb), pts=[(fx, Y), (fx, yb)], color=B_C)
    d.text(fx + 4, Y - 24, 24, 22, m(r"\sigma"), fs=FS)
    d.text(cx0, ya - 32, ch_a.w, 22, m(r"\alpha"), fs=FS + 2, color=A_C)
    d.text(cx0 - 10, yb + 10, ch_b.w + 20, 22, m(r"1-\alpha"), fs=FS + 2, color=B_C)

    ma = d.op(mx, ya, "dot", r=9)
    mb = d.op(mx, yb, "dot", r=9)
    arrow(d, (cx0 + ch_a.w, ya), (ma, 0, 0.5), color=A_C)
    arrow(d, (cx0 + ch_b.w, yb), (mb, 0, 0.5), color=B_C)
    d.edge(a.T, (ma, 0.5, 0), pts=[(a.T[0], 24), (mx, 24)], color=A_C, dashed=True, sw=1.4)
    d.edge(b.Bx(), (mb, 0.5, 1), pts=[(b.Bx()[0], 326), (mx, 326)], color=B_C, dashed=True, sw=1.4)

    s2 = d.op(mx, Y, "add", r=9)
    arrow(d, (ma, 0.5, 1), (s2, 0.5, 0))
    arrow(d, (mb, 0.5, 0), (s2, 0.5, 1))
    d.text(mx - 96, ya + 16, 86, 20, m(r"\alpha\odot a"), fs=FS_X, color=A_C, align="right")
    d.text(mx - 116, yb - 38, 106, 20, m(r"(1{-}\alpha)\odot b"), fs=FS_X, color=B_C, align="right")

    stripes = [mix(B_C, A_C, w) for w in [0.85, 0.3, 0.6, 0.15, 0.9, 0.45, 0.7, 0.25]]
    v = Cub(d, 564, Y, 42, 42, 32, 20, front=lighten(mix(A_C, B_C, 0.5), 0.78), stroke="#555555",
            stripes=[lighten(c, 0.35) for c in stripes])
    arrow(d, (mx + 9, Y), v.L)
    d.text(mx + 10, Y - 26, 42, 20, "BN", fs=FS_X, color=GREY)
    v.title(m("v"), dy=-26, w_extra=0)
    d.legend(4, 344, [("add", "sum"), ("dot", "Hadamard (broadcast over " + m(r"P{\times}P") + ")"),
                      (None, m(r"a,b,U,v\in\mathbb{R}^{K\times P\times P\times C_2}"))], fs=FS_X, gap=18)
    return d


# ============================================================================= FIG 4: SCFG
def fig_scfg():
    d = Diagram("fig_scfg", MOD_W, MOD_H)
    d.shift(4, 0)
    Y, cw = 201, 16
    pur = cmap_hex("Purples", np.linspace(0.25, 0.8, 10))
    fch = [0.9, 0.35, 0.7, 0.15, 0.6, 0.3]
    fsp = [0.3, 0.8, 0.5, 1.0, 0.6, 0.2, 0.7, 0.4]
    gx, gy = 286, Y - 3 * cw
    G = matrix(6, 8, cw, cw, [[u * v for u in fsp] for v in fch], hi=SCFG_C)
    lin_x = gx + G.w + 22
    gtx = lin_x + 86 + 20
    d.panel(126, 18, gtx + 30 - 126, 296, "SCFG", lighten(SCFG_C, 0.93), SCFG_C, fs=FS_T, title_align="right")

    x = Cub(d, 30, Y, 48, 48, 40, 26, front="#EFE9F7", stroke=SCFG_C, stripes=pur)
    x.dims(left="P", bottom="P", depth=r"C_{\mathrm{mid}}", bottom_f=0.3)
    d.text(x.x - 26, x.y - 8, 26, 22, m("x"), fs=FS + 1)

    d.glyph(gx, gy, G)
    rowv = matrix(1, 8, cw, 12, fsp, hi=SCFG_C)
    colv = matrix(6, 1, 12, cw, fch, hi=SCFG_C)
    d.glyph(gx, gy - 22, rowv)
    d.glyph(gx - 22, gy, colv)
    d.text(gx - 32, gy - 34, 32, 32, m(r"\otimes"), fs=20)
    d.text(gx - 10, gy + G.h + 2, G.w + 20, 22, m(r"G\in\mathbb{R}^{C_{\mathrm{mid}}\times N}"), fs=FS_S)

    # spatial descriptor: mean over channels -> P x P map -> flatten -> f_sp
    sp = matrix(5, 5, 9, 9, list(np.clip(np.random.RandomState(3).rand(25), 0.1, 1)), hi=SCFG_C)
    spx, spy = 212, 40
    scy = spy + sp.h / 2
    d.glyph(spx, spy, sp)
    arrow(d, x.T, (spx - 1, scy), pts=[(x.T[0], scy)])
    d.text(132, scy - 24, 78, 20, m(r"\mathrm{mean}_{C_{\mathrm{mid}}}"), fs=FS_X, color=GREY)
    fcol = gx + 4 * cw
    arrow(d, (spx + sp.w + 1, scy), (fcol, gy - 23), pts=[(fcol, scy)])
    d.text(spx + sp.w + 6, scy - 24, 70, 20, "flatten", fs=FS_X, color=GREY, italic=True)
    d.text(fcol + 6, gy - 58, 110, 22, m(r"f_{\mathrm{sp}}\in\mathbb{R}^{N}"), fs=FS_S, align="left")

    # channel descriptor: mean over space -> f_ch
    arrow(d, x.R, (gx - 23, Y))
    d.text(136, Y - 24, 110, 20, m(r"\mathrm{mean}_{P\times P}"), fs=FS_X, color=GREY)
    d.text(gx - 130, Y + 6, 104, 22, m(r"f_{\mathrm{ch}}\in\mathbb{R}^{C_{\mathrm{mid}}}"), fs=FS_S,
           align="right")

    # normalise + shared Linear(N->1) + sigmoid -> channel gate g
    lin = d.block(lin_x, Y - 40, 86, 80, m(r"G/\Vert G\Vert_F") + "<br>Linear " + m(r"N{\to}1") + "<br>"
                  + m(r"\sigma"), SCFG_C, fs=FS_X, bold=False)
    arrow(d, (gx + G.w + 1, Y), (lin, 0, 0.5))
    gate = matrix(6, 1, 13, cw, [0.85, 0.3, 0.65, 0.2, 0.75, 0.45], hi=SCFG_C)
    d.glyph(gtx, gy, gate)
    arrow(d, (lin, 1, 0.5), (gtx - 1, Y))
    d.text(gtx - 22, gy - 26, 58, 22, m("g"), fs=FS + 1)

    gcx, my = gtx + gate.w / 2, 290
    mul = d.op(gcx, my, "dot", r=10)
    arrow(d, (gcx, gy + gate.h + 1), (mul, 0.5, 0))
    bp = x.Bx(0.8)
    d.edge(bp, (mul, 0, 0.5), pts=[(bp[0], my)], color=INK, dashed=True, sw=1.3)
    d.text(150, my + 2, 110, 20, "identity", fs=FS_X, color=GREY, italic=True)
    out = Cub(d, gcx + 30, my, 34, 34, 28, 18, front="#EFE9F7", stroke=SCFG_C,
              stripes=[mix(c, "#FFFFFF", 0.55 * (1 - w)) for c, w in
                       zip(pur, [0.9, 0.3, 0.7, 0.2, 0.8, 0.5, 0.6, 0.35, 0.95, 0.4])])
    arrow(d, (gcx + 11, my), out.L)
    out.title(m(r"x_{\mathrm{scfg}}"), dy=-26, w_extra=0)
    d.legend(14, 340, [("dot", "Hadamard (" + m("g") + " broadcast over " + m(r"P{\times}P") + ")"),
                       (None, m(r"\otimes") + " outer product;  " + m(r"N{=}P^2,\ C_{\mathrm{mid}}{=}128"))],
             fs=FS_X, gap=18)
    return d


if __name__ == "__main__":
    paths = [f().save(HERE) for f in (fig_overall, fig_sbam, fig_cbsa, fig_scfg)]
    for p in paths:
        print("wrote", p)
    if "--no-render" not in sys.argv:
        from render_drawio import find_webapp, find_browser, render
        wa, br = find_webapp(), find_browser()
        for p in paths:
            print("rendered", *render(p, wa, br))
