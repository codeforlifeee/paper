"""
generate_gap_diagram.py -- GAP & Classification Head architecture diagram.

Matches the design language of fig_sbam, fig_cbsa, fig_scfg from build_figures.py:
  - Same palette, fonts, arrow style, panel conventions
  - Same drawio_kit primitives (Diagram, cuboid, stack, block, etc.)
  - Same canvas: 660 x 360 px (-> figure*, width=0.75\textwidth)

Usage: python generate_gap_diagram.py [--no-render]
"""
import os, sys, subprocess

KIT = os.path.expanduser("~/.claude/skills/paper-architecture-diagrams/scripts")
sys.path.insert(0, KIT)
import drawio_kit
drawio_kit.FONT = "Times New Roman"
from drawio_kit import (Diagram, cuboid, stack, m, lighten, SLICES)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Shared palette (identical to build_figures.py) ----
SBAM_C, CBSA_C, SCFG_C = "#3E8E4E", "#D9822B", "#7A56B0"
A_C, B_C = "#2D6CD1", "#C93C32"
INK, GREY, STD_C = "#262626", "#555555", "#6B6B6B"
RES_C = "#4A6FA5"
FS, FS_S, FS_X, FS_T = 16, 14, 13, 17
MOD_W, MOD_H = 660, 360

# Stage II spatial-feature blues (from build_figures.py fig_overall)
COLS2 = ["#C9DAF8", "#A4C2F4", "#6D9EEB", "#9FC5E8", "#CFE2F3", "#B4C7E7"]


class Cub:
    """Cuboid placed by left x and bbox-centre y (same as build_figures.py)."""

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


def build_gap_diagram():
    d = Diagram("fig_gap", MOD_W, MOD_H)
    d.shift(10, 0)
    Y = 146

    # Panel (steel blue -- classification head, not a proposed module)
    d.panel(132, 20, 504, 232, "",
            lighten(RES_C, 0.94), RES_C, fs=FS_T, title_align="right")

    # ---- 1. Input: spatial ResBlock output s in R^{K x P x P} ----
    x = Cub(d, 24, Y, 50, 50, 38, 26,
            front=lighten(RES_C, 0.78), stroke=RES_C,
            stripes=COLS2 + COLS2[:2])
    x.dims(left="P", bottom="P", depth="K", bottom_f=0.3)
    x.title(m(r"\mathbf{s}") + " " + m(r"\in\mathbb{R}^{K\times P\times P}"),
            dy=-28, w_extra=60)
    d.text(x.x - 10, x.y + x.h + x.dy + 20, x.W + 20, 18,
           "Spatial ResBlock", fs=FS_X, color=RES_C, italic=True)

    # ---- 2. GAP (standard gray block, same style as overall fig) ----
    gap = d.block(160, Y - 26, 66, 52,
                  "GAP" + sub(m(r"\mathrm{mean}_{P\times P}")),
                  STD_C, fs=FS_S, bold=False)
    arrow(d, x.R, (gap, 0, 0.5))

    # Formula annotation
    d.rect(146, Y - 60, 94, 24,
           m(r"z_k = \tfrac{1}{P^2}\sum_{i,j}s_{k,i,j}"),
           fill="#FFFFFF", stroke=lighten(STD_C, 0.45), sw=0.8, fs=10)
    d.edge((193, Y - 36), (193, Y - 26), color=STD_C, dashed=True, sw=0.8)

    # ---- 3. Latent vector z in R^K (coloured stack, same glyph as overall) ----
    z = stack(8, 10, 10, 5, 5, colors=SLICES[:8])
    zx = 264
    d.glyph(zx, Y - z.h / 2, z)
    arrow(d, (gap, 1, 0.5), (zx - 1, Y))
    d.text(zx - 12, Y - z.h / 2 - 26, z.w + 24, 22,
           m(r"\mathbf{z}\in\mathbb{R}^{K}"), fs=FS_S)

    # ---- 4. FC Head (deep blue block, linear projection K -> N_c) ----
    fc = d.block(320, Y - 28, 74, 56,
                 "FC Head" + sub(m(r"K{\to}N_c")), A_C, fs=FS_S)
    arrow(d, (zx + z.w + 1, Y), (fc, 0, 0.5))

    # Formula annotation
    d.rect(310, Y - 60, 94, 24,
           m(r"\ell = W_{\!fc}\,z + b"),
           fill="#FFFFFF", stroke=lighten(A_C, 0.45), sw=0.8, fs=10)
    d.edge((357, Y - 36), (357, Y - 26), color=A_C, dashed=True, sw=0.8)

    # ---- 5. Softmax (standard gray, non-bold, matching overall style) ----
    sm = d.block(432, Y - 22, 52, 44, "Softmax", STD_C, fs=FS_X, bold=False)
    arrow(d, (fc, 1, 0.5), (sm, 0, 0.5))
    d.text(395, Y + 4, 38, 16, m(r"\ell\!\in\!\mathbb{R}^{N_c}"), fs=10, color=GREY)

    # ---- 6. Output probability vector y-hat (coloured stack) ----
    yh = stack(9, 10, 10, 5, 5, colors=SLICES)
    yx = 522
    d.glyph(yx, Y - yh.h / 2, yh)
    arrow(d, (sm, 1, 0.5), (yx - 1, Y))
    d.text(yx - 20, Y - yh.h / 2 - 26, yh.w + 40, 22,
           m(r"\hat{\mathbf{y}}\in[0,1]^{N_c}"), fs=FS_S)
    d.text(yx + yh.w + 8, Y - 14, 28, 22,
           m(r"\hat{c}"), fs=FS + 1, color=SBAM_C)

    # ---- 7. Loss box (dashed gray rect, matching overall diagram) ----
    ls = d.rect(92, 274, 396, 40,
                "Training: " + m(r"\mathcal{L}=0.7\,\mathcal{L}_{\mathrm{Spread}}"
                                r"+0.3\,\mathcal{L}_{\mathrm{Focal}}"),
                fill="#FFFFFF", stroke="#8C8C8C", dashed=True, fs=FS_X)
    d.edge((yx + yh.w / 2, Y + yh.h / 2 + 2), (ls, 0.78, 0),
           pts=[(yx + yh.w / 2, 260)],
           color="#8C8C8C", dashed=True, sw=1.1)

    # Ground-truth target label
    d.rect(18, 278, 58, 32, "Target " + m(r"y"),
           fill="#FFFFFF", stroke=GREY, sw=0.8, fs=11)
    d.edge((76, 294), (92, 294), color=GREY, dashed=True, sw=1.0)

    # ---- 8. Legend (matching SBAM / CBSA / SCFG convention) ----
    d.legend(4, 332, [
        (None, m(r"\mathbf{s}") + ": spatial branch output after " + m(r"\times") + "2 ResBlock + ECA;  "
               + m(r"K{=}16,\ N_c") + " = num. classes")
    ], fs=FS_X)

    return d


if __name__ == "__main__":
    print(f"Generating fig_gap in {HERE}...")
    d = build_gap_diagram()
    path = d.save(HERE)
    print("Saved", path)

    if "--no-render" not in sys.argv:
        render_script = os.path.join(KIT, "render_drawio.py")
        cmd = ["python", render_script, path, "--scale", "3"]
        print("Rendering to PDF and PNG...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("RC:", res.returncode)
        if res.stdout.strip():
            print("STDOUT:", res.stdout.strip())
        if res.returncode != 0 and res.stderr:
            print("STDERR:", res.stderr[:400])
