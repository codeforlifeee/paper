import os, sys, math
import numpy as np

KIT = os.path.expanduser('~/.claude/skills/paper-architecture-diagrams/scripts')
sys.path.insert(0, KIT)

import drawio_kit
drawio_kit.FONT = "Times New Roman"
from drawio_kit import Diagram, Glyph
import render_drawio

def star_glyph(r, fill, stroke=None, sw=1):
    points = []
    for i in range(10):
        angle = i * math.pi / 5 - math.pi / 2
        radius = r if i % 2 == 0 else r * 0.45
        points.append(f"{radius * math.cos(angle):.2f},{radius * math.sin(angle):.2f}")
    d_path = "M " + " L ".join(points) + " Z"
    w, h = 2 * r, 2 * r
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-r} {-r} {w} {h}"><path d="{d_path}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{sw}"/></svg>'
    return Glyph(svg, w, h)

# Palette extracted directly from Image 2
OA_C = "#790778"   # Deep Purple
AA_C = "#EFA217"   # Golden Orange
KC_C = "#E70B0B"   # Crimson Red
INK = "#222222"

def build_scarcity_chart(name, y_min, y_max, y_step, ratios, oa_vals, aa_vals, kc_vals,
                         leg_x=435, leg_y=245):
    # Compact canvas with no internal top heading
    W, H = 600, 435
    d = Diagram(name, W, H)
    
    # Plot frame coordinates (tight top margin since title is omitted)
    X0, Y0 = 80, 20
    W_plot, H_plot = 490, 345
    X1, Y1 = X0 + W_plot, Y0 + H_plot
    
    # Outer black box frame (matching Image 2)
    d.rect(X0, Y0, W_plot, H_plot, fill="#FFFFFF", stroke="#000000", sw=1.4, rounded=False)
    
    # Y-axis ticks and light gridlines
    y_vals = np.arange(y_min, y_max + 1e-5, y_step)
    for yv in y_vals:
        frac = (yv - y_min) / (y_max - y_min)
        yp = Y1 - frac * H_plot
        
        # Tick line
        d.edge((X0 - 6, yp), (X0, yp), color="#000000", sw=1.2, arrow=None)
        
        # Subtle dashed gridline inside plot
        if 0 < frac < 1:
            d.edge((X0, yp), (X1, yp), color="#EFEFEF", dashed=True, sw=0.9, arrow=None)
            
        # Tick label
        d.text(X0 - 64, yp - 10, 56, 20, f"{yv:.2f}", fs=12, align="right", color=INK)
        
    # Y-axis label
    d.text(6, Y0 + H_plot / 2 - 15, 30, 30, "Percentage (%)", fs=14, bold=True, color=INK, rotation=270)
    
    # X-axis positions
    n_pts = len(ratios)
    x_pad = 52
    x_step = (W_plot - 2 * x_pad) / (n_pts - 1)
    xs = [X0 + x_pad + i * x_step for i in range(n_pts)]
    
    for i, (xp, r_lab) in enumerate(zip(xs, ratios)):
        # Tick line
        d.edge((xp, Y1), (xp, Y1 + 6), color="#000000", sw=1.2, arrow=None)
        
        # Subtle vertical gridline
        d.edge((xp, Y0), (xp, Y1), color="#F5F5F5", dashed=True, sw=0.8, arrow=None)
        
        # Tick label
        d.text(xp - 30, Y1 + 8, 60, 20, r_lab, fs=13, align="center", color=INK)
        
    # X-axis label
    d.text(X0, Y1 + 34, W_plot, 24, "Training Ratio (%)", fs=14, bold=True, color=INK)
    
    def get_coords(vals):
        pts = []
        for xp, v in zip(xs, vals):
            frac = (v - y_min) / (y_max - y_min)
            yp = Y1 - frac * H_plot
            pts.append((xp, yp))
        return pts
    
    oa_pts = get_coords(oa_vals)
    aa_pts = get_coords(aa_vals)
    kc_pts = get_coords(kc_vals)
    
    # Connecting dashed lines
    for pts, col in [(oa_pts, OA_C), (aa_pts, AA_C), (kc_pts, KC_C)]:
        for i in range(len(pts) - 1):
            d.edge(pts[i], pts[i+1], color=col, dashed=True, sw=2.2, arrow=None)
            
    # Markers and Data Callout Labels
    for i in range(n_pts):
        xp = xs[i]
        yp_oa, v_oa = oa_pts[i][1], oa_vals[i]
        yp_aa, v_aa = aa_pts[i][1], aa_vals[i]
        yp_kc, v_kc = kc_pts[i][1], kc_vals[i]
        
        # 1. OA Circle Marker
        r = 5.0
        d.rect(xp - r, yp_oa - r, 2*r, 2*r, fill=OA_C, stroke=OA_C, rounded=True, arc=r)
        
        # 2. AA Star Marker
        r_s = 6.5
        d.glyph(xp - r_s, yp_aa - r_s, star_glyph(r_s, AA_C))
        
        # 3. Kappa Square Marker
        s = 8.5
        d.rect(xp - s/2, yp_kc - s/2, s, s, fill=KC_C, stroke=KC_C, rounded=False)
        
        # Callout texts with smart collision avoidance
        # OA text always above OA marker
        d.text(xp - 30, yp_oa - 22, 60, 18, f"{v_oa:.2f}", fs=11.5, bold=True, color=OA_C)
        
        # Kappa text always below Kappa marker
        d.text(xp - 30, yp_kc + 8, 60, 18, f"{v_kc:.2f}", fs=11.5, bold=True, color=KC_C)
        
        # AA text:
        if abs(yp_aa - yp_oa) < 18 or abs(yp_aa - yp_kc) < 18:
            # offset slightly to the left
            d.text(xp - 56, yp_aa - 9, 50, 18, f"{v_aa:.2f}", fs=11.5, bold=True, color=AA_C, align="right")
        else:
            d.text(xp - 30, yp_aa - 20 if yp_aa > yp_oa else yp_aa + 8, 60, 18, f"{v_aa:.2f}", fs=11.5, bold=True, color=AA_C)
            
    # Legend Box
    leg_w, leg_h = 105, 88
    d.rect(leg_x, leg_y, leg_w, leg_h, fill="#FFFFFF", stroke="#333333", sw=1.0, rounded=False)
    
    # Legend entries
    ly1 = leg_y + 14
    d.edge((leg_x + 12, ly1), (leg_x + 36, ly1), color=OA_C, dashed=True, sw=2.0, arrow=None)
    d.rect(leg_x + 24 - 4, ly1 - 4, 8, 8, fill=OA_C, stroke=OA_C, rounded=True, arc=4)
    d.text(leg_x + 44, ly1 - 10, 45, 20, "OA", fs=12.5, bold=True, color=INK, align="left")
    
    ly2 = leg_y + 42
    d.edge((leg_x + 12, ly2), (leg_x + 36, ly2), color=AA_C, dashed=True, sw=2.0, arrow=None)
    d.glyph(leg_x + 24 - 5.5, ly2 - 5.5, star_glyph(5.5, AA_C))
    d.text(leg_x + 44, ly2 - 10, 45, 20, "AA", fs=12.5, bold=True, color=INK, align="left")
    
    ly3 = leg_y + 70
    d.edge((leg_x + 12, ly3), (leg_x + 36, ly3), color=KC_C, dashed=True, sw=2.0, arrow=None)
    d.rect(leg_x + 24 - 4, ly3 - 4, 8, 8, fill=KC_C, stroke=KC_C, rounded=False)
    d.text(leg_x + 44, ly3 - 10, 52, 20, "Kappa", fs=12.5, bold=True, color=INK, align="left")
    
    return d

FIG_DIR = r'C:\Users\LENOVO\Desktop\MinorProject_Final\ResearchPaperWrite\ACM_Template\figures'
ratios = ["1%", "2%", "5%", "10%"]

charts = [
    # Indian Pines (no top heading)
    ("fig_scarcity_ip", 60.0, 100.0, 5.0,
     [70.12, 81.55, 93.91, 97.84],
     [66.23, 76.57, 86.65, 93.43],
     [66.07, 78.86, 93.06, 97.55],
     445, 250),
     
    # Pavia University (no top heading)
    ("fig_scarcity_pu", 92.0, 100.0, 1.0,
     [95.41, 98.04, 99.51, 99.83],
     [94.48, 96.85, 99.09, 99.73],
     [93.90, 97.40, 99.35, 99.78],
     445, 250),
     
    # Salinas Valley (no top heading)
    ("fig_scarcity_sv", 96.0, 100.0, 0.5,
     [97.73, 99.11, 99.79, 99.93],
     [98.03, 99.19, 99.77, 99.93],
     [97.48, 99.00, 99.76, 99.92],
     445, 250)
]

wa, br = render_drawio.find_webapp(), render_drawio.find_browser()

for name, ymin, ymax, ystep, oa, aa, kc, lx, ly in charts:
    diag = build_scarcity_chart(name, ymin, ymax, ystep, ratios, oa, aa, kc, lx, ly)
    drawio_p = diag.save(FIG_DIR)
    print("Saved:", drawio_p)
    out_files = render_drawio.render(drawio_p, wa, br)
    print("Rendered:", out_files)

print("All 3 scarcity charts without top headings built and rendered successfully!")
