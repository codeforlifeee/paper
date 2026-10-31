"""3D OA surfaces over (patch size x batch size) for the 60-run SGSCF-Net grid search.

Data source (single source of truth): MASTER_README_120+runs_SGSCF_Net.html, first three ranked tables
(Indian Pines, Pavia University, Salinas Valley). Surface = shape-preserving PCHIP interpolation between
the measured runs (cannot overshoot the data, e.g. above 100 %); markers = the 20 measured runs.

usage: python plot_surface_oa.py   ->  surface_oa_3d.{pdf,png}, surface_oa_{ip,pu,sv}.{pdf,png}
"""
import html, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.abspath(os.path.join(HERE, "..", "..", "..", "MASTER_README_120+runs_SGSCF_Net.html"))
PATCH = [9, 11, 13, 15, 17]
BATCH = [32, 64, 128, 256]
DATASETS = [("ip", "Indian Pines"), ("pu", "Pavia University"), ("sv", "Salinas Valley")]


def load_grid():
    s = open(MASTER, encoding="utf-8").read()
    rows = [[html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
             for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)]
            for r in re.findall(r"<tr[^>]*>(.*?)</tr>", s, flags=re.S)]
    tables, cur = [], None
    for r in rows:
        if r and r[0] == "Rank":
            cur = {}
            tables.append(cur)
        elif cur is not None and len(r) >= 4 and re.fullmatch(r"\d+x\d+", r[1]):
            cur[(int(r[1].split("x")[0]), int(r[2]))] = float(r[3].rstrip("%"))
    grids = {}
    for (key, _), t in zip(DATASETS, tables[:3]):
        assert len(t) == 20, (key, len(t))
        grids[key] = np.array([[t[(p, b)] for b in BATCH] for p in PATCH])
    # cross-check against the published heatmap cells
    assert grids["ip"][0, 3] == 92.61 and grids["ip"][1, 0] == 98.05
    assert grids["pu"][2, 0] == 99.86 and grids["sv"][4, 0] == 100.00
    return grids


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix", "mathtext.rm": "Times New Roman", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# low -> high: pale mint -> aqua -> ocean teal -> royal blue -> deep navy (dark = high OA)
CMAP = colors.LinearSegmentedColormap.from_list(
    "ocean_net", ["#DDF5EA", "#A6E6CD", "#62CFC0", "#2DAAC7", "#2A80C3", "#2659A8", "#1B3A82"])
NET = "#FFFFFF"          # light net reads on the dark (high-OA) surface
FLOOR_LINES = "#0A1A3F"
GAMMA = 1.8              # OA clusters near each dataset's maximum: stretch colour over the high range


def draw(ax, Z, fig, cax):
    xb = np.log2(BATCH)                                   # batch on a log2 axis: 32, 64, 128, 256 equally spaced
    P = np.array(PATCH, float)
    f = RegularGridInterpolator((P, xb), Z, method="pchip")
    pf, bf = np.linspace(P[0], P[-1], 81), np.linspace(xb[0], xb[-1], 61)
    PP, BB = np.meshgrid(pf, bf, indexing="ij")
    ZZ = f(np.stack([PP, BB], -1))

    zmin, zmax = Z.min(), Z.max()
    span = zmax - zmin
    floor = zmin - 0.45 * span
    norm = colors.PowerNorm(GAMMA, zmin, zmax)

    ax.computed_zorder = False
    ax.contourf(PP, BB, ZZ, zdir="z", offset=floor, levels=16, cmap=CMAP, norm=norm, alpha=0.9, zorder=0)
    ax.contour(PP, BB, ZZ, zdir="z", offset=floor, levels=8, colors=FLOOR_LINES, linewidths=0.25, alpha=0.3,
               zorder=0)
    # soft directional lighting for depth, colours still keyed to the colour bar
    shaded = colors.LightSource(azdeg=225, altdeg=38).shade(ZZ, cmap=CMAP, norm=norm, blend_mode="soft",
                                                            vert_exag=2.0 / span, fraction=0.35)
    ax.plot_surface(PP, BB, ZZ, facecolors=shaded, rstride=1, cstride=1, linewidth=0, antialiased=True,
                    shade=False, zorder=1, rasterized=True)
    # net: every 5th fine-grid line in both directions (intersections include all 20 measured runs)
    ax.plot_wireframe(PP, BB, ZZ, rstride=5, cstride=5, color=NET, linewidth=0.35, alpha=0.6, zorder=2)

    fmt = "%.0f%%" if span > 3 else ("%.1f%%" if span > 0.6 else "%.2f%%")
    ax.set_xticks(P, [str(p) for p in PATCH])
    ax.set_yticks(xb, [str(b) for b in BATCH])
    ax.set_zlim(floor, zmax + 0.06 * span)
    ax.set_xlabel("Patch size", labelpad=-6)
    ax.set_ylabel("Batch size", labelpad=-6)
    ax.tick_params(axis="x", pad=-3.5)
    ax.tick_params(axis="y", pad=-3.5)
    ax.tick_params(axis="z", pad=1)
    ax.zaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
    ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.view_init(elev=24, azim=-128)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.97, 0.97, 0.98, 1))
        a.pane.set_edgecolor("#BBBBBB")
        a._axinfo["grid"].update(color="#D9D9D9", linewidth=0.4)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    # gamma-stretched bar: keep only ticks that sit >= 22 % of the bar apart (always keep the top value)
    cand = [t for t in matplotlib.ticker.MaxNLocator(8).tick_values(zmin, zmax) if zmin <= t <= zmax]
    ticks = []
    for t in reversed(cand):
        if not ticks or norm(ticks[-1]) - norm(t) >= 0.22:
            ticks.append(t)
    cb.set_ticks(sorted(ticks))
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f" if span > 1 else "%.2f"))
    cb.set_label("Overall accuracy, OA (%)", fontsize=7.5, labelpad=1.5)
    cb.outline.set_linewidth(0.4)


def main():
    G = load_grid()
    # combined figure (figure*, \textwidth ~ 7 in)
    fig = plt.figure(figsize=(7.16, 2.85))
    for k, (key, name) in enumerate(DATASETS):
        x0 = k / 3
        ax = fig.add_axes([x0 + 0.005, 0.17, 0.325, 0.83], projection="3d")
        cax = fig.add_axes([x0 + 0.075, 0.075, 0.19, 0.028])
        draw(ax, G[key], fig, cax)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"surface_oa_3d.{ext}"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    # individual panels (3-column subfigure layout, 0.315\textwidth each)
    for k, (key, name) in enumerate(DATASETS):
        fig = plt.figure(figsize=(2.4, 2.75))
        ax = fig.add_axes([0.0, 0.17, 1.0, 0.83], projection="3d")
        cax = fig.add_axes([0.2, 0.075, 0.6, 0.028])
        draw(ax, G[key], fig, cax)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(HERE, f"surface_oa_{key}.{ext}"), dpi=600, bbox_inches="tight",
                        pad_inches=0.03)
        plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
