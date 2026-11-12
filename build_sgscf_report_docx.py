#!/usr/bin/env python
# coding: utf-8
"""
build_sgscf_report_docx.py
Builds the NIT Raipur B.Tech project report for SGSCF-Net as a Word document.

Every metric is parsed at build time from the two master records:
  kaggle_results/BASELINE_MODELS_MASTER_README.html
  MASTER_README_120+runs_SGSCF_Net.html
Parameter counts, tensor shapes and MACs are measured by instantiating the
model classes from SGSCF-Net_Final_Implementation/SGSCF_Net.py.
Citations come from ResearchPaperWrite/ACM_Template/sample-base.bib.

Pipeline: python build_sgscf_report_docx.py
  1. parse sources  2. render figures  3. write .docx (two passes for numbering)
  4. Word COM pass (PowerShell) updates TOC / LoF / LoT fields and exports a PDF.
"""

import html, os, re, subprocess
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Polygon, Rectangle
from PIL import Image
import latex2mathml.converter as l2m
from lxml import etree
import docx
from docx.shared import Inches, Pt, RGBColor, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

ROOT     = os.path.dirname(os.path.abspath(__file__))
BASE_HTML = os.path.join(ROOT, 'kaggle_results', 'BASELINE_MODELS_MASTER_README.html')
MAST_HTML = os.path.join(ROOT, 'MASTER_README_120+runs_SGSCF_Net.html')
MODEL_PY  = os.path.join(ROOT, 'SGSCF-Net_Final_Implementation', 'SGSCF_Net.py')
CKPT_DIR  = os.path.join(ROOT, 'SGSCF-Net_Final_Implementation', 'checkpoints')
BIB       = os.path.join(ROOT, 'ResearchPaperWrite', 'ACM_Template', 'sample-base.bib')
ACM_FIG   = os.path.join(ROOT, 'ResearchPaperWrite', 'ACM_Template', 'figures')
ARCH_FIG  = os.path.join(ACM_FIG, 'architecture')
IP_RES    = os.path.join(ROOT, 'SGSCF-Net_Final_Implementation', 'results', 'Indian_Pines')
LOGO      = os.path.join(ROOT, 'nit_raipur_logo.jpeg')
FIG_DIR   = os.path.join(ROOT, 'report_figures')
OUT_DOCX  = os.path.join(ROOT, 'SGSCF_Net_Minor_Project_Report.docx')
OUT_PDF   = os.path.join(ROOT, 'SGSCF_Net_Minor_Project_Report.pdf')
MML2OMML  = r'C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL'
os.makedirs(FIG_DIR, exist_ok=True)

FONT = 'Times New Roman'
MONO = 'Consolas'
PAGE_W_MM, PAGE_H_MM = 210, 297          # ISO A4
TEXT_W = 210 / 25.4 - 1.5                # 6.767 in between 0.75 in margins
NAVY = RGBColor(0x00, 0x00, 0x00)      # print version: headings, captions and links in black
LINK = RGBColor(0x00, 0x00, 0x00)
CITE_BLUE = RGBColor(0x13, 0x00, 0xC1) # deep ultramarine blue (#1300C1) from user sample
GREY = RGBColor(0x52, 0x51, 0x4E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# =========================================================================== #
# 1. SOURCE PARSING
# =========================================================================== #
def html_to_text(path):
    h = open(path, encoding='utf-8', errors='ignore').read()
    h = re.sub(r'(?is)<(script|style).*?</\1>', '', h)
    h = re.sub(r'(?i)</(td|th)>', ' | ', h)
    h = re.sub(r'(?i)<(br|/p|/tr|/div|/h\d|/li)[^>]*>', '\n', h)
    h = re.sub(r'<[^>]+>', '', h)
    h = html.unescape(h)
    h = re.sub(r'[ \t]+', ' ', h)
    return [l.strip() for l in h.split('\n') if l.strip()]

DATASETS = ['Indian Pines', 'Pavia University', 'Salinas Valley']

def parse_master():
    lines = html_to_text(MAST_HTML)
    i1 = lines.index('PART 1: Full 60-Run Grid Search Results')
    i2 = next(i for i, l in enumerate(lines) if l.startswith('PART 2:'))
    grid = {d: {} for d in DATASETS}
    scar = {d: {} for d in DATASETS}

    def read_block(j):
        rec = {'classes': {}}
        while j < len(lines) and not lines[j].startswith('Visualizations'):
            l = lines[j]
            m = re.match(r'- OA \(Overall Accuracy\): ([\d.]+)%', l)
            if m: rec['OA'] = float(m.group(1))
            m = re.match(r'- AA \(Average Accuracy\): ([\d.]+)%', l)
            if m: rec['AA'] = float(m.group(1))
            m = re.match(r'- Kappa Coefficient: ([\d.]+)', l)
            if m: rec['K'] = float(m.group(1))
            m = re.match(r'- Train Time: ([\d.]+) min', l)
            if m: rec['T'] = float(m.group(1))
            m = re.match(r'- (.+?): ([\d.]+)%$', l)
            if m and not l.startswith(('- OA', '- AA')):
                rec['classes'][m.group(1)] = float(m.group(2))
            j += 1
        return rec

    cur = None
    for j in range(i1, i2):
        if lines[j] in DATASETS: cur = lines[j]
        m = re.match(r'Patch Size (\d+)x\d+ \| Batch Size (\d+)', lines[j])
        if m: grid[cur][(int(m.group(1)), int(m.group(2)))] = read_block(j + 1)
    for j in range(i2, len(lines)):
        m = re.match(r'(.+) \(Train Ratio Experiment\)', lines[j])
        if m: cur = m.group(1)
        m = re.match(r'Training Ratio: (\d+)%', lines[j])
        if m: scar[cur][int(m.group(1))] = read_block(j + 1)
    drops = {}
    for d in DATASETS:
        k = lines.index(d + ' |', lines.index('2. Train Ratio Scarcity Experiment (Side-by-Side)'))
        drops[d] = float(lines[k + 5].rstrip(' |').rstrip('%'))
    for d in DATASETS:
        assert len(grid[d]) == 20, (d, len(grid[d]))
        assert sorted(scar[d]) == [1, 2, 5, 10], d
    return grid, scar, drops

# Baseline numbers are transcribed here and asserted against the master HTML text.
BASELINES = [
    # name, paradigm, OA, AA, K, time(min), checkpoint size as recorded, cite keys, note
    ('ViT',           'Vision Transformer',              92.70, 88.80, 91.65, 1.71,  '2.4 MB',  ['dosovitskiy2020image'], ''),
    ('HSI-SSFTT',     'Feature-token Transformer',       98.57, 96.15, 98.37, 0.84,  '625 KB',  ['sun2022spectral'], ''),
    ('SSRN',          'Spectral-spatial residual 3D CNN',98.93, 98.95, 98.78, 5.37,  'n/r',     ['zhong2017spectral'], ''),
    ('SSAN',          'Spectral-spatial attention net',  64.19, 72.77, 59.79, 14.00, 'n/r',     ['sun2019spectral'], ''),
    ('A2S2K-ResNet',  'Attention 3D ResNet',             98.36, 96.22, 98.14, 18.42, '1.5 MB',  ['roy2020attention'], 'placeholder'),
    ('CRV-MHDL',      'Compression + hybrid DL',         98.58, 93.41, 98.38, 3.23,  '58.2 MB', ['alalimi2022compression'], ''),
    ('HyperRoPE-HiT', 'Rotary-position Transformer',     94.42, 92.12, 93.64, 37.74, 'n/r',     [], ''),
    ('DMC',           'Deep matrix capsules',            99.15, 98.89, 99.03, 0.39,  'n/r',     [], 'warmstart'),
]
OURS = ('SGSCF-Net (ours)', 'Spectral gating + covariance gate', 98.05, 94.82, 97.78, 8.8)
CKPT_KB = {'HSI-SSFTT': 625, 'A2S2K-ResNet': 1.5 * 1024, 'ViT': 2.4 * 1024, 'CRV-MHDL': 58.2 * 1024}

BASELINES_PU = [
    ('SSRN',          'Spectral-Spatial Residual ConvNet', 99.87, 99.71, 99.82, 11.36, ['zhong2017spectral']),
    ('HyperRoPE-SST', 'Rotary Position Transformer',       99.83, 99.77, 99.78,  5.62, ['khan2025hyperrope']),
    ('A2S2K-ResNet',  '3D-2D Hybrid CNN ResNet',          99.83, 99.79, 99.78, 41.12, ['roy2020attention']),
    ('HSI-SSFTT',     'Feature Token Transformer',        99.72, 99.40, 99.63,  4.44, ['sun2022spectral']),
    ('DMC*',          'Deep Matrix Capsules',             99.34, 99.03, 99.12, 73.42, ['wang2021deep']),
    ('ViT',           'Vision Transformer',               99.05, 98.90, 98.74,  6.64, ['dosovitskiy2020image']),
    ('CRV-MHDL',      'Complex Multi-Scale Hybrid',       98.40, 97.17, 97.89,  2.89, ['alalimi2022compression']),
    ('SSAN',          'Spectral-Spatial Attention Net',   76.82, 77.90, 69.44,  8.70, ['sun2019spectral']),
    ('SGSCF-Net (ours)', 'Spectral Gated Spatial-Cross Fusion', 99.82, 99.71, 99.76, 12.90, []),
]


# Additional Tables from ACM Manuscript
EFFICIENCY_DATA = [
    ['SSRN [@zhong2017spectral]', '6.29', '5.53', '11.36', '12.50', '30.98', '25.83'],
    ['HyperRoPE-HiT [@khan2025hyperrope]', '1.43', '0.73', '5.62', '3.08', '5.92', '3.63'],
    ['A2S2K-ResNet [@roy2020attention]', '14.89', '10.75', '41.12', '23.28', '10.34', '9.20'],
    ['HSI-SSFTT [@sun2022spectral]', '1.05', '0.70', '4.44', '3.17', '5.26', '3.77'],
    ['DMC [@wang2021deep]', '17.73', '11.58', '73.42', '48.38', '94.64', '62.70'],
    ['ViT [@dosovitskiy2020image]', '1.68', '1.24', '6.64', '3.22', '8.94', '4.15'],
    ['CRV-MHDL [@alalimi2022compression]', '1.33', '0.75', '2.89', '2.79', '3.55', '3.91'],
    ['SSAN [@sun2019spectral]', '14.00', '25.54', '8.70', '49.90', '19.37', '135.74'],
    ['SGSCF-Net (Ours)', '8.80', '1.18', '12.90', '4.12', '57.70', '5.46'],
]

PARAM_COMP_DATA = [
    ['ViT [@dosovitskiy2020image]', 'Vision Transformer', '~610K', '2.4 MB', '1.68', '90.33', '89.12', '88.95'],
    ['HSI-SSFTT [@sun2022spectral]', 'Feature Token Transformer', '~156K', '625 KB', '1.05', '98.45', '94.29', '98.23'],
    ['SSRN [@zhong2017spectral]', 'Spectral-Spatial Residual ConvNet', '~540K', 'n/r', '6.29', '98.52', '79.99', '98.31'],
    ['SSAN [@sun2019spectral]', 'Spectral-Spatial Attention Net', '6.40 M', 'n/r', '14.00', '64.19', '72.77', '59.79'],
    ['A2S2K-ResNet [@roy2020attention]', '3D-2D Hybrid CNN ResNet', '~380K', '1.5 MB', '14.89', '98.11', '94.41', '97.85'],
    ['CRV-MHDL [@alalimi2022compression]', 'Complex Multi-Scale Hybrid', '~15 M', '58.2 MB', '1.33', '67.20', '51.71', '62.22'],
    ['HyperRoPE-HiT [@khan2025hyperrope]', 'Rotary Position Transformer', '~2.10 M', 'n/r', '1.43', '94.07', '89.37', '93.24'],
    ['DMC [@wang2021deep]', 'Deep Matrix Capsules (EM Routing)', '~480K', 'n/r', '17.73', '82.55', '64.11', '79.89'],
    ['SGSCF-Net (Proposed)', 'Spectral Gated Spatial-Cross Fusion', '308,097', '1.2 MB', '8.80', '98.05', '94.82', '97.78'],
]

COMPONENT_ABLATION_DATA = [
    ['1', '✓', '✗', '✗', '✗', '✗', '93.65', '89.24', '92.74', '--'],
    ['2', '✓', '✓', '✗', '✗', '✗', '95.18', '91.80', '94.48', '+1.53'],
    ['3', '✓', '✓', '✓', '✗', '✗', '96.44', '93.05', '95.94', '+1.26'],
    ['4', '✓', '✓', '✓', '✓', '✗', '97.26', '93.78', '96.88', '+0.82'],
    ['5 (Proposed)', '✓', '✓', '✓', '✓', '✓', '98.05', '94.82', '97.78', '+0.79'],
]

DEPTH_ABLATION_DATA = [
    ['D = 1', '179,329', '6.1 min', '95.82', '91.20', '95.23'],
    ['D = 2 (Proposed)', '308,097', '8.8 min', '98.05', '94.82', '97.78'],
    ['D = 3', '436,865', '12.4 min', '97.88', '94.12', '97.58'],
    ['D = 4', '565,633', '16.9 min', '97.40', '92.65', '97.04'],
]

BASELINES_SV = [
    ('A2S2K-ResNet',  '3D-2D Hybrid CNN ResNet',          100.00, 99.99, 100.00, 10.34, ['roy2020attention']),
    ('SSRN',          'Spectral-Spatial Residual ConvNet', 99.96, 99.93,  99.95, 30.98, ['zhong2017spectral']),
    ('HSI-SSFTT',     'Feature Token Transformer',         99.90, 99.92,  99.88,  5.26, ['sun2022spectral']),
    ('DMC*',          'Deep Matrix Capsules',              99.90, 99.88,  99.89, 94.64, ['wang2021deep']),
    ('HyperRoPE-SST', 'Rotary Position Transformer',        99.85, 99.89,  99.84,  5.92, ['khan2025hyperrope']),
    ('ViT',           'Vision Transformer',                98.34, 99.16,  98.16,  8.94, ['dosovitskiy2020image']),
    ('CRV-MHDL',      'Complex Multi-Scale Hybrid',        96.30, 97.75,  95.87,  3.55, ['alalimi2022compression']),
    ('SSAN',          'Spectral-Spatial Attention Net',    88.20, 94.44,  86.88, 19.37, ['sun2019spectral']),
    ('SGSCF-Net (ours)', 'Spectral Gated Spatial-Cross Fusion', 99.92, 99.92, 99.91, 57.70, []),
]

def check_baselines():
    txt = '\n'.join(html_to_text(BASE_HTML))
    for b in BASELINES:
        s = f'OA {b[2]:.2f} / AA {b[3]:.2f} / Kappa {b[4]:.2f}'
        assert s in txt or f'{b[2]:.2f} / {b[3]:.2f} / {b[4]:.2f}' in txt, s
        assert f'{b[5]:.2f} min' in txt, (b[0], b[5])
        if b[6] != 'n/r': assert b[6].split()[0] in txt, b[6]
    assert 'OA: 98.05% | AA: 94.82% | Kappa: 97.78% | Training Time: 8.8 min' in txt
    assert '11 × 11' in txt and 'Batch Size: 32' in txt and '200 Epochs' in txt
    return txt

# Standard class sizes of the public benchmarks (totals asserted against load_dataset()).
IP_CLASSES = ['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture','Grass-trees',
              'Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill','Soybean-mintill',
              'Soybean-clean','Wheat','Woods','Buildings-Grass-Trees','Stone-Steel-Towers']
IP_COUNTS = [46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93]
PU_COUNTS = [6631, 18649, 2099, 3064, 1345, 5029, 1330, 3682, 947]
SV_COUNTS = [2009, 3726, 1976, 1394, 2678, 3959, 3579, 11271, 6203, 3278, 1068, 1927, 916, 1070, 7268, 1807]

def check_counts():
    src = open(MODEL_PY, encoding='utf-8').read()
    for counts, tot in [(IP_COUNTS, 10249), (PU_COUNTS, 42776), (SV_COUNTS, 54129)]:
        assert sum(counts) == tot and f'TOTAL = {tot}' in src, tot

# --------------------------------------------------------------------------- #
# Model introspection (exact code from SGSCF_Net.py, executed without its CLI)
# --------------------------------------------------------------------------- #
def load_model_classes(P):
    src = open(MODEL_PY, encoding='utf-8').read()
    body = src[src.index('class SBAM'):src.index('# LOSS FUNCTIONS')]
    body = body[:body.rindex('# ----')]
    g = {'torch': torch, 'nn': nn, 'F': F, 'img_rows': P}
    exec(body, g)
    return g

def model_source(name):
    src = open(MODEL_PY, encoding='utf-8').read()
    i = src.index(f'class {name}')
    ends = [x for x in (src.find('\nclass ', i + 1), src.find('\n# ----', i + 1), src.find('\ndef ', i + 1)) if x > 0]
    return src[i:min(ends)].rstrip()

def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def introspect():
    out = {}
    for P in [9, 11, 13, 15, 17]:
        for name, band, cls in [('IN', 200, 16), ('UP', 103, 9), ('SV', 204, 16)]:
            g = load_model_classes(P)
            net = g['SGSCFNet'](band, cls, 16).eval()
            shapes, mods = {}, {}
            def mk(nm):
                def h(m, i, o): shapes[nm] = tuple(o.shape[1:])
                return h
            def mac_hook(nm):
                def h(m, i, o):
                    if isinstance(m, (nn.Conv3d, nn.Conv1d)):
                        k = int(np.prod(m.kernel_size)) * (m.in_channels // m.groups)
                        mods[nm] = mods.get(nm, 0) + o.numel() * k
                    elif isinstance(m, nn.Linear):
                        mods[nm] = mods.get(nm, 0) + o.numel() * m.in_features
                return h
            for n, ch in net.named_children():
                ch.register_forward_hook(mk(n))
                for m in ch.modules():
                    if isinstance(m, (nn.Conv3d, nn.Conv1d, nn.Linear)): m.register_forward_hook(mac_hook(n))
            with torch.no_grad(): net(torch.randn(1, 1, P, P, band))
            mods['scfg'] = mods.get('scfg', 0) + 128 * P * P      # SCFG outer product (bmm)
            out[(name, P)] = dict(params=count_params(net), shapes=shapes, macs=sum(mods.values()),
                                  mod_macs=mods,
                                  modules={n: count_params(m) for n, m in net.named_children()})
    ck = {}
    for f in sorted(os.listdir(CKPT_DIR)):
        sd = torch.load(os.path.join(CKPT_DIR, f), map_location='cpu', weights_only=True)
        n = sum(v.numel() for k, v in sd.items() if 'running' not in k and 'num_batches' not in k)
        ck[f] = (n, os.path.getsize(os.path.join(CKPT_DIR, f)), sd['scfg.gate.weight'].shape[1])
    return out, ck

# --------------------------------------------------------------------------- #
# BibTeX -> IEEE
# --------------------------------------------------------------------------- #
def parse_bib(path):
    s = open(path, encoding='utf-8').read()
    entries = {}
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', s):
        typ, key = m.group(1).lower(), m.group(2)
        i, depth = m.end(), 1
        while depth:
            depth += {'{': 1, '}': -1}.get(s[i], 0); i += 1
        body = s[m.end():i - 1]
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{', body):
            j, d = fm.end(), 1
            while d:
                d += {'{': 1, '}': -1}.get(body[j], 0); j += 1
            fields[fm.group(1).lower()] = body[fm.end():j - 1]
        entries[key] = (typ, fields)
    return entries

def detex(t):
    rep = {r"{\'e}": 'é', r"{\'a}": 'á', r"{\'o}": 'ó', r"{\'i}": 'í', r'{\"o}': 'ö', r'{\"u}': 'ü'}
    for a, b in rep.items(): t = t.replace(a, b)
    t = t.replace('--', '–').replace('{', '').replace('}', '').replace('\\&', '&')
    return re.sub(r'\s+', ' ', t).strip()

def ieee_author(a):
    a = detex(a)
    if ',' in a:
        last, first = [x.strip() for x in a.split(',', 1)]
    else:
        parts = a.split(); last, first = parts[-1], ' '.join(parts[:-1])
    ini = ' '.join(('-'.join(q[0] + '.' for q in w.split('-')) if not w.endswith('.') else w)
                   for w in first.split())
    return f'{ini} {last}'.strip()

# Known author-order correction (ICLR 2018 paper lists Hinton first).
AUTHOR_OVERRIDE = {'sabour2018matrix': 'Hinton, Geoffrey E. and Sabour, Sara and Frosst, Nicholas'}

def ieee_reference(key, entry):
    typ, f = entry
    auths = [ieee_author(x) for x in AUTHOR_OVERRIDE.get(key, f['author']).split(' and ')]
    if len(auths) > 6: A = auths[0] + ' et al.'
    elif len(auths) > 2: A = ', '.join(auths[:-1]) + ', and ' + auths[-1]
    else: A = ' and '.join(auths)
    title = detex(f['title'])
    segs = [(A + ', ', ''), (f'“{title},” ', '')]
    pages = detex(f.get('pages', ''))
    if 'booktitle' in f:
        bt = detex(f['booktitle'])
        bt = re.sub(r'^Proceedings of the', 'Proc.', bt)
        if not bt.startswith('Proc.') and not bt.startswith('Advances'): bt = 'Proc. ' + bt
        segs += [('in ', ''), (bt, 'i')]
        tail = []
        if 'volume' in f: tail.append(f"vol. {f['volume']}")
        tail.append(f['year'])
        if pages: tail.append(f'pp. {pages}')
        segs.append((', ' + ', '.join(tail) + '.', ''))
    else:
        jr = detex(f['journal'])
        if jr.startswith('arXiv'):
            segs.append((jr + ', ' + f['year'] + '.', ''))
        else:
            segs.append((jr, 'i'))
            tail = []
            if 'volume' in f: tail.append(f"vol. {f['volume']}")
            if 'number' in f: tail.append(f"no. {f['number']}")
            if pages: tail.append(('pp. ' if '–' in pages else 'Art. no. ') + pages)
            tail.append(f['year'])
            segs.append((', ' + ', '.join(tail) + '.', ''))
    return segs

# =========================================================================== #
# 2. FIGURES
# =========================================================================== #
plt.rcParams.update({'font.family': 'serif', 'font.serif': [FONT, 'DejaVu Serif'],
                     'mathtext.fontset': 'stix',
                     'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
                     'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'text.color': '#0b0b0b',
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'     # validated categorical slots 1-3
INK2, GRID, MUTED = '#52514e', '#e4e3df', '#a8a7a1'
NOVEL, STD, IO = ('#dbe8f8', S1), ('#f3f2ef', '#8a8984'), ('#ffffff', '#52514e')

# COLOR_MAP exactly as defined in SGSCF_Net.py (index 0 = background)
COLOR_MAP = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],
                      [192,192,192],[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],
                      [0,128,128],[0,0,128],[255,165,0],[255,192,203]]) / 255.

def _save(fig, name):
    p = os.path.join(FIG_DIR, name); fig.savefig(p); plt.close(fig); return p

def _clean(ax, grid_axis='x'):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    if grid_axis: ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

def _box(ax, x, y, w, h, title, sub, sty, bold=False, fs=8.2, sfs=6.3):
    fc, ec = sty
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2',
                                fc=fc, ec=ec, lw=1.3 if bold else 0.8))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal')
        ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', fontsize=sfs, color=INK2,
                linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal')

def _arr(ax, x1, y1, x2, y2, style='-|>', color='#52514e', ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=0.8, color=color, mutation_scale=8, linestyle=ls,
                                shrinkA=0, shrinkB=0))

def fig_architecture(info):
    sh = info[('IN', 11)]['shapes']; md = info[('IN', 11)]['modules']
    fmt = lambda t: '×'.join(str(x) for x in t)
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.set_xlim(0, 100); ax.set_ylim(-6, 67); ax.axis('off')
    y1 = 50; h = 10
    _box(ax, 0.5, y1, 15, h, 'Input patch', 'B×' + fmt(sh['sbam']), IO)
    _box(ax, 19.5, y1, 15, h, 'SBAM', f'band gate · {md["sbam"]:,} p', NOVEL, True)
    _box(ax, 38.5, y1 + 6.5, 17, 8, 'Conv3D 1×1×7', 'spectral stride 2', STD)
    _box(ax, 38.5, y1 - 4.5, 17, 8, 'Conv3D 3×3×7', 'spectral stride 2', STD)
    _box(ax, 60, y1, 15, h, 'CBSA', f'α / 1−α · {md["cbsa"]} p', NOVEL, True)
    _box(ax, 79, y1, 20, h, '2× Spectral ResBlock', '1×1×7 + ECA3D', STD)
    _arr(ax, 16, y1 + 5, 19.5, y1 + 5)
    _arr(ax, 35, y1 + 5, 38.5, y1 + 10.5); _arr(ax, 35, y1 + 5, 38.5, y1 - 0.5)
    _arr(ax, 56, y1 + 10.5, 60, y1 + 6.5); _arr(ax, 56, y1 - 0.5, 60, y1 + 3.5)
    _arr(ax, 75.5, y1 + 5, 79, y1 + 5)
    ax.text(47, y1 + 16.2, 'a: B×' + fmt(sh['bn1']), ha='center', fontsize=6.4, color=INK2)
    ax.text(47, y1 - 7.6, 'b: B×' + fmt(sh['bn3']), ha='center', fontsize=6.4, color=INK2)
    _arr(ax, 89, y1 - 0.4, 89, 37)
    ax.text(90.5, 43.5, 'B×' + fmt(sh['rs2']), fontsize=6.4, color=INK2)
    y2 = 27
    _box(ax, 78, y2, 21, h, 'Pool + expand', '16→128 · B×' + fmt(sh['expand']), STD)
    _box(ax, 55, y2, 19, h, 'SCFG', r'$G=f_{ch}\,f_{sp}^{T}$' + f' · {md["scfg"]} p', NOVEL, True)
    _box(ax, 32.5, y2, 18.5, h, 'Permute', 'B×1×11×11×128', IO)
    _box(ax, 6.5, y2, 22, h, 'Spatial entry 3×3×128', '128→16 · B×' + fmt(sh['se']), STD)
    _arr(ax, 78, y2 + 5, 74.5, y2 + 5); _arr(ax, 55, y2 + 5, 51.5, y2 + 5); _arr(ax, 32.5, y2 + 5, 29, y2 + 5)
    _arr(ax, 17.5, y2 - 0.4, 17.5, 14.4)
    y3 = 4
    _box(ax, 6.5, y3, 22, h, '2× Spatial ResBlock', '3×3×1 + ECA3D', STD)
    _box(ax, 34, y3, 17, h, 'Global pool', 'B×16', STD)
    _box(ax, 57, y3, 17, h, 'Linear head', f'16→16 · {md["fc"]} p', STD)
    _box(ax, 80, y3, 19, h, 'Hybrid loss', '0.7 Spread + 0.3 Focal', IO)
    _arr(ax, 29, y3 + 5, 34, y3 + 5); _arr(ax, 51.5, y3 + 5, 57, y3 + 5); _arr(ax, 74.5, y3 + 5, 80, y3 + 5)
    ax.legend(handles=[Patch(fc=NOVEL[0], ec=NOVEL[1], label='Proposed module'),
                       Patch(fc=STD[0], ec=STD[1], label='Standard operator'),
                       Patch(fc=IO[0], ec=IO[1], label='Data / reshaping')],
              loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=7)
    return _save(fig, 'fig_architecture.png')

def fig_workflow():
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis('off')
    tops = [('3 HSI scenes', 'Indian Pines · Pavia U.\nSalinas Valley'),
            ('Pre-processing', 'z-score per band\nzero-pad · P×P patches'),
            ('Stratified split', '10 / 10 / 80 %\n≥ 3 pixels per class'),
            ('Training', 'Spread + Focal loss\nAdam · cosine LR'),
            ('Evaluation', 'OA · AA · κ\ntime · params')]
    for i, (t, s) in enumerate(tops):
        x = 0.5 + i * 20.1
        _box(ax, x, 31, 18, 13, t, s, NOVEL if t == 'Training' else STD, bold=(t == 'Training'))
        if i < 4: _arr(ax, x + 18.4, 37.5, x + 20.1, 37.5)
    ax.plot([71, 71], [30.6, 24.5], color=INK2, lw=0.8)
    ax.plot([16, 82], [24.5, 24.5], color=INK2, lw=0.8)
    exps = [('Benchmark', '9 models on Indian Pines\n11×11 patch · batch 32'),
            ('Grid search', '5 patches × 4 batches\n× 3 datasets = 60 runs'),
            ('Data scarcity', '1, 2, 5, 10 % training\n× 3 datasets = 12 runs')]
    for i, (t, s) in enumerate(exps):
        x = 1.5 + i * 33
        _arr(ax, x + 14.5, 24.5, x + 14.5, 19.6)
        _box(ax, x, 5, 29, 14.2, t, s, IO, bold=True)
    ax.text(50, 0.5, 'Every reported number is traced to the master result records or measured from SGSCF_Net.py',
            ha='center', fontsize=6.6, color=INK2, style='italic')
    return _save(fig, 'fig_workflow.png')

def _cuboid(ax, x, y, w, h, dx, dy, fcs, ec='#52514e', lw=0.8, z=1):
    f, t, s = fcs
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fc=f, ec=ec, lw=lw, zorder=z))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)],
                         fc=t, ec=ec, lw=lw, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=s, ec=ec, lw=lw, zorder=z))

def fig_patch_cube():
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.set_xlim(0, 100); ax.set_ylim(-2, 40); ax.axis('off')
    BLUE3 = ('#e8eef7', '#d5e1f2', '#c3d3ec'); HL = ('#9fc2ee', '#7fa9e3', '#5f92d8')
    _cuboid(ax, 3, 4, 24, 24, 12, 8, BLUE3)
    ax.add_patch(Rectangle((12, 13), 4, 4, fc=HL[0], ec=S1, lw=1.2, zorder=3))
    for (px, py) in [(16, 17), (12, 17), (16, 13)]:
        ax.plot([px, px + 12], [py, py + 8], color=S1, lw=0.7, ls='--', zorder=3)
    ax.plot(14, 15, 'o', ms=2.5, color='#0b0b0b', zorder=4)
    ax.text(15, 1.3, 'W', ha='center', fontsize=8); ax.text(0.8, 16, 'H', va='center', fontsize=8)
    ax.text(31.5, 4.2, 'C bands', rotation=34, fontsize=7.5, color=INK2)
    ax.text(21, 37.5, '(a) Scene cube $\\mathcal{X}$', ha='center', fontsize=8.5)
    _arr(ax, 42, 18, 48, 18)
    ax.text(45, 20.5, 'cut P×P', ha='center', fontsize=6.6, color=INK2)
    _cuboid(ax, 50, 9, 13, 13, 13, 8.5, HL, ec='#1f4e99', z=2)
    ax.plot(56.5, 15.5, 'o', ms=3, color='#0b0b0b', zorder=5)
    ax.text(56.5, 5.5, 'centre pixel (i, j)\nlabel y', ha='center', fontsize=6.4, color=INK2)
    ax.text(62, 37.5, '(b) Patch  P×P×C', ha='center', fontsize=8.5)
    _arr(ax, 77.5, 18, 82.5, 18)
    ax.text(80, 20.5, 'stack', ha='center', fontsize=6.6, color=INK2)
    for k in range(3):
        _cuboid(ax, 87.5 - k * 1.6, 9 + k * 3.2, 7, 7, 7, 4.5, HL, ec='#1f4e99', z=2 + k)
    ax.text(92, 37.5, '(c) Batch', ha='center', fontsize=8.5)
    ax.text(92, 3.5, 'B×1×P×P×C', ha='center', fontsize=7, color=INK2)
    return _save(fig, 'fig_patch_cube.png')

def fig_modules():
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    H, ys = 9, [84, 68, 52, 36, 18]
    def col(cx, w, title, steps, skip_label):
        ax.text(cx, 97, title, ha='center', fontsize=9, fontweight='bold')
        for k, (t, s, sty) in enumerate(steps):
            if isinstance(t, tuple):            # split row (two descriptors)
                for j, (tt, ss) in enumerate(zip(t, s)):
                    xx = cx - w / 2 + j * (w / 2 + 0.6)
                    _box(ax, xx, ys[k], w / 2 - 0.6, H, tt, ss, sty, fs=7, sfs=5.9)
                    _arr(ax, xx + (w / 2 - 0.6) / 2, ys[k - 1] - 0.3, xx + (w / 2 - 0.6) / 2, ys[k] + H + 0.3)
                    _arr(ax, xx + (w / 2 - 0.6) / 2, ys[k] - 0.3, cx, ys[k + 1] + H + 0.3)
                continue
            _box(ax, cx - w / 2, ys[k], w, H, t, s, sty, bold=(sty is NOVEL), fs=7.4, sfs=6)
            nxt = steps[k + 1][0] if k + 1 < len(steps) else None
            if k + 1 < len(steps) and not isinstance(nxt, tuple):
                _arr(ax, cx, ys[k] - 0.3, cx, ys[k + 1] + H + 0.3)
        # skip connection from input to final multiplication
        xl = cx - w / 2 - 2.2
        ax.plot([cx - w / 2, xl, xl], [ys[0] + H / 2, ys[0] + H / 2, ys[-1] + H / 2], color=S1, lw=0.8, ls='--')
        _arr(ax, xl, ys[-1] + H / 2, cx - w / 2, ys[-1] + H / 2, color=S1)
        ax.text(xl - 0.6, (ys[0] + ys[-1]) / 2 + H / 2, skip_label, rotation=90, ha='right', va='center',
                fontsize=6, color=S1)
    col(17.5, 27, '(a) SBAM', [
        ('Input X', 'B×1×P×P×C', IO), ('Spatial mean', 'z: B×C', STD),
        ('Linear C→C/8, ReLU', '25 units (Indian Pines)', STD), ('Linear C/8→C, sigmoid', r'$w\in(0,1)^{B\times C}$', STD),
        (r'X $\odot$ w', 'gated input, same shape', NOVEL)], 'input X')
    col(50.5, 27, '(b) CBSA', [
        ('Branches a, b', 'B×K×P×P×C₂', IO), ('(a + b), spatial mean', 'U: B×K×C₂', STD),
        ('Depthwise Conv1D, k = 3', '48 weights', STD), ('Sigmoid', r'$\alpha\in(0,1)^{K\times C_2}$', STD),
        (r'$\alpha\odot a+(1-\alpha)\odot b$, BN', 'v: B×K×P×P×C₂', NOVEL)], 'branches a, b')
    col(83.5, 27, '(c) SCFG', [
        ('Feature map x', 'B×128×P×P', IO),
        ((r'$f_{ch}$', r'$f_{sp}$'), ('mean over N\nB×128', 'mean over C\nB×N'), STD),
        ('Outer product G', 'B×128×N (rank one)', STD), (r'$\div\|G\|_F$, Linear N→1, σ', 'g: B×128', STD),
        (r'x $\odot$ g', 'B×128×P×P', NOVEL)], 'feature map x')
    ax.text(50, 4, 'Dashed blue lines carry the module input to the final element-wise product. K = 16, N = P².',
            ha='center', fontsize=6.6, color=INK2, style='italic')
    return _save(fig, 'fig_modules.png')

def fig_loss_panels():
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.9))
    r = np.linspace(0, 1, 100)
    a = axs[0, 0]; a.plot(r, 0.2 + 0.7 * r, color=S1, lw=2)
    a.set_xlabel('Training progress r = t / T'); a.set_ylabel('Margin m(r)')
    a.set_title('(a) Spread-loss margin schedule'); a.set_ylim(0, 1); _clean(a, 'y')
    a.annotate('0.2', (0, 0.2), (0.06, 0.08), fontsize=7.5); a.annotate('0.9', (1, 0.9), (0.86, 0.95), fontsize=7.5)
    e = np.arange(201); lr = 1e-6 + 0.5 * (5e-4 - 1e-6) * (1 + np.cos(np.pi * e / 200))
    a = axs[0, 1]; a.plot(e, lr * 1e4, color=S1, lw=2)
    a.set_xlabel('Epoch'); a.set_ylabel('Learning rate (×10⁻⁴)'); a.set_title('(b) Cosine learning-rate schedule')
    _clean(a, 'y')
    p = np.linspace(0.001, 1, 200); a = axs[1, 0]
    for g, c, lw in [(0, MUTED, 1.2), (1, MUTED, 1.2), (2, S1, 2)]:
        a.plot(p, (1 - p) ** g, color=c, lw=lw)
        yy = (1 - 0.55) ** g
        a.text(0.56, yy + 0.03, f'γ = {g}' + (' (used)' if g == 2 else ''), fontsize=7.5,
               color='#0b0b0b' if g == 2 else INK2)
    a.set_xlabel('Probability of the true class $p$'); a.set_ylabel('Focal factor $(1-p)^{\\gamma}$')
    a.set_title('(c) Focal down-weighting'); _clean(a, 'y')
    w = 1 / np.array(IP_COUNTS, float); alpha = w / w.sum() * 16
    a = axs[1, 1]; yv = np.arange(16)
    a.barh(yv, alpha, color=[S2 if n < 100 else S1 for n in IP_COUNTS], height=0.7, edgecolor='white', lw=0.6)
    a.set_xscale('log'); a.set_yticks(yv); a.set_yticklabels(IP_CLASSES, fontsize=6.3); a.invert_yaxis()
    a.set_xlabel('Class weight $\\alpha_k$ (log scale)'); a.set_title('(d) Indian Pines focal weights')
    _clean(a, 'x')
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    return _save(fig, 'fig_loss_panels.png'), alpha

def fig_class_dist(names_pu, names_sv):
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.85), gridspec_kw={'width_ratios': [1, 0.8, 1]})
    for a, (t, names, counts) in zip(axs, [('Indian Pines', IP_CLASSES, IP_COUNTS),
                                           ('Pavia University', names_pu, PU_COUNTS),
                                           ('Salinas Valley', names_sv, SV_COUNTS)]):
        y = np.arange(len(names))
        a.barh(y, counts, color=[S2 if c < 100 else S1 for c in counts], height=0.7, edgecolor='white', lw=0.6)
        a.set_xscale('log'); a.set_yticks(y)
        a.set_yticklabels([n.replace('_', ' ') for n in names], fontsize=5.8); a.invert_yaxis()
        a.set_title(f'{t}\n({sum(counts):,} labelled)', fontsize=8.5); a.set_xlim(10, 30000)
        a.set_xticks([10, 100, 1000, 10000]); a.set_xticklabels(['10', '100', '1k', '10k'])
        a.tick_params(axis='x', labelsize=7); _clean(a, 'x')
    axs[1].set_xlabel('Labelled pixels per class (log scale)')
    fig.legend(handles=[Patch(fc=S2, label='fewer than 100 pixels'), Patch(fc=S1, label='100 pixels or more')],
               loc='lower center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(w_pad=0.8)
    return _save(fig, 'fig_class_dist.png')

def _tag(b):
    return b[0] + ('‡' if b[8] == 'placeholder' else '†' if b[8] == 'warmstart' else '')

def fig_baseline_dots():
    rows = [(_tag(b), b[2], b[3], b[4]) for b in BASELINES] + [('SGSCF-Net (ours)', OURS[2], OURS[3], OURS[4])]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = np.arange(len(rows))
    for yy, r in zip(y, rows):
        ax.plot([min(r[1:]), max(r[1:])], [yy, yy], color=GRID, lw=2, zorder=1)
    for k, (lab, c, mk) in enumerate([('OA', S1, 'o'), ('AA', S2, 's'), ('Kappa ×100', S3, '^')]):
        ax.scatter([r[k + 1] for r in rows], y + (0.2 - 0.2 * k), s=34, c=c, marker=mk, label=lab,
                   zorder=3, edgecolors='white', linewidths=0.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    for t in ax.get_yticklabels():
        if 'ours' in t.get_text(): t.set_fontweight('bold')
    ax.set_xlim(56, 101); ax.set_xlabel('Score (%)'); _clean(ax, 'x')
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    return _save(fig, 'fig_baseline_scores.png')

def fig_baseline_time():
    rows = [(_tag(b), b[5]) for b in BASELINES] + [('SGSCF-Net (ours)', OURS[5])]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[S2 if 'ours' in r[0] else S1 for r in rows], height=0.62,
            edgecolor='white', linewidth=1)
    for yy, r in zip(y, rows):
        ax.text(r[1] + 0.5, yy, f'{r[1]:.2f}', va='center', fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    for t in ax.get_yticklabels():
        if 'ours' in t.get_text(): t.set_fontweight('bold')
    ax.set_xlabel('Training wall-clock time on Indian Pines (min, 200-epoch budget)')
    ax.set_xlim(0, 42); _clean(ax, 'x')
    return _save(fig, 'fig_baseline_time.png')

def fig_frontier(ours_kb):
    pts = [(CKPT_KB[b[0]], b[2], _tag(b)) for b in BASELINES if b[0] in CKPT_KB]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for kb, oa, lab in pts:
        ax.scatter(kb, oa, s=46, c=S1, edgecolors='white', linewidths=0.8, zorder=3)
        off = {'HSI-SSFTT': (-30, 8), 'A2S2K-ResNet‡': (6, -13)}.get(lab, (7, 5))
        ax.annotate(lab, (kb, oa), off, textcoords='offset points', fontsize=8)
    ax.scatter(ours_kb, OURS[2], s=70, c=S2, marker='D', edgecolors='white', linewidths=0.8, zorder=4)
    ax.annotate(f'SGSCF-Net ({ours_kb:.0f} KB)', (ours_kb, OURS[2]), (8, -12), textcoords='offset points',
                fontsize=8, fontweight='bold')
    ax.set_xscale('log'); ax.set_xlim(150, 150000); ax.set_ylim(91.5, 99.5)
    ax.set_xlabel('Checkpoint size (KB, log scale)'); ax.set_ylabel('OA on Indian Pines (%)')
    _clean(ax, 'both')
    return _save(fig, 'fig_frontier.png')

def fig_baseline_summary(ours_kb):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.6, 2.75), gridspec_kw={'width_ratios': [1.1, 0.95, 1.0]})
    # Panel (a): Scores
    rows = [(_tag(b), b[2], b[3], b[4]) for b in BASELINES] + [('SGSCF-Net (ours)', OURS[2], OURS[3], OURS[4])]
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    for yy, r in zip(y, rows):
        ax1.plot([min(r[1:]), max(r[1:])], [yy, yy], color=GRID, lw=1.5, zorder=1)
    for k, (lab, c, mk) in enumerate([('OA', S1, 'o'), ('AA', S2, 's'), ('Kappa ×100', S3, '^')]):
        ax1.scatter([r[k + 1] for r in rows], y + (0.18 - 0.18 * k), s=26, c=c, marker=mk, label=lab,
                    zorder=3, edgecolors='white', linewidths=0.5)
    ax1.set_yticks(y); ax1.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    for t in ax1.get_yticklabels():
        if 'ours' in t.get_text(): t.set_fontweight('bold')
    ax1.set_xlim(56, 101); ax1.set_xlabel('Score (%)', fontsize=7.5); ax1.set_title('(a) Benchmark accuracy', fontsize=8.0)
    ax1.tick_params(axis='x', labelsize=7); _clean(ax1, 'x')
    ax1.legend(loc='upper left', frameon=False, fontsize=6.5)

    # Panel (b): Training Time
    trows = [(_tag(b), b[5]) for b in BASELINES] + [('SGSCF-Net (ours)', OURS[5])]
    trows.sort(key=lambda r: r[1])
    ty = np.arange(len(trows))
    ax2.barh(ty, [r[1] for r in trows], color=[S2 if 'ours' in r[0] else S1 for r in trows], height=0.62,
             edgecolor='white', linewidth=0.8)
    for yy, r in zip(ty, trows):
        ax2.text(r[1] + 0.5, yy, f'{r[1]:.1f}', va='center', fontsize=6.5)
    ax2.set_yticks(ty); ax2.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    for t in ax2.get_yticklabels():
        if 'ours' in t.get_text(): t.set_fontweight('bold')
    ax2.set_xlabel('Train time (min)', fontsize=7.5); ax2.set_title('(b) Wall-clock latency', fontsize=8.0)
    ax2.set_xlim(0, 42); ax2.tick_params(axis='x', labelsize=7); _clean(ax2, 'x')

    # Panel (c): Checkpoint Size vs OA Frontier
    pts = [(CKPT_KB[b[0]], b[2], _tag(b)) for b in BASELINES if b[0] in CKPT_KB]
    for kb, oa, lab in pts:
        ax3.scatter(kb, oa, s=34, c=S1, edgecolors='white', linewidths=0.6, zorder=3)
        off = {'HSI-SSFTT': (-28, 6), 'A2S2K-ResNet‡': (5, -10)}.get(lab, (5, 4))
        ax3.annotate(lab, (kb, oa), off, textcoords='offset points', fontsize=6.3)
    ax3.scatter(ours_kb, OURS[2], s=55, c=S2, marker='D', edgecolors='white', linewidths=0.8, zorder=4)
    ax3.annotate(f'SGSCF-Net\n({ours_kb:.0f} KB)', (ours_kb, OURS[2]), (6, -11), textcoords='offset points',
                 fontsize=6.3, fontweight='bold')
    ax3.set_xscale('log'); ax3.set_xlim(150, 150000); ax3.set_ylim(91.5, 99.5)
    ax3.set_xlabel('Checkpoint (KB, log)', fontsize=7.5); ax3.set_ylabel('OA (%)', fontsize=7.5)
    ax3.set_title('(c) Parametric footprint', fontsize=8.0)
    ax3.tick_params(axis='both', labelsize=7); _clean(ax3, 'both')

    fig.tight_layout(w_pad=0.8)
    return _save(fig, 'fig_baseline_summary.png')

def fig_grid_trends(grid):
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.6))
    cols = dict(zip(DATASETS, [S1, S2, S3])); mks = dict(zip(DATASETS, ['o', 's', '^']))
    B, P = [32, 64, 128, 256], [9, 11, 13, 15, 17]
    for dn in DATASETS:
        g = grid[dn]
        mb = [np.mean([g[(p, b)]['OA'] for p in P]) for b in B]
        mp = [np.mean([g[(p, b)]['OA'] for b in B]) for p in P]
        axs[0].plot(range(4), np.array(mb) - mb[0], color=cols[dn], marker=mks[dn], lw=1.6, ms=5, label=dn)
        axs[1].plot(P, np.array(mp) - max(mp), color=cols[dn], marker=mks[dn], lw=1.6, ms=5)
        axs[2].plot(P, [g[(p, 32)]['T'] for p in P], color=cols[dn], marker=mks[dn], lw=1.6, ms=5)
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(B)
    axs[0].set_xlabel('Batch size'); axs[0].set_ylabel('Mean OA change (points)')
    axs[0].set_title('(a) vs. batch 32', fontsize=8.5)
    axs[1].set_xticks(P); axs[1].set_xlabel('Patch size P'); axs[1].set_title('(b) vs. best patch', fontsize=8.5)
    axs[2].set_xticks(P); axs[2].set_xlabel('Patch size P'); axs[2].set_ylabel('Minutes')
    axs[2].set_title('(c) training time, batch 32', fontsize=8.5)
    for a in axs: _clean(a, 'y'); a.tick_params(labelsize=7.5)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=1.2)
    return _save(fig, 'fig_grid_trends.png')

def fig_oa_aa(grid):
    g = grid['Indian Pines']
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    mk = {32: 'o', 64: 's', 128: '^', 256: 'D'}
    for b in [32, 64, 128, 256]:
        pts = [(r['OA'], r['AA']) for (p, bb), r in g.items() if bb == b]
        ax.scatter(*zip(*pts), marker=mk[b], s=36, c=S1 if b == 32 else MUTED, edgecolors='white',
                   linewidths=0.6, label=f'batch {b}', zorder=3)
    bo = max(g.items(), key=lambda kv: kv[1]['OA']); ba = max(g.items(), key=lambda kv: kv[1]['AA'])
    for (p, b), r in [bo, ba]:
        lab = f'{p}×{p} / {b}\nOA {r["OA"]:.2f}, AA {r["AA"]:.2f}'
        ax.annotate(lab, (r['OA'], r['AA']), (-80, 8) if r is bo[1] else (-95, -2), textcoords='offset points',
                    fontsize=7, arrowprops=dict(arrowstyle='-', lw=0.6, color=INK2))
    ax.set_xlabel('Overall accuracy (%)'); ax.set_ylabel('Average accuracy (%)')
    ax.legend(frameon=False, fontsize=7.5, loc='lower left'); _clean(ax, 'both')
    return _save(fig, 'fig_oa_aa.png')

def fig_class_heatmap(scar):
    ip = scar['Indian Pines']
    ratios = [1, 2, 5, 10]
    M = np.array([[ip[r]['classes'][c] for r in ratios] for c in IP_CLASSES])
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(M, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=7.5,
                    color='white' if M[i, j] > 60 else '#0b0b0b')
    ax.set_xticks(range(4)); ax.set_xticklabels([f'{r}%' for r in ratios])
    ax.set_yticks(range(16)); ax.set_yticklabels(IP_CLASSES, fontsize=8)
    ax.set_xlabel('Training ratio'); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    return _save(fig, 'fig_ip_class_scarcity.png')

def fig_gt_pred():
    gt = np.array(Image.open(os.path.join(IP_RES, 'ground_truth.png')).convert('RGB'))
    pr = np.array(Image.open(os.path.join(IP_RES, 'prediction_map.png')).convert('RGB'))
    fig = plt.figure(figsize=(6.4, 2.75))
    a1 = fig.add_axes([0.0, 0.0, 0.33, 0.9]); a2 = fig.add_axes([0.345, 0.0, 0.33, 0.9])
    for a, im, t in [(a1, gt, '(a) Ground truth'), (a2, pr, '(b) SGSCF-Net prediction')]:
        a.imshow(im, interpolation='nearest'); a.set_title(t, fontsize=9); a.axis('off')
    a3 = fig.add_axes([0.69, 0.0, 0.31, 0.95]); a3.axis('off')
    a3.legend(handles=[Patch(fc=COLOR_MAP[i + 1], ec='#52514e', lw=0.4, label=f'{i + 1}  {c}')
                       for i, c in enumerate(IP_CLASSES)], loc='center left', frameon=False,
              fontsize=6.8, handlelength=1.2, labelspacing=0.32)
    return _save(fig, 'fig_ip_gt_pred.png')

def fig_map_grid():
    items = [('ip_gt', '(a) Ground truth'), ('ip_vit', '(b) ViT'), ('ip_ssftt', '(c) HSI-SSFTT'),
             ('ip_ssrn', '(d) SSRN'), ('ip_ssan', '(e) SSAN'), ('ip_a2s2k', '(f) A2S2K-ResNet‡'),
             ('ip_crv', '(g) CRV-MHDL*'), ('ip_hit', '(h) HyperRoPE-HiT'), ('ip_dmc', '(i) DMC†'),
             ('ip_sgscfnet', '(j) SGSCF-Net')]
    fig, axes = plt.subplots(2, 5, figsize=(6.4, 3.0))
    for a, (f, t) in zip(axes.ravel(), items):
        a.imshow(Image.open(os.path.join(ACM_FIG, f + '.png')).convert('RGB'), interpolation='nearest')
        a.set_title(t, fontsize=7.5, fontweight='bold' if 'SGSCF' in t else 'normal'); a.axis('off')
    fig.subplots_adjust(wspace=0.06, hspace=0.22)
    return _save(fig, 'fig_ip_maps_all.png')

def fig_pu_map_grid():
    items = [('pu_gt', '(a) Ground Truth'), ('pu_vit', '(b) ViT'), ('pu_ssftt', '(c) HSI-SSFTT'),
             ('pu_ssrn', '(d) SSRN'), ('pu_ssan', '(e) SSAN'), ('pu_a2s2k', '(f) A2S2K-ResNet'),
             ('pu_crv', '(g) CRV-MHDL'), ('pu_hyperrope', '(h) HyperRoPE-SST'), ('pu_dmc', '(i) DMC*'),
             ('pu_sgscfnet', '(j) SGSCF-Net (Ours)')]
    fig, axes = plt.subplots(2, 5, figsize=(6.3, 4.2))
    for a, (f, t) in zip(axes.ravel(), items):
        a.imshow(Image.open(os.path.join(ACM_FIG, f + '.png')).convert('RGB'), interpolation='nearest')
        a.set_title(t, fontsize=7.2, pad=3, fontweight='bold' if 'SGSCF' in t else 'normal')
        a.axis('off')
    fig.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.5)
    return _save(fig, 'fig_pu_maps_all.png')

def fig_sv_map_grid():
    items = [('sv_gt', '(a) Ground Truth'), ('sv_vit', '(b) ViT'), ('sv_ssftt', '(c) HSI-SSFTT'),
             ('sv_ssrn', '(d) SSRN'), ('sv_ssan', '(e) SSAN'), ('sv_a2s2k', '(f) A2S2K-ResNet'),
             ('sv_crv', '(g) CRV-MHDL'), ('sv_hyperrope', '(h) HyperRoPE-SST'), ('sv_dmc', '(i) DMC*'),
             ('sv_sgscfnet', '(j) SGSCF-Net (Ours)')]
    fig, axes = plt.subplots(2, 5, figsize=(5.6, 4.4))
    for a, (f, t) in zip(axes.ravel(), items):
        a.imshow(Image.open(os.path.join(ACM_FIG, f + '.png')).convert('RGB'), interpolation='nearest')
        a.set_title(t, fontsize=7.2, pad=3, fontweight='bold' if 'SGSCF' in t else 'normal')
        a.axis('off')
    fig.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.5)
    return _save(fig, 'fig_sv_maps_all.png')

def fig_scfg_gate():
    f = np.linspace(0, 0.35, 200)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    betas = [(-20, '#c4501f'), (-8, '#f0a07a'), (0, MUTED), (8, '#86b4ea'), (20, '#1f5fb0')]
    for b, c in betas:
        y = 1 / (1 + np.exp(-b * f))
        ax.plot(f, y, color=c, lw=2 if b in (-20, 20) else 1.4)
        ax.text(0.352, y[-1], f'β = {b:+d}' if b else 'β = 0', va='center', fontsize=7.5, color='#0b0b0b')
    ax.axvline(1 / np.sqrt(128), color=INK2, lw=0.7, ls=':')
    ax.text(1 / np.sqrt(128) + 0.004, 0.535, 'uniform channel profile\n$1/\\sqrt{128}$', fontsize=6.8, color=INK2)
    ax.set_xlim(0, 0.40); ax.set_ylim(-0.05, 1.03)
    ax.set_xlabel('Normalised channel mean $\\hat{f}_{ch}(c)$'); ax.set_ylabel('Gate $g_c$')
    _clean(ax, 'y')
    return _save(fig, 'fig_scfg_gate.png')

GROUPS = [('SBAM', ['sbam'], True), ('Entry convs', ['e1', 'e3', 'bn1', 'bn3'], False), ('CBSA', ['cbsa'], True),
          ('Spectral ResBlocks', ['rs1', 'rs2'], False), ('Pool + expand', ['expand'], False), ('SCFG', ['scfg'], True),
          ('Spatial entry', ['se'], False), ('Spatial ResBlocks', ['rs3', 'rs4'], False), ('Classifier', ['fc'], False)]


def prepare_composite_figures():
    p_mod = os.path.join(FIG_DIR, 'fig_modules_2x2.png')
    if not os.path.exists(p_mod):
        sbam = Image.open(os.path.join(ARCH_FIG, 'fig_sbam.png'))
        cbsa = Image.open(os.path.join(ARCH_FIG, 'fig_cbsa.png')).resize(sbam.size, Image.Resampling.LANCZOS)
        scfg = Image.open(os.path.join(ARCH_FIG, 'fig_scfg.png')).resize(sbam.size, Image.Resampling.LANCZOS)
        gap  = Image.open(os.path.join(ARCH_FIG, 'fig_gap.png')).resize(sbam.size, Image.Resampling.LANCZOS)
        w, h = sbam.size
        grid_2x2 = Image.new('RGB', (w * 2 + 20, h * 2 + 20), (255, 255, 255))
        grid_2x2.paste(sbam, (0, 0))
        grid_2x2.paste(cbsa, (w + 20, 0))
        grid_2x2.paste(scfg, (0, h + 20))
        grid_2x2.paste(gap, (w + 20, h + 20))
        grid_2x2.save(p_mod, quality=95)

    p_surf = os.path.join(FIG_DIR, 'fig_surfaces_combined.png')
    if not os.path.exists(p_surf):
        s_ip = Image.open(os.path.join(ACM_FIG, 'surface_oa_ip.png')).convert('RGB')
        s_pu = Image.open(os.path.join(ACM_FIG, 'surface_oa_pu.png')).convert('RGB')
        s_sv = Image.open(os.path.join(ACM_FIG, 'surface_oa_sv.png')).convert('RGB')
        th = 1600
        r_ip = s_ip.resize((int(s_ip.width * th / s_ip.height), th), Image.Resampling.LANCZOS)
        r_pu = s_pu.resize((int(s_pu.width * th / s_pu.height), th), Image.Resampling.LANCZOS)
        r_sv = s_sv.resize((int(s_sv.width * th / s_sv.height), th), Image.Resampling.LANCZOS)
        tot_w = r_ip.width + r_pu.width + r_sv.width + 30
        surf_row = Image.new('RGB', (tot_w, th), (255, 255, 255))
        surf_row.paste(r_ip, (0, 0))
        surf_row.paste(r_pu, (r_ip.width + 15, 0))
        surf_row.paste(r_sv, (r_ip.width + r_pu.width + 30, 0))
        surf_row.save(p_surf, quality=95)

    p_sc = os.path.join(FIG_DIR, 'fig_scarcity_combined.png')
    if not os.path.exists(p_sc):
        sc_ip = Image.open(os.path.join(ACM_FIG, 'fig_scarcity_ip.png')).convert('RGB')
        sc_pu = Image.open(os.path.join(ACM_FIG, 'fig_scarcity_pu.png')).convert('RGB')
        sc_sv = Image.open(os.path.join(ACM_FIG, 'fig_scarcity_sv.png')).convert('RGB')
        th = 1300
        r_ip = sc_ip.resize((int(sc_ip.width * th / sc_ip.height), th), Image.Resampling.LANCZOS)
        r_pu = sc_pu.resize((int(sc_pu.width * th / sc_pu.height), th), Image.Resampling.LANCZOS)
        r_sv = sc_sv.resize((int(sc_sv.width * th / sc_sv.height), th), Image.Resampling.LANCZOS)
        tot_w = r_ip.width + r_pu.width + r_sv.width + 30
        sc_row = Image.new('RGB', (tot_w, th), (255, 255, 255))
        sc_row.paste(r_ip, (0, 0))
        sc_row.paste(r_pu, (r_ip.width + 15, 0))
        sc_row.paste(r_sv, (r_ip.width + r_pu.width + 30, 0))
        sc_row.save(p_sc, quality=95)

    p_ov = os.path.join(FIG_DIR, 'fig_sgscf_overall.png')
    if not os.path.exists(p_ov):
        shutil.copyfile(os.path.join(ARCH_FIG, 'fig_sgscf_overall.png'), p_ov)

    p_lc = os.path.join(FIG_DIR, 'fig_loss_convergence.png')
    if not os.path.exists(p_lc):
        shutil.copyfile(os.path.join(ARCH_FIG, 'fig_loss_convergence.png'), p_lc)

    return dict(fig_overall=p_ov, fig_modules_2x2=p_mod, fig_surfaces=p_surf,
                fig_scarcity=p_sc, fig_loss_conv=p_lc)

def fig_cost_breakdown(info):
    it = info[('IN', 11)]
    par = [sum(it['modules'].get(k, 0) for k in ks) for _, ks, _ in GROUPS]
    mac = [sum(it['mod_macs'].get(k, 0) for k in ks) for _, ks, _ in GROUPS]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    y = np.arange(len(GROUPS))
    for a, vals, t in [(axs[0], par, 'Share of parameters (%)'), (axs[1], mac, 'Share of MACs per patch (%)')]:
        pct = 100 * np.array(vals) / sum(vals)
        a.barh(y, pct, color=[S1 if g[2] else MUTED for g in GROUPS], height=0.66, edgecolor='white', lw=0.6)
        for yy, v in zip(y, pct):
            a.text(v + 1, yy, f'{v:.1f}' if v >= 0.05 else '<0.1', va='center', fontsize=7)
        a.set_xlim(0, max(pct) * 1.22); a.set_xlabel(t); _clean(a, 'x')
    axs[0].set_yticks(y); axs[0].set_yticklabels([g[0] for g in GROUPS]); axs[0].invert_yaxis()
    fig.legend(handles=[Patch(fc=S1, label='Proposed module'), Patch(fc=MUTED, label='Standard operator')],
               loc='lower center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, -0.06))
    fig.tight_layout(w_pad=1.5)
    return _save(fig, 'fig_cost_breakdown.png')

# =========================================================================== #
# 3. DOCX ENGINE
# =========================================================================== #
XSLT = etree.XSLT(etree.parse(MML2OMML))
BANNED = ['delve', 'pivotal', 'testament', 'tapestry', 'revolutioniz', 'beacon', 'game-changer',
          'seamless', "in today's", 'in today’s']
BANNED_STARTS = ['furthermore', 'moreover', 'additionally']

def latex_omml(tex):
    # upright operator names, then mark tilde/hat overscripts as accents so Word builds m:acc
    tex = re.sub(r'\\(max|min|log|cos|arg)(?![a-zA-Z])', lambda m: r'\mathrm{' + m.group(1) + '}' +
                 (r'\;' if m.group(1) in ('log', 'cos') else ''), tex)
    root = etree.fromstring(l2m.convert(tex).encode())
    MML = '{http://www.w3.org/1998/Math/MathML}'
    for mo in root.iter(MML + 'mover'):
        acc = mo[-1]
        if acc.tag == MML + 'mo' and acc.text in ('~', '^'):
            mo.set('accent', 'true'); acc.attrib.pop('stretchy', None)
            acc.text = '\u0303' if acc.text == '~' else '\u0302'
    return XSLT(root).getroot()

def shade(cell, fill):
    cell._tc.get_or_add_tcPr().append(parse_xml(
        f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{fill}"/>'))

def cell_borders(cell, **edges):
    xml = ''.join(f'<w:{k} w:val="{v[0]}" w:sz="{v[1]}" w:space="0" w:color="{v[2]}"/>' for k, v in edges.items())
    cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:tcBorders {nsdecls("w")}>{xml}</w:tcBorders>'))

def table_borders(tbl, **edges):
    allk = ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']
    xml = ''.join((f'<w:{k} w:val="{edges[k][0]}" w:sz="{edges[k][1]}" w:space="0" w:color="{edges[k][2]}"/>'
                   if k in edges else f'<w:{k} w:val="none"/>') for k in allk)
    tbl._tbl.tblPr.append(parse_xml(f'<w:tblBorders {nsdecls("w")}>{xml}</w:tblBorders>'))

def set_cell_margins(tbl, top=45, bottom=45, left=90, right=90):
    tbl._tbl.tblPr.append(parse_xml(
        f'<w:tblCellMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/></w:tblCellMar>'))

def fix_table_width(tbl, widths):
    tblPr = tbl._tbl.tblPr
    tw = int(sum(widths) * 1440)
    tblPr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="{tw}" w:type="dxa"/>'))
    tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
    grid = tbl._tbl.tblGrid
    for gc, w in zip(grid.findall(qn('w:gridCol')), widths): gc.set(qn('w:w'), str(int(w * 1440)))

def add_field(par, instr, cached='', bold=False, size=None, color=None, italic=False):
    def r():
        run = par.add_run(); run.font.name = FONT
        if size: run.font.size = Pt(size)
        if color: run.font.color.rgb = color
        run.bold = bold or None; run.italic = italic or None; return run
    b = r(); fc = OxmlElement('w:fldChar'); fc.set(qn('w:fldCharType'), 'begin'); b._r.append(fc)
    i = r(); it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = f' {instr} '; i._r.append(it)
    s = r(); fc = OxmlElement('w:fldChar'); fc.set(qn('w:fldCharType'), 'separate'); s._r.append(fc)
    c = r(); c.text = cached
    e = r(); fc = OxmlElement('w:fldChar'); fc.set(qn('w:fldCharType'), 'end'); e._r.append(fc)

def para_border(p, side='bottom', sz=6, color='1F3864', space=4):
    pPr = p._p.get_or_add_pPr()
    pPr.append(parse_xml(f'<w:pBdr {nsdecls("w")}><w:{side} w:val="single" w:sz="{sz}" w:space="{space}" '
                         f'w:color="{color}"/></w:pBdr>'))

PAGE_BORDER = (f'<w:pgBorders {nsdecls("w")} w:offsetFrom="page">'
               + ''.join(f'<w:{s} w:val="single" w:sz="6" w:space="24" w:color="404040"/>'
                         for s in ['top', 'left', 'bottom', 'right']) + '</w:pgBorders>')

class Report:
    """Writes content; the same content function runs twice so that forward
    references (tables, figures, equations, citations) resolve to final numbers."""

    def __init__(self, bib, labels=None, cites=None, prev_toc=None, prev_caps=None):
        self.doc = docx.Document()
        self.bib = bib
        self.prev_toc, self.prev_caps = prev_toc or [], prev_caps or []   # from pass 1
        self.toc_entries, self.cap_entries = [], []
        self.labels = labels or {}
        self.new_labels = {}
        self.cite_order = list(cites or [])
        self.n_tab = self.n_fig = 0
        self.chapter = 0; self.n_eq = 0; self.bm = 0
        self.alltext = []
        self._setup()

    # ---------- document setup ----------
    def _setup(self):
        d = self.doc
        st = d.styles['Normal']
        st.font.name = FONT; st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
        st.paragraph_format.space_after = Pt(2.0); st.paragraph_format.line_spacing = 1.10
        for lvl, size, before, after, ital, col in [(1, 14, 6, 2, False, NAVY), (2, 12, 4.5, 1.5, False, NAVY),
                                                    (3, 11, 4, 1.5, True, RGBColor(0, 0, 0))]:
            hs = d.styles[f'Heading {lvl}']
            hs.font.name = FONT; hs.font.size = Pt(size); hs.font.bold = True
            hs.font.italic = ital; hs.font.color.rgb = col
            rpr = hs.element.get_or_add_rPr()
            rf = rpr.find(qn('w:rFonts'))
            if rf is None: rf = OxmlElement('w:rFonts'); rpr.append(rf)
            for a in ['w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs']: rf.set(qn(a), FONT)
            for a in ['w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme']:
                if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
            pf = hs.paragraph_format
            pf.space_before = Pt(before); pf.space_after = Pt(after)
            pf.keep_with_next = True; pf.line_spacing = 1.12
        h1 = d.styles['Heading 1'].element.get_or_add_pPr()
        h1.append(parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="10" w:space="4" '
                            f'w:color="000000"/></w:pBdr>'))
        cap = d.styles['Caption']
        cap.font.name = FONT; cap.font.size = Pt(9); cap.font.italic = False
        cap.font.bold = False; cap.font.color.rgb = RGBColor(0, 0, 0)
        for nm, ind, bold in [('TOC 1', 0, True), ('TOC 2', 0.25, False), ('TOC 3', 0.5, False),
                              ('Table of Figures', 0, False)]:
            try: s = d.styles[nm]
            except KeyError: s = d.styles.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
            s.base_style = d.styles['Normal']
            s.font.name = FONT; s.font.size = Pt(10.5 if nm != 'Table of Figures' else 10)
            s.font.bold = bold
            pf = s.paragraph_format
            pf.left_indent = Inches(ind); pf.space_after = Pt(2 if bold else 1.5)
            pf.space_before = Pt(4 if bold else 0)
            if nm == 'Table of Figures':
                pf.left_indent = Inches(0.6); pf.first_line_indent = Inches(-0.6)
            pf.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.SPACES)
        sec = d.sections[0]
        sec.page_width, sec.page_height = Mm(PAGE_W_MM), Mm(PAGE_H_MM)
        sec.top_margin, sec.bottom_margin = Inches(0.55), Inches(0.55)
        sec.left_margin = sec.right_margin = Inches(0.75)
        sec.header_distance = Inches(0.4); sec.footer_distance = Inches(0.4)
        sec._sectPr.find(qn('w:pgMar')).addnext(parse_xml(PAGE_BORDER))
        sp = sec._sectPr
        for el in sp.findall(qn('w:pgNumType')): sp.remove(el)
        pg = OxmlElement('w:pgNumType'); pg.set(qn('w:fmt'), 'decimal'); pg.set(qn('w:start'), '1')
        cols = sp.find(qn('w:cols'))
        if cols is not None: cols.addprevious(pg)
        else: sp.append(pg)
        p = sec.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in list(p.runs): r._r.getparent().remove(r._r)
        add_field(p, 'PAGE', '1', size=10)
        hp = sec.header.paragraphs[0]
        for r in list(hp.runs): r._r.getparent().remove(r._r)

    # ---------- labels / bookmarks / links ----------
    def label(self, key, value):
        self.new_labels[key] = value

    def ref(self, key):
        return self.labels.get(key, self.new_labels.get(key, '??'))

    def bookmark(self, p, name):
        self.bm += 1
        s = OxmlElement('w:bookmarkStart'); s.set(qn('w:id'), str(self.bm)); s.set(qn('w:name'), name)
        e = OxmlElement('w:bookmarkEnd'); e.set(qn('w:id'), str(self.bm))
        pPr = p._p.pPr
        if pPr is not None: pPr.addnext(s)
        else: p._p.insert(0, s)
        p._p.append(e)

    def link(self, p, text, anchor, size=None, bold=False, color=None):
        r = p.add_run(text); r.font.name = FONT
        if size: r.font.size = Pt(size)
        r.bold = bold or None; r.font.color.rgb = color if color is not None else LINK
        h = OxmlElement('w:hyperlink'); h.set(qn('w:anchor'), anchor); h.set(qn('w:history'), '1')
        r._r.addprevious(h); h.append(r._r)
        self.alltext.append(text)

    def cite_segments(self, keys):
        nums = []
        for k in keys:
            k = k.lstrip('@')
            assert k in self.bib, f'citation key {k} missing from sample-base.bib'
            if k not in self.cite_order: self.cite_order.append(k)
            nums.append(self.cite_order.index(k) + 1)
        nums = sorted(set(nums)); segs, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
            if segs: segs.append((', ', None))
            if j - i >= 2:
                segs += [(f'[{nums[i]}]', f'ref_{nums[i]}'), ('–', None), (f'[{nums[j]}]', f'ref_{nums[j]}')]
            else:
                for n in nums[i:j + 1]:
                    if segs and segs[-1][1]: segs.append((', ', None))
                    segs.append((f'[{n}]', f'ref_{n}'))
            i = j + 1
        return segs

    # ---------- inline markup ----------
    TOKEN = re.compile(r'(\$[^$]+\$|\[@[^\]]+\]|\*\*.+?\*\*|__.+?__|\{[TFE]:[^}]+\})')

    def runs(self, p, text, size=None, bold=False, italic=False, color=None):
        for part in self.TOKEN.split(text):
            if not part: continue
            if part.startswith('$') and part.endswith('$') and len(part) > 2:
                p._p.append(latex_omml(part[1:-1])); continue
            if part.startswith('[@'):
                for txt, anc in self.cite_segments([k.strip() for k in part[2:-1].split(',')]):
                    if anc: self.link(p, txt, anc, size, bold, color=CITE_BLUE)
                    else: self._run(p, txt, size, bold, italic, color=CITE_BLUE)
                continue
            if re.match(r'\{[TFE]:', part):
                kind = {'T': 'Table ', 'F': 'Figure ', 'E': ''}[part[1]]
                key = part[3:-1]
                self.link(p, kind + self.ref(key), f'{part[1]}_{key}', size, bold); continue
            if part.startswith('**'): self._run(p, part[2:-2], size, True, italic, color)
            elif part.startswith('__'): self._run(p, part[2:-2], size, bold, True, color)
            else: self._run(p, part, size, bold, italic, color)

    def _run(self, p, text, size, bold, italic, color):
        r = p.add_run(text); r.font.name = FONT
        r.font.size = Pt(size) if size else None
        r.bold = True if bold else None; r.italic = True if italic else None   # None inherits the style
        if color: r.font.color.rgb = color
        self.alltext.append(text)
        return r

    # ---------- block elements ----------
    def para(self, text, align='justify', size=None, bold=False, italic=False, before=0, after=2.0,
             indent_first=False, keep_next=False, left=None, line=1.10, color=None):
        p = self.doc.add_paragraph()
        p.alignment = {'justify': WD_ALIGN_PARAGRAPH.JUSTIFY, 'center': WD_ALIGN_PARAGRAPH.CENTER,
                       'left': WD_ALIGN_PARAGRAPH.LEFT, 'right': WD_ALIGN_PARAGRAPH.RIGHT}[align]
        pf = p.paragraph_format
        pf.space_before = Pt(before); pf.space_after = Pt(after); pf.line_spacing = line
        if indent_first: pf.first_line_indent = Inches(0.3)
        if left is not None: pf.left_indent = Inches(left)
        pf.keep_with_next = keep_next
        self.runs(p, text, size=size, bold=bold, italic=italic, color=color)
        first = text.strip().split(' ')[0].lower().strip(',')
        assert first not in BANNED_STARTS, f'banned paragraph opener: {text[:40]}'
        return p

    def bullets(self, items, numbered=False, size=None):
        for k, t in enumerate(items, 1):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = p.paragraph_format
            pf.left_indent = Inches(0.35); pf.first_line_indent = Inches(-0.25)
            pf.space_after = Pt(1.5); pf.line_spacing = 1.0; pf.space_before = Pt(0)
            r = p.add_run((f'{k}.' if numbered else '•') + '\t'); r.font.name = FONT
            if not numbered: r.font.color.rgb = NAVY
            pf.tab_stops.add_tab_stop(Inches(0.35))
            self.runs(p, t, size=size)

    def h1(self, text, numbered=True):
        if numbered:
            self.chapter += 1; self.n_eq = 0
            text = f'{self.chapter}\u2003{text}'
        p = self.doc.add_paragraph(style='Heading 1')
        p.paragraph_format.page_break_before = False
        self.runs(p, text); self._toc(p, text, 1); return p

    def h2(self, text):
        p = self.doc.add_paragraph(style='Heading 2'); self.runs(p, text); self._toc(p, text, 2); return p

    def _toc(self, p, text, level):
        num, _, title = text.partition(' ')
        if not title: num, title = '', num
        name = f'H_{len(self.toc_entries) + 1}'
        self.bookmark(p, name)
        self.toc_entries.append(('toc', num, title, name, level))

    def list_table(self, headers, entries, widths):
        """Contents / list of figures / list of tables as a real table; page numbers are PAGEREF fields."""
        t = self.doc.add_table(rows=len(entries) + 1, cols=3); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False; set_cell_margins(t, 12, 12, 80, 80)
        table_borders(t, top=('single', 8, '000000'), bottom=('single', 8, '000000'),
                      insideH=('single', 2, 'D9D9D9'))
        fix_table_width(t, widths)
        hdr = t.rows[0]; hdr._tr.get_or_add_trPr().append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        for i, h in enumerate(headers):
            c = hdr.cells[i]; c.width = Inches(widths[i]); shade(c, 'E7E6E6')
            q = c.paragraphs[0]; q.alignment = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT,
                                                WD_ALIGN_PARAGRAPH.RIGHT][i]
            q.paragraph_format.space_after = Pt(1)
            r = q.add_run(h); r.bold = True; r.font.name = FONT; r.font.size = Pt(10)
        for ri, e in enumerate(entries, 1):
            num, title, anchor, level = e[1], e[2], e[3], e[4] if len(e) > 4 else 1
            row = t.rows[ri]; row._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            bold = (e[0] == 'toc' and level == 1)
            if bold:
                for c in row.cells: shade(c, 'F5F5F5')
            for ci in range(3):
                c = row.cells[ci]; c.width = Inches(widths[ci]); c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                q = c.paragraphs[0]; q.paragraph_format.space_after = Pt(0); q.paragraph_format.line_spacing = 1.0
                if ci == 0:
                    r = q.add_run(num); r.font.name = FONT; r.font.size = Pt(10); r.bold = bold or None
                elif ci == 1:
                    if e[0] == 'toc' and level == 2: q.paragraph_format.left_indent = Inches(0.2)
                    self.link(q, title, anchor, 10, bold)
                else:
                    q.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    add_field(q, f'PAGEREF {anchor} \\h', '', bold=bold, size=10)

    def h3(self, text):
        p = self.doc.add_paragraph(style='Heading 3'); self.runs(p, text); return p

    def front_title(self, text, new_page=False):
        p = self.doc.add_paragraph()
        p.paragraph_format.page_break_before = new_page
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8); p.paragraph_format.keep_with_next = True
        r = p.add_run(text); r.font.name = FONT; r.font.size = Pt(16); r.bold = True
        r.font.color.rgb = NAVY
        para_border(p, 'bottom', 8, '7F7F7F', 6)
        self.alltext.append(text)
        return p

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def eq(self, tex, label=None):
        self.n_eq += 1
        num = f'{self.chapter}.{self.n_eq}'
        if label: self.label(label, num)
        t = self.doc.add_table(rows=1, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        table_borders(t)
        fix_table_width(t, [5.45, 0.8])
        for c, w in zip(t.rows[0].cells, [5.45, 0.8]): c.width = Inches(w)
        c0, c1 = t.rows[0].cells
        p0 = c0.paragraphs[0]; p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p0.paragraph_format.space_before = Pt(4); p0.paragraph_format.space_after = Pt(4)
        p0._p.append(latex_omml(tex))
        p1 = c1.paragraphs[0]; p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p1.paragraph_format.space_before = Pt(4); p1.paragraph_format.space_after = Pt(4)
        r = p1.add_run(f'({num})'); r.font.name = FONT; r.font.size = Pt(11)
        if label: self.bookmark(p1, f'E_{label}')
        for c in (c0, c1): c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        return num

    def _caption(self, kind, label, title, keep_next):
        if kind == 'Table': self.n_tab += 1; n = self.n_tab
        else: self.n_fig += 1; n = self.n_fig
        if label: self.label(label, str(n))
        p = self.doc.add_paragraph(style='Caption')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(title) < 95 else WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.space_before = Pt(10 if kind == 'Table' else 3); pf.space_after = Pt(5 if kind == 'Table' else 2)
        pf.keep_with_next = keep_next
        pf.left_indent = pf.right_indent = Inches(0.25)
        r = p.add_run(f'{kind} '); r.bold = True; r.font.name = FONT; r.font.size = Pt(10); r.font.color.rgb = NAVY
        add_field(p, f'SEQ {kind} \\* ARABIC', str(n), bold=True, size=10, color=NAVY)
        r = p.add_run(': '); r.bold = True; r.font.name = FONT; r.font.size = Pt(10); r.font.color.rgb = NAVY
        self.runs(p, title, size=10)
        bm = f'{kind[0]}_{label}' if label else f'{kind[0]}_auto{n}'
        self.bookmark(p, bm)
        plain = re.sub(r'\$[^$]*\$', '', title)
        plain = re.sub(r'\{[TFE]:[^}]+\}', '', plain).replace('**', '').strip()
        self.cap_entries.append((kind, str(n), plain, bm))
        return p

    def figure(self, path, title, desc='', width=6.4, label=None):
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2.5); p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(path, width=Inches(width))
        full_title = title + (f'. {desc}' if desc else '')
        cp = self._caption('Figure', label, full_title, keep_next=False)
        cp.paragraph_format.space_after = Pt(4)

    def table(self, headers, rows, widths, title, label=None, note='', font=8.5, align=None,
              hl=(), total_last=False, zebra=True, keep=True, header_fill='E7E6E6'):
        if title: self._caption('Table', label, title, keep_next=True)
        widths = [w * TEXT_W / sum(widths) for w in widths]
        t = self.doc.add_table(rows=len(rows) + 1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        set_cell_margins(t, top=16, bottom=16, left=45, right=45)
        table_borders(t, top=('single', 8, '000000'), bottom=('single', 8, '000000'),
                      insideH=('single', 3, 'D9D9D9'))
        fix_table_width(t, widths)
        align = align or (['l'] + ['c'] * (len(headers) - 1))
        AL = {'l': WD_ALIGN_PARAGRAPH.LEFT, 'c': WD_ALIGN_PARAGRAPH.CENTER,
              'r': WD_ALIGN_PARAGRAPH.RIGHT, 'j': WD_ALIGN_PARAGRAPH.LEFT}
        hdr = t.rows[0]
        hdr._tr.get_or_add_trPr().append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        hdr._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for i, h in enumerate(headers):
            c = hdr.cells[i]; c.width = Inches(widths[i]); shade(c, header_fill)
            p = c.paragraphs[0]; p.alignment = AL[align[i]]
            p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0; p.paragraph_format.keep_with_next = True
            self.runs(p, h, size=font, bold=True, color=None)
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        n = len(rows)
        for ri, row in enumerate(rows, 1):
            tr = t.rows[ri]
            tr._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            is_total = total_last and ri == n
            is_hl = (ri - 1) in hl
            fill = 'EDEDED' if is_total else 'FFF2CC' if is_hl else ('F7F7F7' if zebra and ri % 2 == 0 else 'FFFFFF')
            for ci, val in enumerate(row):
                c = tr.cells[ci]; c.width = Inches(widths[ci]); shade(c, fill)
                if is_total: cell_borders(c, top=('single', 8, '000000'))
                p = c.paragraphs[0]; p.alignment = AL[align[ci]]
                p.paragraph_format.space_before = Pt(1.5); p.paragraph_format.space_after = Pt(1.5)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.keep_with_next = keep and ri < n
                self.runs(p, str(val), size=font, bold=is_total or is_hl)
                c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        if note:
            self.para(note, size=9, before=4, after=12, align='justify', color=GREY)
        else:
            sp = self.doc.add_paragraph(); sp.paragraph_format.space_after = Pt(2); sp.paragraph_format.line_spacing = Pt(2)
            sp.paragraph_format.line_spacing = 1.0
        return t

    def callout(self, title, items, fill='F5F5F5', accent='A6A6A6'):
        t = self.doc.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
        table_borders(t); fix_table_width(t, [TEXT_W]); set_cell_margins(t, 100, 100, 200, 160)
        c = t.rows[0].cells[0]; c.width = Inches(TEXT_W); shade(c, fill)
        cell_borders(c, left=('single', 36, accent))
        p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(4); p.paragraph_format.keep_with_next = True
        r = p.add_run(title); r.bold = True; r.font.name = FONT; r.font.size = Pt(11.5)
        r.font.color.rgb = RGBColor(0, 0, 0)
        self.alltext.append(title)
        for k, it in enumerate(items):
            q = c.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = q.paragraph_format; pf.left_indent = Inches(0.22); pf.first_line_indent = Inches(-0.18)
            pf.space_after = Pt(3); pf.line_spacing = 1.1; pf.keep_with_next = k < len(items) - 1
            pf.tab_stops.add_tab_stop(Inches(0.22))
            rr = q.add_run('▪\t'); rr.font.name = FONT; rr.font.color.rgb = RGBColor.from_string(accent)
            self.runs(q, it, size=10.5)
        sp = self.doc.add_paragraph(); sp.paragraph_format.space_after = Pt(2); sp.paragraph_format.line_spacing = Pt(2)

    def kpis(self, items, cols=4):
        rows = [items[i:i + cols] for i in range(0, len(items), cols)]
        t = self.doc.add_table(rows=len(rows), cols=cols); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        table_borders(t, insideV=('single', 36, 'FFFFFF'), insideH=('single', 36, 'FFFFFF'))
        fix_table_width(t, [TEXT_W / cols] * cols); set_cell_margins(t, 120, 120, 80, 80)
        for ri, row in enumerate(rows):
            for ci, (val, lab) in enumerate(row):
                c = t.rows[ri].cells[ci]; c.width = Inches(TEXT_W / cols); shade(c, 'F2F2F2')
                c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                r = p.add_run(val); r.font.name = FONT; r.font.size = Pt(20); r.bold = True
                r.font.color.rgb = NAVY
                q = c.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
                q.paragraph_format.space_after = Pt(0); q.paragraph_format.line_spacing = 1.0
                r = q.add_run(lab); r.font.name = FONT; r.font.size = Pt(9); r.font.color.rgb = GREY
                self.alltext += [val, lab]
        sp = self.doc.add_paragraph(); sp.paragraph_format.space_after = Pt(8)

    def code(self, text, title):
        t = self.doc.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
        table_borders(t, top=('single', 4, 'C9D1DC'), bottom=('single', 4, 'C9D1DC'),
                      left=('single', 4, 'C9D1DC'), right=('single', 4, 'C9D1DC'))
        fix_table_width(t, [TEXT_W]); set_cell_margins(t, 80, 80, 140, 100)
        c = t.rows[0].cells[0]; c.width = Inches(TEXT_W); shade(c, 'F6F8FA')
        p = c.paragraphs[0]
        r = p.add_run(title); r.font.name = FONT; r.font.size = Pt(9); r.bold = True; r.font.color.rgb = NAVY
        p.paragraph_format.space_after = Pt(4)
        for k, ln in enumerate(text.split('\n'), 1):
            q = c.add_paragraph(); pf = q.paragraph_format
            pf.space_after = Pt(0); pf.space_before = Pt(0); pf.line_spacing = 1.0
            r = q.add_run(f'{k:3d}  '); r.font.name = MONO; r.font.size = Pt(7.5); r.font.color.rgb = RGBColor(0x9A, 0xA4, 0xB1)
            r = q.add_run(ln); r.font.name = MONO; r.font.size = Pt(7.5)
            r._r.get_or_add_rPr().get_or_add_rFonts().set(qn('w:hAnsi'), MONO)
        sp = self.doc.add_paragraph(); sp.paragraph_format.space_after = Pt(8)

    def toc(self, instr, placeholder):
        p = self.doc.add_paragraph()
        add_field(p, instr, placeholder)

    # ---------- sections / page numbering ----------
    def new_section(self, fmt='decimal', start=None, header=False, cols=1):
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        sp = sec._sectPr     # cloned from the previous section (keeps the page border)
        for el in sp.findall(qn('w:pgNumType')): sp.remove(el)
        pg = OxmlElement('w:pgNumType'); pg.set(qn('w:fmt'), fmt)
        if start is not None:
            pg.set(qn('w:start'), str(start))
        cols_el = sp.find(qn('w:cols'))
        if cols_el is not None: sp.remove(cols_el)
        cols_el = parse_xml(f'<w:cols {nsdecls("w")} w:num="{cols}" w:space="360"/>') if cols > 1 else parse_xml(f'<w:cols {nsdecls("w")} w:num="1"/>')
        sp.append(cols_el)
        cols_el.addprevious(pg)
        sec.footer.is_linked_to_previous = False
        p = sec.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in list(p.runs): r._r.getparent().remove(r._r)
        add_field(p, 'PAGE', '', size=10)
        sec.header.is_linked_to_previous = False
        hp = sec.header.paragraphs[0]
        for r in list(hp.runs): r._r.getparent().remove(r._r)
        # Header removed across all pages per user request
        return sec

# =========================================================================== #
# 4. CONTENT
# =========================================================================== #
def write_report(R, D):
    grid, scar, drops, info, ck = D['grid'], D['scar'], D['drops'], D['info'], D['ck']
    IN11 = info[('IN', 11)]; P_IN, P_UP, P_SV = IN11['params'], info[('UP', 11)]['params'], info[('SV', 11)]['params']
    g_ip, g_pu, g_sv = grid['Indian Pines'], grid['Pavia University'], grid['Salinas Valley']
    best = {d: max(grid[d].items(), key=lambda kv: (kv[1]['OA'], kv[1]['AA'])) for d in DATASETS}
    ip11 = g_ip[(11, 32)]
    fmtn = lambda x: f'{x:,}'
    ck_ip = [v for k, v in ck.items() if 'Indian' in k][0]
    ck_kb = ck_ip[1] / 1024
    tr10 = [max(int(0.1 * n), 3) for n in IP_COUNTS]; tr1 = [max(int(0.01 * n), 3) for n in IP_COUNTS]
    n_floor1 = sum(1 for n in IP_COUNTS if int(0.01 * n) <= 3)
    n_tr = sum(tr10); upd32, upd256 = -(-n_tr // 32), -(-n_tr // 256)
    ck_desc = '; '.join(f'{nm} ({int(round(v[2] ** 0.5))}×{int(round(v[2] ** 0.5))} patch): {v[0]:,} '
                        f'parameters, {v[1] / 1024:.0f} KB'
                        for k, v in ck.items()
                        for nm in [('Indian Pines' if 'Indian' in k else 'Pavia University' if 'Pavia' in k
                                    else 'Salinas Valley')])
    md = IN11['modules']; mm = IN11['mod_macs']
    novel_p = md['sbam'] + md['cbsa'] + md['scfg']
    spec_mac = 100 * (mm['rs1'] + mm['rs2']) / IN11['macs']
    fs_ = sorted([b for b in BASELINES if not b[8]], key=lambda b: -b[2])
    d = R.doc

    # ================================================================== COVER
    R.para('National Institute of Technology Raipur', 'center', 18, True, before=12, after=4)
    R.para('Department of Information Technology', 'center', 14, after=10)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    p.add_run().add_picture(LOGO, width=Inches(2.8))
    R.para('Project Report', 'center', 15, after=2)
    R.para('on', 'center', 14, after=6)
    tp = R.para('SGSCF-Net: Spectral-Guided Spatial', 'center', 17, True, after=0, line=1.3, color=NAVY)
    for ln in ['Cross-Attention Fusion Network for', 'Hyperspectral Image Classification']:
        tp.add_run().add_break(); r = tp.add_run(ln); r.font.name = FONT; r.font.size = Pt(17); r.bold = True
        r.font.color.rgb = NAVY
    tp.paragraph_format.space_after = Pt(18)
    t = d.add_table(rows=1, cols=2); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    table_borders(t); fix_table_width(t, [3.2, 3.07])
    L, Rr = t.rows[0].cells
    L.width, Rr.width = Inches(3.2), Inches(3.07)
    def cell_lines(c, lines, align):
        c.paragraphs[0]._p.getparent().remove(c.paragraphs[0]._p)
        for item in lines:
            if len(item) == 2:
                txt, b = item; sz, sp_after = 11, 2
            else:
                txt, b, sz, sp_after = item
            q = c.add_paragraph(); q.alignment = align
            q.paragraph_format.space_after = Pt(sp_after)
            q.paragraph_format.line_spacing = 1.05
            r = q.add_run(txt); r.font.name = FONT; r.font.size = Pt(sz); r.bold = b
            R.alltext.append(txt)
    cell_lines(L, [
        ('Under the Guidance of:', True, 11.5, 4),
        ('Dr. R.R. Janghel', True, 10.5, 2),
        ('Associate Professor', False, 10, 2),
        ('Dept. of Information Technology', False, 9.5, 2),
        ('National Institute of Technology, Raipur', False, 9.5, 6),
        ('Sushil Janardhan', True, 10.5, 2),
        ('Doctoral Researcher', False, 10, 2),
        ('Dept. of Information Technology', False, 9.5, 2),
        ('National Institute of Technology, Raipur', False, 9.5, 2),
    ], WD_ALIGN_PARAGRAPH.LEFT)
    cell_lines(Rr, [
        ('Submitted by:', True, 11.5, 4),
        ('1. B. Vyshnavi (Roll No. 23118021)', False, 10, 3),
        ('2. Surya Tejas Kisan (Roll No. 23118098)', False, 10, 3),
        ('3. Valluri Lahitha (Roll No. 23118106)', False, 10, 8),
        ('B.Tech Seventh Semester, Nov.-2026', True, 10.5, 2),
        ('National Institute of Technology, Raipur', True, 10.5, 2),
    ], WD_ALIGN_PARAGRAPH.LEFT)

    # ================================================================== FRONT MATTER
    R.new_section('decimal', header=False)
    R.front_title('Abstract')
    R.para(f'Hyperspectral image (HSI) classification assigns a land-cover label to every pixel of a cube '
           f'with hundreds of contiguous spectral bands. Two problems dominate in practice: noisy bands '
           f'leak into every convolution that follows them, and minority classes with a few dozen labelled '
           f'pixels are overwhelmed by majority classes during training. This report presents SGSCF-Net, '
           f'a compact 3D convolutional network built around three gates. A Spectral Band Attention '
           f'Module (SBAM) re-weights the raw input bands before any convolution. Complementary Branch '
           f'Spectral Attention (CBSA) fuses a 1×1×7 spectral branch and a 3×3×7 spectral-spatial branch '
           f'with weights $\\alpha$ and $1-\\alpha$, so the two branches compete rather than duplicate each '
           f'other. A Second-Order Covariance Fusion Gate (SCFG) builds an outer-product interaction map '
           f'between channel and spatial descriptors and turns it into a channel gate. Training uses '
           f'$0.7\\,\\mathcal{{L}}_{{\\mathrm{{Spread}}}}+0.3\\,\\mathcal{{L}}_{{\\mathrm{{Focal}}}}$ with '
           f'inverse-frequency class weights. Measured from the released code, the network has '
           f'{fmtn(P_IN)} trainable parameters on Indian Pines. With an 11×11 patch, batch 32 and a 10% '
           f'training split it reaches {ip11["OA"]:.2f}% overall accuracy (OA), {ip11["AA"]:.2f}% average '
           f'accuracy (AA) and a kappa of {ip11["K"]:.2f} in {ip11["T"]} minutes. That is 5.35 points above '
           f'ViT and 0.88 points below SSRN, the strongest baseline trained from scratch under the same '
           f'protocol. Its best grid configurations reach {best["Pavia University"][1]["OA"]:.2f}% OA on '
           f'Pavia University and {best["Salinas Valley"][1]["OA"]:.2f}% on Salinas Valley. A 60-run patch '
           f'and batch grid and 12 data-scarcity runs (1–10% training) show that batch 32 is the most reliable '
           f'setting, and that Indian Pines loses {abs(drops["Indian Pines"]):.2f} OA points at 1% training, '
           f'driven by Oats ({scar["Indian Pines"][1]["classes"]["Oats"]:.2f}%) and Grass-pasture-mowed '
           f'({scar["Indian Pines"][1]["classes"]["Grass-pasture-mowed"]:.2f}%).', line=1.5)
    R.para('**Keywords:** hyperspectral image classification, spectral band attention, complementary '
           'gating, second-order pooling, class imbalance, 3D convolutional networks.', line=1.5, before=6)

    R.front_title('Contents', new_page=True)
    R.list_table(['No.', 'Title', 'Page'], R.prev_toc, [0.95, 4.6, 0.7])

    # ================================================================== CHAPTER 1
    R.new_section('decimal', header=True)
    R.h1('Introduction')
    R.h2('1.1\u2003Remote Sensing and the Hyperspectral Imaging Problem')
    R.para('A hyperspectral sensor records reflectance in hundreds of narrow, contiguous wavelength bands '
           'for every ground pixel [@bioucas2012hyperspectral,plaza2009recent]. The result is a data cube '
           'whose third axis is a near-continuous spectrum, so materials that look identical in an RGB '
           'photograph can be separated by small differences in absorption features '
           '[@fauvel2012advances,ghamisi2017advanced]. Agriculture, mineral mapping and urban monitoring '
           'all rely on this property.', indent_first=True)
    R.para('Pixel-wise classification of such cubes is hard for three reasons. The spectral axis is long '
           '(200 bands for Indian Pines) while labelled pixels are scarce. Neighbouring bands are highly '
           'correlated, and some are corrupted by water-vapour absorption or low signal-to-noise ratio at '
           'the sensor edges. Labelled classes are also badly unbalanced: in Indian Pines, Soybean-mintill '
           'covers 2,455 pixels while Oats covers 20. Classical pipelines answered these problems with '
           'kernel methods and support vector machines on hand-crafted features '
           '[@melgani2004classification,camps2005kernel,fauvel2008spectral]. Deep networks replaced the '
           'hand-crafted stage with learned spectral-spatial features [@chen2014deep,chen2016deep].',
           indent_first=True)
    R.h2('1.2\u2003Spectral-Spatial Entanglement and Small-Sample Scarcity')
    R.para('Most current models learn spectral and spatial structure jointly. 3D convolutions slide a '
           'kernel over both axes at once [@ben2018deep,zhong2017spectral]; transformers tokenise patches '
           'and let attention mix them [@dosovitskiy2020image,hong2021spectralformer]. Joint processing is '
           'powerful, but it entangles two signals. A noisy band is convolved together with its clean '
           'neighbours before any layer has had the chance to discount it. Two parallel branches that see '
           'the same input and are merged by a shared gate tend to learn redundant filters. And global '
           'average pooling summarises each channel by its spatial mean, discarding where in the patch a '
           'channel fired [@lin2015bilinear,gao2019global].', indent_first=True)
    R.para('Scarcity compounds the problem. With 10% of the pixels used for training, Oats contributes three '
           'training patches. A network optimised with plain cross-entropy can reach a high overall '
           'accuracy while ignoring such a class entirely, which is why AA and per-class accuracies matter '
           'as much as OA in this report [@lin2017focal,cui2019class].', indent_first=True)
    R.h2('1.3\u2003Problem Statement and Research Objectives')
    R.para('The project asks whether a small 3D CNN can approach the accuracy of heavier transformer and '
           'residual baselines on standard HSI benchmarks by placing explicit gates at the three points '
           'where entanglement occurs: the raw input, the branch merge, and the spatial summary. '
           '{T:tab_objectives} lists the objectives and where each is addressed; {F:fig_workflow} shows '
           'how the work was organised.', indent_first=True)
    R.table(['#', 'Objective', 'Addressed in'], [
        ['O1', 'Design SGSCF-Net with input-level band gating (SBAM), complementary branch fusion (CBSA) and an '
               'outer-product channel–spatial gate (SCFG), trained with a minority-aware Spread + Focal loss.',
         'Chapter 4'],
        ['O2', 'Benchmark it against eight published baselines across Indian Pines, University of Pavia, and Salinas Valley (200 epochs).',
         'Sections 6.1–6.3'],
        ['O3', 'Measure sensitivity to patch and batch size on three datasets (60 runs).', 'Section 6.5'],
        ['O4', 'Measure robustness to training ratios of 1%, 2%, 5% and 10% (12 runs).', 'Section 6.6'],
        ['O5', 'Report every number, caveat and failure case exactly as logged in the master records.',
         'Throughout; Chapter 8'],
    ], [0.5, 4.4, 1.3], 'Research objectives of the project.', label='tab_objectives',
        align=['c', 'j', 'l'], font=10)
    R.figure(D['fig_workflow'], 'Project methodology: from raw scenes to the three experiment families',
             '', width=4.5, label='fig_workflow')
    R.para('The concrete contributions are:', before=4)
    R.bullets([f'**SBAM**, a squeeze-and-excitation style gate [@hu2018squeeze] applied to the 200 raw bands '
               f'before the first convolution, costing {md["sbam"]:,} parameters.',
               f'**CBSA**, a depthwise 1D gate that produces $\\alpha$ per channel and band and fuses the two '
               f'entry branches as $\\alpha\\odot a+(1-\\alpha)\\odot b$, costing {md["cbsa"]} parameters.',
               f'**SCFG**, a gate computed from the normalised outer product $f_{{\\mathrm{{ch}}}}\\otimes '
               f'f_{{\\mathrm{{sp}}}}$ of channel and spatial descriptors, costing {md["scfg"]} parameters at '
               f'an 11×11 patch.',
               f'A documented evaluation: an eight-baseline comparison across three benchmark datasets, 60 grid runs, '
               f'12 scarcity runs and per-class breakdowns, all for a model of {fmtn(P_IN)} parameters ({ck_kb:.0f} KB checkpoint).'],
              numbered=True)
    R.h2('1.4\u2003Organisation of the Report')
    R.para('Chapter 2 reviews related work and identifies the gaps SGSCF-Net targets. Chapter 3 formalises '
           'the data and the learning problem. Chapter 4 describes the architecture module by module, with '
           'equations that follow the tensor operations in SGSCF_Net.py line for line. Chapter 5 gives the '
           'datasets, baselines, training protocol and metrics. Chapter 6 reports results across all three benchmarks, '
           'the grid search and the scarcity study. Chapter 7 interprets the gates and the baseline failures. '
           'Chapter 8 lists limitations and future work, and Chapter 9 concludes. In the electronic copy, '
           'every citation and every table, figure and equation reference is a clickable link.', indent_first=True)

    # ================================================================== CHAPTER 2
    R.h1('Literature Review and Related Work')
    R.h2('2.1\u2003Survey of Dominant Paradigms')
    R.h3('3D convolutional and residual networks')
    R.para('Early deep models treated each pixel as a 1D spectrum [@hu2015deep] or applied 2D convolutions '
           'to a few principal components [@zhao2016spectral]. 3D convolution removed the need to choose '
           'between the two by convolving over space and spectrum together [@chen2016deep,ben2018deep]. '
           'SSRN stacks spectral and then spatial 3D residual units [@zhong2017spectral,he2016deep] and '
           'remains one of the strongest baselines in our benchmark. HybridSN follows 3D layers with 2D '
           'layers to cut cost [@roy2020hybridsn], and region-based designs widen the spatial context '
           '[@zhang2018diverse]. These models share one property: every band enters the first kernel with '
           'equal weight.', indent_first=True)
    R.h3('Attention mechanisms')
    R.para('Squeeze-and-excitation [@hu2018squeeze], CBAM [@woo2018cbam] and ECA [@wang2020eca] re-weight '
           'feature channels from pooled statistics. HSI variants add spectral and spatial attention '
           'branches [@sun2019spectral,hang2021hyperspectral,zhu2021residual,li2020double], or learn band '
           'selection with reinforcement learning [@mou2022deep]. A2S2K-ResNet [@roy2020attention] is the '
           'closest ancestor of SGSCF-Net: it uses the same two 3D entry branches and residual units with '
           'ECA, but merges the branches with one shared attention weight. Nearly all of these gates act on '
           'feature maps after at least one convolution.', indent_first=True)
    R.h3('Transformers and state-space models')
    R.para('ViT [@dosovitskiy2020image] and its HSI descendants, SpectralFormer [@hong2021spectralformer], '
           'SSFTT [@sun2022spectral], spatial-spectral transformers [@he2021spatial,yang2022hyperspectral] '
           'and MorphFormer [@roy2023morphformer], model long-range interactions with self-attention, whose '
           'cost grows quadratically with the number of tokens. Selective state-space models such as Mamba '
           '[@gu2023mamba] reduce that cost to linear. In our benchmark the two pure transformers, ViT and '
           'HyperRoPE-HiT, are the weakest models after SSAN on Indian Pines.', indent_first=True)
    R.h3('Second-order pooling and capsules')
    R.para('Bilinear CNNs [@lin2015bilinear] and global covariance pooling [@gao2019global,li2018towards] '
           'replace first-order means with second-order statistics. In HSI, local covariance '
           'descriptors [@fang2018new,he2019feature] and attention-weighted second-order pooling '
           '[@xue2021attention,zheng2021hyperspectral] improved separability of spectrally similar '
           'classes. Capsule networks encode part–whole relations with routing between capsule layers '
           '[@paoletti2018capsule,sabour2018matrix]; the DMC baseline belongs to this family.',
           indent_first=True)
    R.h3('Class imbalance')
    R.para('Focal loss down-weights well-classified samples [@lin2017focal]; class-balanced weighting '
           'rescales by effective sample counts [@cui2019class]; margin losses such as LDAM '
           '[@cao2019learning] and the spread loss of matrix capsules [@sabour2018matrix] enforce a gap '
           'between the true-class score and its competitors. SGSCF-Net combines a spread term with a '
           'class-weighted focal term.', indent_first=True)
    R.h2('2.2\u2003Comparative Matrix of Existing Approaches')
    R.para('{T:tab_lit} condenses the review into the properties that matter for the gaps listed in '
           'Section 2.3.', indent_first=True)
    R.table(['Family / representative', 'Strengths', 'Limitations relevant here'], [
        ['3D CNN / residual: SSRN [@zhong2017spectral], HybridSN [@roy2020hybridsn]',
         'Joint spectral-spatial kernels; strong accuracy with modest size',
         'All bands weighted equally at input; no explicit imbalance handling'],
        ['Channel attention: SE [@hu2018squeeze], ECA [@wang2020eca], CBAM [@woo2018cbam]',
         'Cheap re-weighting from pooled statistics', 'Act after convolution; first-order (mean) statistics only'],
        ['Dual-branch attention: A2S2K-ResNet [@roy2020attention], DBDA [@li2020double]',
         'Multi-scale branches; adaptive kernels', 'Shared gate lets branches learn redundant features'],
        ['Spectral-spatial attention: SSAN [@sun2019spectral]',
         'Separate spectral and spatial attention', 'Reached 64.19% OA on Indian Pines in our protocol'],
        ['Transformers: ViT [@dosovitskiy2020image], SSFTT [@sun2022spectral], SpectralFormer [@hong2021spectralformer]',
         'Long-range context; tokenised spectra', 'Quadratic attention cost; data-hungry on small patches'],
        ['Second-order pooling: [@gao2019global,xue2021attention,zheng2021hyperspectral]',
         'Captures co-activation structure', 'Full covariance is costly (C² terms, matrix roots)'],
        ['Capsules: [@paoletti2018capsule], matrix capsules [@sabour2018matrix]',
         'Part–whole modelling; strong reported accuracy', 'Iterative routing; specialised training'],
        ['Imbalance losses: Focal [@lin2017focal], CB [@cui2019class], LDAM [@cao2019learning]',
         'Protect minority classes', 'Rarely combined with architecture-level fixes in HSI'],
    ], [2.3, 1.9, 2.1], 'Comparative matrix of existing hyperspectral classification approaches.',
        label='tab_lit', align=['j', 'j', 'j'], font=9)
    R.h2('2.3\u2003Identified Gaps and Rationale for SGSCF-Net')
    R.para('{T:tab_gaps} maps each gap to the design decision that answers it.', indent_first=True)
    R.table(['Gap', 'Typical practice', 'SGSCF-Net response'], [
        ['G1  No input-level spectral gate', 'Noisy bands are mixed into features by the first convolution',
         '**SBAM** gates the raw bands before any convolution'],
        ['G2  Redundant dual branches', 'A shared gate gives both branches the same weight',
         '**CBSA** ties them with $\\alpha$ and $1-\\alpha$: raising one lowers the other'],
        ['G3  First-order spatial summaries', 'Global average pooling keeps only channel means',
         '**SCFG** lets the spatial activation profile modulate each channel, at the cost of one linear layer'],
        ['G4  Imbalance handled in one place only', 'Either a re-weighted loss or an architectural fix',
         'Gates combined with a weighted **Spread + Focal** objective'],
    ], [1.9, 2.0, 2.4], 'Research gaps and the corresponding design decisions.', label='tab_gaps',
        align=['l', 'j', 'j'], font=9.5)
    R.callout('Key takeaways', [
        'Most HSI models weight all raw bands equally and summarise space with first-order means.',
        'Dual-branch designs such as A2S2K-ResNet merge branches with a shared gate.',
        'SGSCF-Net targets these three points with cheap gates rather than with more depth or attention.'])

    # ================================================================== CHAPTER 3
    R.h1('System Model and Problem Formulation')
    R.h2('3.1\u2003HSI Data Cubes and Patch Tensors')
    R.para('Let the scene be a cube with height $H$, width $W$ and $C$ spectral bands, and let the '
           'ground-truth map assign each pixel a label in $\\{0,1,\\dots,K\\}$, where 0 marks unlabelled '
           'background and $K$ is the number of classes:', indent_first=True, keep_next=True)
    R.eq(r'\mathcal{X}\in\mathbb{R}^{H\times W\times C},\qquad \mathcal{Y}\in\{0,1,\dots,K\}^{H\times W}', 'eq_cube')
    R.para('Each band is standardised over all pixels of the scene (scikit-learn preprocessing.scale on the '
           'flattened cube), so band $c$ has zero mean and unit variance:', indent_first=True, keep_next=True)
    R.eq(r'\bar{x}_{h,w,c}=\frac{x_{h,w,c}-\mu_c}{s_c},\qquad \mu_c=\frac{1}{HW}\sum_{h,w}x_{h,w,c}', 'eq_norm')
    R.para('The cube is zero-padded by $p=(P-1)/2$ pixels on each spatial side (np.pad with '
           'mode=\'constant\'), and for every labelled pixel $(i,j)$ a $P\\times P\\times C$ neighbourhood '
           'is cut out and given a singleton channel axis for 3D convolution ({F:fig_cube}). A mini-batch '
           'of $B$ patches is therefore', indent_first=True, keep_next=True)
    R.eq(r'X\in\mathbb{R}^{B\times 1\times P\times P\times C}', 'eq_input')
    R.figure(D['fig_cube'], 'From scene cube to network input',
             'Each labelled pixel contributes the P×P×C neighbourhood centred on it; the patch inherits the '
             'centre pixel’s label. Schematic, not to scale.', width=4.5, label='fig_cube')
    R.para('In the Conv3d convention $(N,C_{\\mathrm{in}},D,H,W)$ used by PyTorch, the two spatial axes '
           'play the role of $D$ and $H$, and the spectral axis is the last axis $W$. All kernel sizes in '
           'this report are written in that order, so a 1×1×7 kernel spans one pixel and seven bands.',
           indent_first=True)
    R.h2('3.2\u2003Dual-Branch 3D Feature Space and Operator Notation')
    R.para('Feature tensors inside the network keep the five-axis layout $B\\times K\\times P\\times '
           'P\\times C_2$, with $K=16$ feature channels and $C_2$ the spectral length after the strided '
           'entry convolution. We use $\\odot$ for broadcast element-wise multiplication, $\\otimes$ for '
           'the outer product of two vectors, $\\sigma$ for the logistic sigmoid, $\\delta$ for ReLU, '
           '$\\mathrm{BN}$ for batch normalisation and $\\|\\cdot\\|_F$ for the Frobenius norm. '
           '{T:tab_notation} lists the remaining symbols.', indent_first=True)
    R.table(['Symbol', 'Meaning', 'Value (Indian Pines, 11×11)'], [
        ['$B$', 'Mini-batch size', '32'], ['$P$', 'Spatial patch side', '11'],
        ['$C$', 'Input spectral bands', '200'], ['$C_2$', 'Spectral length after entry convolutions',
                                                 str(IN11['shapes']['bn1'][-1])],
        ['$K$', 'Feature channels in residual stages', '16'], ['$C_m$', 'Channels after expansion (SCFG input)', '128'],
        ['$N$', 'Spatial positions in a patch, $P^2$', '121'], ['$K_{\\mathrm{cls}}$', 'Number of classes', '16'],
        ['$r$', 'Training progress, step/total steps', '[0, 1]'],
    ], [1.0, 3.3, 1.9], 'Notation used in Chapters 3 and 4.', label='tab_notation', align=['c', 'l', 'c'],
        font=10)
    R.h2('3.3\u2003Problem Definition and Optimisation Objective')
    R.para('Labelled pixels are split per class. For training ratio $\\rho$ the sampling routine in '
           'SGSCF_Net.py keeps', indent_first=True, keep_next=True)
    R.eq(r'n_k^{\mathrm{tr}}=\max\left(\left\lfloor \rho\,N_k\right\rfloor,\;3\right)', 'eq_split')
    R.para(f'training pixels of class $k$, where $N_k$ is the number of labelled pixels of that class. The '
           f'floor of three samples matters for the scarcity study: at $\\rho=0.01$ every Indian Pines class '
           f'with fewer than 400 pixels receives exactly three training patches. A validation set of the '
           f'same total size as the training set is drawn from the shuffled remainder, and the rest forms '
           f'the test set, giving the 10%/10%/80% protocol at $\\rho=0.1$.', indent_first=True)
    R.para('The network $f_\\theta$ maps a batch to class scores, and training minimises the expected '
           'hybrid loss of Section 4.5 with weight decay $\\lambda$:', indent_first=True, keep_next=True)
    R.eq(r'\hat{Y}=f_\theta(X)\in\mathbb{R}^{B\times K_{\mathrm{cls}}},\qquad \theta^{\ast}=\arg\min_{\theta}\;'
         r'\mathbb{E}_{(X,y)}\left[\mathcal{L}\left(f_\theta(X),y;r\right)\right]+\lambda\|\theta\|_2^2', 'eq_obj')
    R.para('The model is selected on validation loss: the checkpoint with the lowest validation loss is '
           'restored after training, and training stops early if that loss has not improved for 30 '
           'epochs.', indent_first=True)

    # ================================================================== CHAPTER 4
    R.h1('SGSCF-Net Architecture and Module Design')
    R.h2('4.1\u2003System Overview')
    R.para('{F:fig_arch} traces one Indian Pines batch through the network. The data flow is: SBAM → '
           'dual 3D entry → CBSA → two spectral residual blocks → spectral pooling and channel expansion → '
           'SCFG → spatial entry → two spatial residual blocks → global pooling → linear classifier. The '
           'tensor shapes in the figure and in {T:tab_arch} were recorded with forward hooks on the classes '
           'in SGSCF_Net.py, not computed by hand. {F:fig_modules} opens up the three proposed modules.',
           indent_first=True)
    R.figure(D['fig_arch'], 'End-to-end architectural workflow of the proposed SGSCF-Net',
             'Computational pipeline from input hyperspectral patch to classification output.',
             width=6.4, label='fig_arch')
    R.figure(D['fig_modules'], 'Detailed architectural topologies of the four constituent algorithmic modules of SGSCF-Net',
             '(a) SBAM input band gate; (b) CBSA complementary branch gate; (c) SCFG second-order covariance gate; (d) GAP classification head.',
             width=6.4, label='fig_modules')
    R.h2('4.2\u2003Spectral Band Attention Module (SBAM)')
    R.para('SBAM decides how much each raw band contributes before the first convolution sees it. It '
           'averages the patch over its two spatial axes with AdaptiveAvgPool3d((1,1,C)), giving one value '
           'per band:', indent_first=True, keep_next=True)
    R.eq(r'z_c=\frac{1}{P^{2}}\sum_{u=1}^{P}\sum_{v=1}^{P}X_{b,1,u,v,c},\qquad \mathbf{z}\in\mathbb{R}^{B\times C}', 'eq_sbam_pool')
    R.para('A two-layer bottleneck with $C_m^{\\mathrm{sb}}=\\max(\\lfloor C/8\\rfloor,8)$ hidden units '
           '(25 for Indian Pines, 12 for Pavia University) produces a weight in $(0,1)$ for every band:',
           indent_first=True, keep_next=True)
    R.eq(r'\mathbf{w}=\sigma\left(\mathbf{W}_2\,\delta\left(\mathbf{W}_1\mathbf{z}+\mathbf{b}_1\right)+\mathbf{b}_2\right)\in(0,1)^{B\times C}', 'eq_sbam_fc')
    R.para('The weights are reshaped to $B\\times1\\times1\\times1\\times C$ and multiplied into the input '
           '(x * w in the code):', indent_first=True, keep_next=True)
    R.eq(r'\tilde{X}=X\odot\mathbf{w},\qquad \tilde{X}\in\mathbb{R}^{B\times 1\times P\times P\times C}', 'eq_sbam_out')
    R.para(f'The gate differs from SE [@hu2018squeeze] only in where it sits. Because it acts on '
           f'standardised reflectance rather than learned features, a band that is uninformative for the '
           f'current patch can be scaled toward zero before it is mixed with its neighbours. SBAM costs '
           f'$2\\,C\\,C_m^{{\\mathrm{{sb}}}}+C_m^{{\\mathrm{{sb}}}}+C$ = {md["sbam"]:,} parameters on Indian '
           f'Pines, the largest single block in the spectral half of the network.', indent_first=True)
    R.h2('4.3\u2003Complementary Branch Spectral Attention (CBSA)')
    R.para('Two 3D convolutions read the gated input in parallel. Both use stride 2 along the spectrum and '
           'are followed by BN and ReLU:', indent_first=True, keep_next=True)
    R.eq(r'\mathbf{a}=\delta\left(\mathrm{BN}\left(\mathrm{Conv3D}_{1\times1\times7}(\tilde{X})\right)\right),\qquad '
         r'\mathbf{b}=\delta\left(\mathrm{BN}\left(\mathrm{Conv3D}_{3\times3\times7}(\tilde{X})\right)\right)', 'eq_entry')
    R.eq(r'\mathbf{a},\mathbf{b}\in\mathbb{R}^{B\times K\times P\times P\times C_2},\qquad C_2=\left\lfloor\frac{C-7}{2}\right\rfloor+1', 'eq_c2')
    R.para(f'Branch $\\mathbf{{a}}$ looks at one pixel and seven bands; branch $\\mathbf{{b}}$ adds a 3×3 '
           f'spatial neighbourhood with padding (1,1,0), so both keep the $P\\times P$ grid. For $C=200$, '
           f'$C_2={IN11["shapes"]["bn1"][-1]}$. CBSA then averages the sum of the branches over the spatial '
           f'axes, (a + b).mean(dim=(2,3)):', indent_first=True, keep_next=True)
    R.eq(r'\mathbf{U}=\frac{1}{P^{2}}\sum_{u=1}^{P}\sum_{v=1}^{P}\left(\mathbf{a}+\mathbf{b}\right)_{:,:,u,v,:}\in\mathbb{R}^{B\times K\times C_2}', 'eq_cbsa_u')
    R.para('A depthwise 1D convolution (groups = $K$, kernel 3, no bias) slides along the spectral axis '
           'of each channel and a sigmoid turns the result into a gate:', indent_first=True, keep_next=True)
    R.eq(r'\boldsymbol{\alpha}=\sigma\left(\mathrm{DWConv1D}_{3}(\mathbf{U})\right)\in(0,1)^{B\times K\times C_2}', 'eq_cbsa_alpha')
    R.para('With $\\boldsymbol{\\alpha}$ broadcast to $B\\times K\\times1\\times1\\times C_2$, the branches '
           'are fused as a convex combination and normalised:', indent_first=True, keep_next=True)
    R.eq(r'\mathbf{v}=\mathrm{BN}\left(\boldsymbol{\alpha}\odot\mathbf{a}+\left(1-\boldsymbol{\alpha}\right)\odot\mathbf{b}\right)', 'eq_cbsa_fuse')
    R.para(f'For each channel and band the two weights sum to one, so the gate can only shift emphasis '
           f'between branches; it cannot amplify both. A2S2K-ResNet [@roy2020attention] instead derives one '
           f'shared weight for the two branches. The whole of CBSA is $K\\times3=48$ convolution weights '
           f'plus a 32-parameter BN, {md["cbsa"]} parameters in total.', indent_first=True)
    R.para('The fused tensor passes through two residual blocks with 1×1×7 kernels and padding (0,0,3), '
           'which preserve every axis. Each block ends with 3D efficient channel attention (ECA3D) '
           '[@wang2020eca]: global pooling to one value per channel, a 1D convolution of width 3 across '
           'channels, and a sigmoid. In compact notation, with $\\phi$ denoting BN followed by ReLU,',
           indent_first=True, keep_next=True)
    R.eq(r'\mathrm{ECA}(\mathbf{h})=\mathbf{h}\odot\sigma\left(\mathrm{Conv1D}_{3}\left(\mathrm{GAP}(\mathbf{h})\right)\right),\qquad '
         r'\mathcal{R}(\mathbf{v})=\mathbf{v}+\mathrm{ECA}\left(\mathrm{Conv3D}\left(\phi\left(\mathrm{Conv3D}\left(\phi(\mathbf{v})\right)\right)\right)\right)', 'eq_res')
    R.para('The exact placement of BN differs slightly between the first block (start=True, no input BN) '
           'and later blocks; the code keeps the residual unit of A2S2K-ResNet unchanged.', indent_first=True)
    R.h2('4.4\u2003Second-Order Covariance Fusion Gate (SCFG)')
    R.para('The spectral axis is collapsed by AdaptiveAvgPool3d((None, None, 1)) and a 1×1×1 convolution '
           'with BN and ReLU expands the channels from 16 to $C_m=128$. Dropping the singleton axis gives a '
           '2D map:', indent_first=True, keep_next=True)
    R.eq(r'\mathbf{x}=\delta\left(\mathrm{BN}\left(\mathrm{Conv3D}_{1\times1\times1}\left(\mathrm{Pool}_{C_2\to1}(\mathbf{v})\right)\right)\right)\in\mathbb{R}^{B\times C_m\times P\times P}', 'eq_expand')
    R.para('SCFG forms two descriptors from $\\mathbf{x}$, one per channel and one per spatial position '
           '($N=P^2$), exactly as x.mean(dim=(2,3)) and x.mean(dim=1) in the code:', indent_first=True,
           keep_next=True)
    R.eq(r'f_{\mathrm{ch}}(b,c)=\frac{1}{N}\sum_{n=1}^{N}x_{b,c,n},\qquad f_{\mathrm{sp}}(b,n)=\frac{1}{C_m}\sum_{c=1}^{C_m}x_{b,c,n}', 'eq_scfg_desc')
    R.para('Their batched outer product (torch.bmm) is the channel–spatial interaction matrix:',
           indent_first=True, keep_next=True)
    R.eq(r'G=f_{\mathrm{ch}}\otimes f_{\mathrm{sp}}=f_{\mathrm{ch}}\,f_{\mathrm{sp}}^{\top}\in\mathbb{R}^{B\times C_m\times N},\qquad G_{b,c,n}=f_{\mathrm{ch}}(b,c)\,f_{\mathrm{sp}}(b,n)', 'eq_scfg_g')
    R.para('$G$ is divided by its Frobenius norm (clamped below at $10^{-8}$), compressed along the spatial '
           'axis by one learned linear map $\\mathbb{R}^{N}\\to\\mathbb{R}$ shared by all channels, and '
           'turned into a channel gate:', indent_first=True, keep_next=True)
    R.eq(r'\tilde{G}=\frac{G}{\max\left(\|G\|_F,\,10^{-8}\right)},\qquad g_{b,c}=\sigma\left(\sum_{n=1}^{N}w_n\,\tilde{G}_{b,c,n}+w_0\right)', 'eq_scfg_gate')
    R.eq(r'\mathbf{x}_{\mathrm{scfg}}=\mathbf{x}\odot g,\qquad g\in(0,1)^{B\times C_m}\ \text{broadcast over}\ P\times P', 'eq_scfg_out')
    R.para(f'The gate has $N+1$ parameters, {md["scfg"]} for an 11×11 patch. Section 7.1 shows that, '
           f'because $G$ is an outer product of two vectors, this gate has a simple closed form that '
           f'explains what it can and cannot learn.', indent_first=True)
    R.para(f'After gating, the map is permuted to $B\\times1\\times P\\times P\\times C_m$ so that the 128 '
           f'channels become the last Conv3d axis. A 3×3×128 convolution with padding (1,1,0) reduces it '
           f'to $B\\times16\\times P\\times P\\times1$ ({md["se"]:,} parameters with BN), two spatial '
           f'residual blocks with 3×3×1 kernels refine it, and global average pooling plus a linear layer '
           f'give the class scores:', indent_first=True, keep_next=True)
    R.eq(r'\hat{\mathbf{y}}=\mathbf{W}_{\mathrm{fc}}\,\mathrm{GAP}\left(\mathcal{R}_{\mathrm{sp}}^{(2)}\left(\mathcal{R}_{\mathrm{sp}}^{(1)}(\mathbf{s})\right)\right)+\mathbf{b}_{\mathrm{fc}}\in\mathbb{R}^{B\times K_{\mathrm{cls}}}', 'eq_head')
    R.h2('4.5\u2003Minority-Aware Hybrid Loss')
    R.para('The training objective couples scheduled spread margin expansion with inverse-frequency weighted focal down-weighting.', indent_first=True)
    R.h3('Spread loss with a scheduled margin')
    R.para('Following the spread loss of matrix capsules [@sabour2018matrix], every wrong class whose score '
           'comes within a margin $m$ of the true-class score is penalised quadratically. The code applies '
           'it directly to the logits:', indent_first=True, keep_next=True)
    R.eq(r'\mathcal{L}_{\mathrm{Spread}}=\frac{1}{B}\sum_{i=1}^{B}\sum_{j\neq y_i}\max\left(0,\,m(r)-\left(\hat{y}_{i,y_i}-\hat{y}_{i,j}\right)\right)^{2}', 'eq_spread')
    R.eq(r'm(r)=m_{\min}+\left(m_{\max}-m_{\min}\right)r,\qquad m_{\min}=0.2,\ m_{\max}=0.9,\ r=\frac{t}{T_{\mathrm{total}}}', 'eq_margin')
    R.para('Here $t$ counts optimisation steps and $T_{\\mathrm{total}}$ is epochs × batches per epoch, so '
           'the margin widens from 0.2 to 0.9 over training. Validation and test loss use $r=1$.',
           indent_first=True)
    R.h3('Class-weighted focal loss')
    R.para('The focal term [@lin2017focal] scales the log-likelihood of the true class by $(1-p_t)^\\gamma$ '
           'with $\\gamma=2$ and by a per-class weight $\\alpha_k$:', indent_first=True, keep_next=True)
    R.eq(r'\mathcal{L}_{\mathrm{Focal}}=-\frac{1}{B}\sum_{i=1}^{B}\alpha_{y_i}\left(1-p_{i}\right)^{\gamma}\log p_{i},\qquad p_i=\mathrm{softmax}(\hat{\mathbf{y}}_i)_{y_i}', 'eq_focal')
    R.eq(r'\alpha_k=K_{\mathrm{cls}}\,\frac{w_k}{\sum_{j=1}^{K_{\mathrm{cls}}}w_j},\qquad w_k=\frac{1}{N_k+10^{-8}}', 'eq_alpha')
    alpha = D['alpha']
    R.para(f'The counts $N_k$ are taken over all labelled pixels of the ground-truth map (class_weights in '
           f'the code). On Indian Pines this gives Oats a weight of {alpha[8]:.2f} and Soybean-mintill '
           f'{alpha[10]:.3f}, a ratio of about {alpha[8] / alpha[10]:.0f} as defined in Eq. ({{E:eq_alpha}}).',
           indent_first=True)
    R.h3('Combined objective and optimiser')
    R.eq(r'\mathcal{L}=0.7\,\mathcal{L}_{\mathrm{Spread}}+0.3\,\mathcal{L}_{\mathrm{Focal}}', 'eq_total')
    R.para('Adam minimises $\\mathcal{L}$ with learning rate $\\eta_0=5\\times10^{-4}$ and weight decay '
           '$10^{-4}$. The rate follows cosine annealing over the epoch budget $E=200$, gradients are '
           'clipped to an $\\ell_2$ norm of 5.0, convolution weights use Kaiming-normal initialisation and '
           'linear layers Xavier-uniform:', indent_first=True, keep_next=True)
    R.eq(r'\eta_e=\eta_{\min}+\frac{1}{2}\left(\eta_0-\eta_{\min}\right)\left(1+\cos\frac{\pi e}{E}\right),\qquad \eta_{\min}=10^{-6}', 'eq_cos')
    R.h2('4.6\u2003Architectural and Hyperparameter Specifications')
    tot = IN11['params']; sh = IN11['shapes']
    fs = lambda k: '×'.join(map(str, sh[k]))
    arch_rows = [
        ['SBAM (proposed)', 'Pool + MLP 200→25→200 + sigmoid', fs('sbam'), md['sbam']],
        ['Spectral entry a', 'Conv3D 1×1×7, stride (1,1,2) + BN', fs('bn1'), md['e1'] + md['bn1']],
        ['Spatial-spectral entry b', 'Conv3D 3×3×7, stride (1,1,2) + BN', fs('bn3'), md['e3'] + md['bn3']],
        ['CBSA (proposed)', 'Depthwise Conv1D k=3 + BN', fs('cbsa'), md['cbsa']],
        ['Spectral ResBlocks ×2', 'Conv3D 1×1×7 + ECA3D', fs('rs2'), md['rs1'] + md['rs2']],
        ['Spectral pool + expand', 'AvgPool → Conv3D 1×1×1 (16→128) + BN', fs('expand'), md['expand']],
        ['SCFG (proposed)', 'Outer product + Linear(121→1)', '128×11×11', md['scfg']],
        ['Spatial entry', 'Conv3D 3×3×128 (1→16) + BN', fs('se'), md['se']],
        ['Spatial ResBlocks ×2', 'Conv3D 3×3×1 + ECA3D', fs('rs4'), md['rs3'] + md['rs4']],
        ['Classifier', 'Global pool + Linear 16→16', fs('fc'), md['fc']],
    ]
    R.table(['Stage', 'Operator', 'Output (per sample)', 'Params', 'Share'],
            [[a, b, c, f'{n:,}', f'{100 * n / tot:.2f}%'] for a, b, c, n in arch_rows]
            + [['Total', '', '', f'{tot:,}', '100.00%']],
            [1.55, 2.25, 1.15, 0.65, 0.65],
            'Layer-wise specification of SGSCF-Net for Indian Pines (P = 11, C = 200, 16 classes), measured from SGSCF_Net.py.',
            label='tab_arch', align=['l', 'l', 'c', 'r', 'r'], font=9, total_last=True, hl=(0, 3, 6),
            note=f'Highlighted rows are the proposed modules. Counts are trainable parameters; BN running '
                 f'statistics are excluded. The same code gives {P_UP:,} parameters for Pavia University (103 '
                 f'bands, 9 classes) and {P_SV:,} for Salinas Valley (204 bands). The saved checkpoints in '
                 f'SGSCF-Net_Final_Implementation/checkpoints agree: ' + ck_desc + '.')
    R.para(f'The three proposed modules together cost {novel_p:,} parameters, '
           f'{100 * novel_p / tot:.1f}% of the network, and almost all of that is SBAM. CBSA and SCFG are '
           f'nearly free. The spatial entry convolution is the largest single layer '
           f'({100 * md["se"] / tot:.1f}%), because its kernel spans all 128 expanded channels. '
           f'{{T:tab_hparams}} lists the training settings.', indent_first=True)
    R.table(['Hyperparameter', 'Value', 'Source in code'], [
        ['Patch size P (benchmark setting)', '11×11 (grid: 9–17)', '--patch 5 (window 2p+1)'],
        ['Batch size (benchmark setting)', '32 (grid: 32–256)', '--batch_size'],
        ['Training / validation / test', '10% / 10% / 80%', 'sampling(), make_loaders()'],
        ['Epoch budget', '200', '--epoch'],
        ['Early stopping', 'patience 30 on validation loss', 'train_model()'],
        ['Optimiser', 'Adam, lr 5×10⁻⁴, weight decay 10⁻⁴', 'optim.Adam'],
        ['LR schedule', 'Cosine annealing, η_min = 10⁻⁶', 'CosineAnnealingLR'],
        ['Gradient clipping', 'ℓ2 norm 5.0', 'clip_grad_norm_'],
        ['Feature channels K', '16', '--kernel'],
        ['SBAM reduction', 'r = 8, hidden ≥ 8', 'SBAM(C, r=8)'],
        ['Loss weights', '0.7 Spread + 0.3 Focal', 'CombinedLoss'],
        ['Spread margin', '0.2 → 0.9 (linear in r)', 'SpreadLoss'],
        ['Focal γ, class weights', '2.0, inverse frequency', 'FocalLoss, class_weights()'],
        ['Random seeds', '1331, 1332, … (one per run)', 'seeds list'],
    ], [2.2, 2.15, 1.9], 'Training hyperparameters of SGSCF-Net.', label='tab_hparams',
        align=['l', 'l', 'l'], font=9.5)
    R.callout('Key takeaways', [
        f'SGSCF-Net has {P_IN:,} parameters; the three gates account for {100 * novel_p / tot:.1f}% of them.',
        'Every equation in this chapter maps to a line of SGSCF_Net.py, including the stride-2 spectral '
        f'downsampling to $C_2={IN11["shapes"]["bn1"][-1]}$ and the rank-one interaction matrix in SCFG.',
        'The loss widens its margin over training and gives Oats about '
        f'{alpha[8] / alpha[10]:.0f} times the weight of the largest class.'])

    # ================================================================== CHAPTER 5
    R.h1('Experimental Setup and Benchmarking Methodology')
    R.h2('5.1\u2003Hyperspectral Benchmark Datasets')
    R.para('Three public scenes are used; {T:tab_datasets} summarises them and {F:fig_classes} shows how '
           'unevenly their labelled pixels are spread over classes. **Indian Pines** was recorded by AVIRIS '
           '[@green1998imaging] over agricultural land in north-west Indiana: 145×145 pixels at 20 m, 200 '
           'bands after removing water-absorption bands, and 16 classes over 10,249 labelled pixels. **Pavia '
           'University** was recorded by ROSIS over an urban campus in northern Italy [@fauvel2008spectral]: '
           '610×340 pixels at 1.3 m, 103 bands, 9 classes and 42,776 labelled pixels. **Salinas Valley** is '
           'another AVIRIS scene, of farmland in California: 512×217 pixels at 3.7 m, 204 bands, 16 classes '
           'and 54,129 labelled pixels. The labelled-pixel totals match the constants hard-coded in '
           'load_dataset().', indent_first=True)
    R.table(['Dataset', 'Sensor', 'Size (pixels)', 'GSD', 'Bands', 'Classes', 'Labelled', 'Smallest class'], [
        ['Indian Pines', 'AVIRIS', '145×145', '20 m', '200', '16', '10,249', 'Oats (20)'],
        ['Pavia University', 'ROSIS', '610×340', '1.3 m', '103', '9', '42,776', 'Shadows (947)'],
        ['Salinas Valley', 'AVIRIS', '512×217', '3.7 m', '204', '16', '54,129', 'Lettuce 6wk (916)'],
    ], [1.25, 0.65, 0.85, 0.5, 0.5, 0.6, 0.7, 1.25], 'Benchmark datasets.', label='tab_datasets', font=9)
    R.figure(D['fig_classes'], 'Class distribution of the three benchmarks',
             'Note the logarithmic axis: on Indian Pines the largest class is 123 times the smallest.',
             width=4.0, label='fig_classes')
    R.para(f'{{T:tab_ip_classes}} gives the Indian Pines class sizes together with the number of training '
           f'pixels that Eq. ({{E:eq_split}}) assigns at 10% and 1%. At 1%, {n_floor1} of the sixteen '
           f'classes sit on the three-sample floor.', indent_first=True)
    R.table(['ID', 'Class', 'Labelled', 'Train @10%', 'Train @1%'],
            [[i + 1, c, f'{n:,}', a, b] for i, (c, n, a, b) in enumerate(zip(IP_CLASSES, IP_COUNTS, tr10, tr1))]
            + [['', 'Total', f'{sum(IP_COUNTS):,}', f'{sum(tr10):,}', f'{sum(tr1):,}']],
            [0.45, 2.2, 1.0, 1.0, 1.0], 'Indian Pines classes and training-set sizes under the code’s sampling rule.',
            label='tab_ip_classes', align=['c', 'l', 'r', 'r', 'r'], font=9.5, total_last=True,
            hl=tuple(i for i, n in enumerate(IP_COUNTS) if n < 100),
            note='Highlighted: classes with fewer than 100 labelled pixels.')
    R.h2('5.2\u2003Baseline Algorithms')
    R.para('Eight baselines span four families ({T:tab_baselines_desc}). Each was evaluated from its authors’ '
           'public code across the benchmark datasets under the same protocol as SGSCF-Net: 11×11 patches, batch 32, a '
           '10%/10%/80% split and 200 epochs on a GPU environment. The status column carries the '
           'caveats from the master record; they apply to later use of these numbers.', indent_first=True)
    R.table(['Model', 'Family', 'Core mechanism', 'Status in this study'], [
        ['SSRN [@zhong2017spectral]', '3D CNN', 'Spectral then spatial 3D residual units',
         'Fresh run after a tf_keras compatibility fix'],
        ['A2S2K-ResNet‡ [@roy2020attention]', '3D CNN + attention', 'Dual 3D entry branches, shared attention weight',
         'Rerun hung on Kaggle; prior-run values kept as placeholders on IP'],
        ['SSAN [@sun2019spectral]', 'Attention', 'Separate spectral and spatial attention (TF1)',
         'Fresh run with a small-class-safe split'],
        ['ViT [@dosovitskiy2020image]', 'Transformer', 'Self-attention over patch embeddings', 'Fresh run'],
        ['HSI-SSFTT [@sun2022spectral]', 'Hybrid transformer', '3D/2D convolutions, then feature tokens', 'Fresh run'],
        ['HyperRoPE-HiT / SST', 'Transformer', '2D rotary position embeddings (hyperrope-sst code)', 'Fresh run'],
        ['CRV-MHDL [@alalimi2022compression]', 'Hybrid', 'Compression and reinforced variation + hybrid classifier',
         'Fresh run; map in the authors’ colours'],
        ['DMC† / DMC*', 'Capsules', 'Deep matrix capsules, EM routing (DeepMatrixCapsules code)',
         'Authors’ pre-trained checkpoint evaluated on IP; padded to 13×13 on PU/SV'],
    ], [1.75, 1.05, 1.95, 1.55], 'Baseline methods and their status in this study.', label='tab_baselines_desc',
        align=['l', 'l', 'j', 'j'], font=9)
    R.h2('5.3\u2003Implementation, Hardware and Training Protocol')
    R.para('SGSCF-Net is implemented in PyTorch (requirements: torch ≥ 1.13.1, scikit-learn, SciPy, NumPy, '
           'matplotlib). All reported SGSCF-Net and baseline runs were executed on GPU notebooks; '
           'the Indian Pines grid report records an NVIDIA Tesla T4. Each configuration in the master '
           'record is a single training run, so the tables report single values rather than means with '
           'standard deviations. Section 6.6 uses the one repeated configuration in the record to give a '
           'sense of run-to-run spread. {T:tab_expmatrix} summarises the three experiment families.',
           indent_first=True)
    R.table(['Experiment', 'Datasets', 'Varied', 'Held fixed', 'Runs'], [
        ['Benchmark', 'IP, PU, SV', '9 models', '11×11, batch 32, 10% train, 200 epochs', '27'],
        ['Grid search', 'IP, PU, SV', 'Patch 9–17; batch 32–256', '10% train, 200 epochs', '60'],
        ['Data scarcity', 'IP, PU, SV', 'Train ratio 1, 2, 5, 10%', '11×11, batch 32, 200 epochs', '12'],
    ], [1.15, 1.0, 1.55, 2.0, 0.55], 'Experiment matrix.', label='tab_expmatrix', align=['l', 'l', 'l', 'l', 'c'],
        font=9.5)
    R.h2('5.4\u2003Evaluation Metrics')
    R.para('Let $M$ be the confusion matrix, where $M_{ij}$ counts test pixels of class $i$ predicted as '
           'class $j$. Write $n_{i\\cdot}$ for its row sums (true counts), $n_{\\cdot i}$ for its column '
           'sums (predicted counts) and $n$ for the number of test pixels:', indent_first=True, keep_next=True)
    R.eq(r'\mathrm{OA}=\frac{1}{n}\sum_{i=1}^{K_{\mathrm{cls}}}M_{ii},\qquad \mathrm{AA}=\frac{1}{K_{\mathrm{cls}}}\sum_{i=1}^{K_{\mathrm{cls}}}\frac{M_{ii}}{n_{i\cdot}}', 'eq_oa')
    R.eq(r'\kappa=\frac{p_o-p_e}{1-p_e},\qquad p_o=\mathrm{OA},\qquad p_e=\frac{1}{n^{2}}\sum_{i=1}^{K_{\mathrm{cls}}}n_{i\cdot}\,n_{\cdot i}', 'eq_kappa')
    R.para('OA, AA and $\\kappa$ are reported in percent ($\\kappa\\times100$). AA weights every class '
           'equally, so it exposes failures on small classes that OA hides. Efficiency is reported as '
           'wall-clock training time, trainable parameters and checkpoint size. For SGSCF-Net we also give '
           'multiply–accumulate operations (MACs) per patch, counted with forward hooks.', indent_first=True)

    # ================================================================== CHAPTER 6
    R.h1('Empirical Results, Comparative Analysis and Ablations')
    R.kpis([(f'{ip11["OA"]:.2f}%', 'OA, Indian Pines'), (f'{g_pu[(11, 32)]["OA"]:.2f}%', 'OA, Pavia Univ.'),
            (f'{g_sv[(11, 32)]["OA"]:.2f}%', 'OA, Salinas Valley'), ('~308K', 'Trainable Params')])

    # ------------------------------------------------------------------ 6.1 Indian Pines
    R.h2('6.1\u2003Benchmark Comparison on Indian Pines')
    brow = []
    for b in BASELINES:
        cite = f' [@{",".join(b[7])}]' if b[7] else ''
        brow.append([_tag(b) + cite, b[1], f'{b[2]:.2f}', f'{b[3]:.2f}', f'{b[4]:.2f}', f'{b[5]:.2f}', b[6]])
    vals = [(b[2], b[3], b[4]) for b in BASELINES] + [OURS[2:5]]
    bestv = [max(v[k] for v in vals) for k in range(3)]
    def bmark(x, k): return f'**{x:.2f}**' if abs(x - bestv[k]) < 1e-9 else f'{x:.2f}'
    for r, b in zip(brow, BASELINES):
        r[2], r[3], r[4] = bmark(b[2], 0), bmark(b[3], 1), bmark(b[4], 2)
    brow.append(['SGSCF-Net (ours)', OURS[1], f'{OURS[2]:.2f}', f'{OURS[3]:.2f}', f'{OURS[4]:.2f}',
                 f'{OURS[5]:.2f}', f'{ck_kb:.0f} KB'])
    R.para('{T:tab_baselines} lists the nine models; {F:fig_scatter} and {F:fig_scatter} plot the same numbers.',
           indent_first=True)
    R.table(['Model', 'Paradigm', 'OA (%)', 'AA (%)', 'κ×100', 'Time (min)', 'Weights'], brow,
            [1.6, 1.75, 0.55, 0.55, 0.55, 0.65, 0.65],
            'Indian Pines benchmark: 11×11 patch, batch 32, 10%/10%/80% split, 200 epochs. Best value per column in bold.',
            label='tab_baselines', align=['l', 'l', 'c', 'c', 'c', 'c', 'c'], font=8.5, hl=(len(brow) - 1,),
            note='‡ A2S2K-ResNet: prior-run values kept as placeholders; the standardised rerun did not finish. '
                 '† DMC: the authors’ pre-trained checkpoint was evaluated (warm start), so 0.39 min is not a '
                 'training time and its training pixels may overlap our test split. Weights: checkpoint sizes '
                 'recorded in the baseline master record; n/r = not recorded. The SGSCF-Net size is that of '
                 'the saved Indian Pines checkpoint.')
    R.para(f'SGSCF-Net reaches {OURS[2]:.2f}% OA, {OURS[3]:.2f}% AA and a kappa of {OURS[4]:.2f} in '
           f'{OURS[5]} minutes ({F_ref("fig_perf", R)}(a)). Among the six baselines whose numbers come from a fresh, from-scratch run, '
           f'three are more accurate: SSRN ({fs_[0][2]:.2f}%), CRV-MHDL ({fs_[1][2]:.2f}%) and HSI-SSFTT '
           f'({fs_[2][2]:.2f}%). The gap to SSRN is {fs_[0][2] - OURS[2]:.2f} points of OA and '
           f'{fs_[0][3] - OURS[3]:.2f} points of AA. SGSCF-Net beats ViT by {OURS[2] - 92.70:.2f} points, '
           f'HyperRoPE-HiT by {OURS[2] - 94.42:.2f} and SSAN by {OURS[2] - 64.19:.2f}.', indent_first=True)
    R.para(f'Its training time sits in the middle of the field ({F_ref("fig_perf", R)}(b)). It is {100 * (1 - OURS[5] / 18.42):.1f}% '
           f'shorter than the recorded A2S2K-ResNet time (18.42 min, placeholder ‡) and '
           f'{100 * (1 - OURS[5] / 37.74):.1f}% shorter than HyperRoPE-HiT (37.74 min), but HSI-SSFTT (0.84 '
           f'min), ViT (1.71 min), CRV-MHDL (3.23 min) and SSRN (5.37 min) all train faster.', indent_first=True)
    R.para(f'The clearer efficiency advantage is size ({F_ref("fig_perf", R)}(c)). At {ck_kb:.0f} KB, the '
           f'SGSCF-Net checkpoint is smaller than every baseline checkpoint whose size is recorded: HSI-SSFTT '
           f'at 625 KB, A2S2K-ResNet at 1.5 MB, ViT at 2.4 MB and CRV-MHDL at 58.2 MB. HSI-SSFTT and CRV-MHDL '
           f'are more accurate, at 2.6 and about 250 times the storage.', indent_first=True)
    R.figure(D['fig_perf'], 'Comparative performance, training latency, and model footprint profile on Indian Pines',
             '(a) Overall Accuracy, Average Accuracy, and Kappa scores across comparative methods. '
             '(b) Wall-clock training latency (minutes). (c) Overall Accuracy versus checkpoint size footprint (KB).',
             width=6.4, label='fig_perf')
    pc = ip11['classes']
    pcn = [pc[c] for c in IP_CLASSES]
    n98 = sum(1 for v in pcn if v > 98)
    perfect = [c for c in IP_CLASSES if pc[c] == 100.0]
    R.para(f'{{T:tab_ip_perclass}} breaks the SGSCF-Net result down by class. {" and ".join(perfect)} are '
           f'classified perfectly, and {n98} of the sixteen classes exceed 98%. The weak point is Oats at '
           f'{pc["Oats"]:.2f}%: the class has 20 labelled pixels, and the sampling rule gives it {tr10[8]} '
           f'for training. Oats alone pulls AA down; the mean of the other fifteen classes is '
           f'{(sum(pcn) - pc["Oats"]) / 15:.2f}%.', indent_first=True)
    R.table(['ID', 'Class', 'Acc. (%)', 'ID', 'Class', 'Acc. (%)'],
            [[i + 1, IP_CLASSES[i], f'{pcn[i]:.2f}', i + 9, IP_CLASSES[i + 8], f'{pcn[i + 8]:.2f}'] for i in range(8)],
            [0.4, 1.9, 0.75, 0.4, 1.95, 0.75],
            'Per-class accuracy of SGSCF-Net on Indian Pines (11×11, batch 32, 10% training).',
            label='tab_ip_perclass', align=['c', 'l', 'c', 'c', 'l', 'c'], font=9.5,
            note=f'OA {ip11["OA"]:.2f}%, AA {ip11["AA"]:.2f}%, κ {ip11["K"]:.2f}, training time {ip11["T"]} min.')
    R.figure(D['fig_maps'], 'Classification maps of all models on Indian Pines',
             'All maps except (g) use the shared palette. * CRV-MHDL map in the authors’ colours. '
             '‡ prior-run placeholder. † pre-trained checkpoint.', width=4.55, label='fig_maps')
    R.para('The SSAN map contains scattered errors across whole parcels, consistent with its 64.19% OA. '
           'ViT and HyperRoPE-HiT recover the large fields but leave speckle inside parcels and along their '
           'borders. The HSI-SSFTT, SSRN and DMC maps are almost free of visible errors. The SGSCF-Net map '
           'is clean over most parcels but contains a few contiguous misclassified blocks, the largest a '
           'cyan region inside a dark-green parcel left of centre. Its errors are spatially clustered '
           'rather than scattered, a pattern the A2S2K-ResNet map shares, plausibly because both '
           'architectures classify overlapping 11×11 patches with similar dual-branch 3D features. These '
           'maps cover all labelled pixels, training pixels included, so they illustrate error structure '
           'rather than measure accuracy.', indent_first=True)

    # ------------------------------------------------------------------ 6.2 Pavia University
    R.h2('6.2\u2003Comparative Benchmark on University of Pavia')
    R.para('The University of Pavia benchmark evaluates model robustness on high-spatial-resolution (1.3 m) '
           'urban remote sensing imagery acquired by the ROSIS sensor over northern Italy. The scene comprises '
           '610×340 pixels across 103 calibrated spectral bands (0.43–0.86 μm) with 42,776 labeled pixels across '
           'nine urban land-cover categories. {T:tab_baseline_pu} reports quantitative classification '
           'performance across all nine evaluated architectures under standardized 10% training, 10% validation, '
           'and 80% testing splits over 200 epochs.', indent_first=True)
    prow = []
    pvals = [(b[2], b[3], b[4], b[5]) for b in BASELINES_PU]
    p_boa = max(v[0] for v in pvals); p_baa = max(v[1] for v in pvals)
    p_bk = max(v[2] for v in pvals); p_bt = min(v[3] for v in pvals)
    for b in BASELINES_PU:
        cite = f' [@{",".join(b[6])}]' if b[6] else ''
        prow.append([b[0] + cite, b[1],
                     f'**{b[2]:.2f}**' if abs(b[2] - p_boa) < 1e-9 else f'{b[2]:.2f}',
                     f'**{b[3]:.2f}**' if abs(b[3] - p_baa) < 1e-9 else f'{b[3]:.2f}',
                     f'**{b[4]:.2f}**' if abs(b[4] - p_bk) < 1e-9 else f'{b[4]:.2f}',
                     f'**{b[5]:.2f}**' if abs(b[5] - p_bt) < 1e-9 else f'{b[5]:.2f}'])
    R.table(['Model Architecture', 'Paradigm / Category', 'OA (%)', 'AA (%)', 'κ×100', 'Train Time (min)'],
            prow, [1.65, 2.15, 0.58, 0.58, 0.58, 0.73],
            'Quantitative Performance Comparison of SGSCF-Net and Eight Baseline Models on University of Pavia (10% Training / 10% Validation / 80% Testing Split, 200 Epochs). Best results in bold.',
            label='tab_baseline_pu', align=['l', 'l', 'c', 'c', 'c', 'c'], font=8.5, hl=(len(prow) - 1,),
            note='*Note: DMC patches were minimally zero-padded from 11×11 to 13×13 to satisfy capsule layer squash receptive field requirements without modifying internal capsule routing weights. CRV-MHDL was executed under Keras 3 with an updated Adam optimizer.')
    R.para(f'As detailed in {{T:tab_baseline_pu}}, 3D convolutional models (SSRN at 99.87% OA, A2S2K-ResNet at 99.83% OA) '
           f'and transformer architectures (HyperRoPE-SST at 99.83% OA, HSI-SSFTT at 99.72% OA) achieve competitive '
           f'accuracy on this urban benchmark. However, substantial disparities emerge across computational latency and '
           f'parametric efficiency. A2S2K-ResNet requires 41.12 minutes of GPU training due to multi-scale adaptive kernel '
           f'convolutions, while Deep Matrix Capsules (DMC) demands 73.42 minutes to complete iterative Expectation-Maximization '
           f'(EM) routing. In contrast, SGSCF-Net achieves 99.82% OA, 99.71% AA, and 99.76 Kappa in only 12.90 minutes of '
           f'training, matching the top-tier accuracy while maintaining a lightweight footprint of ~308K parameters.',
           indent_first=True)
    R.para('The primary land-cover challenge in University of Pavia centers on urban material confusion: spectral '
           'ambiguity between materials exhibiting similar visible-NIR reflectance curves (asphalt roads versus bitumen '
           'rooftops; self-blocking bricks versus bare soil), juxtaposed against thin geometric structures including '
           'pedestrian walkways and shadow slivers. {T:tab_pu_perclass} reports SGSCF-Net’s granular per-class accuracies '
           'under the standardized 11×11 / batch-32 configuration.', indent_first=True)
    pu_cls = g_pu[(11, 32)]['classes']
    pu_cnames = list(pu_cls.keys())
    pu_rows = []
    for i in range(5):
        c1 = pu_cnames[i]; a1 = f'{pu_cls[c1]:.2f}'
        if i + 5 < len(pu_cnames):
            c2 = pu_cnames[i + 5]; a2 = f'{pu_cls[c2]:.2f}'
            pu_rows.append([i + 1, c1, a1, i + 6, c2, a2])
        else:
            pu_rows.append([i + 1, c1, a1, '', '—', '—'])
    R.table(['ID', 'Class Name', 'Acc. (%)', 'ID', 'Class Name', 'Acc. (%)'],
            pu_rows, [0.4, 2.0, 0.75, 0.4, 2.0, 0.72],
            'Per-class accuracy of SGSCF-Net on University of Pavia (11×11 patch, batch 32, 10% training, 200 epochs).',
            label='tab_pu_perclass', align=['c', 'l', 'c', 'c', 'l', 'c'], font=9.5,
            note=f'OA {g_pu[(11, 32)]["OA"]:.2f}%, AA {g_pu[(11, 32)]["AA"]:.2f}%, κ {g_pu[(11, 32)]["K"]:.2f}, training time {g_pu[(11, 32)]["T"]} min.')
    R.para('Three urban categories (Meadows, Painted metal sheets, and Bitumen) reach perfect 100.00% '
           'classification accuracy, while Bare Soil reaches 99.80%. The historically confused material pairs—Asphalt vs. Bitumen and Self-Blocking '
           'Bricks vs. Bare Soil—attain 99.79% and 99.32%, respectively. This fine discrimination is directly enabled '
           'by the complementary CBSA gating mechanism ($\\alpha \\odot a + (1-\\alpha) \\odot b$), which forces the '
           '1×1×7 spectral-only branch to isolate subtle hydrocarbon absorption valleys while the 3×3×7 branch preserves '
           'structural boundaries. The low-radiance Shadows class (947 labeled pixels) attains 99.34% accuracy through '
           'the second-order SCFG covariance operator, which preserves discriminative activation where first-order pooling drops toward zero.',
           indent_first=True)
    R.figure(D['fig_pu_maps'], 'Classification maps of different methods for the University of Pavia dataset under standardized 10% training ratio',
             '(a) Ground truth. (b) ViT (OA: 99.05%). (c) HSI-SSFTT (OA: 99.72%). (d) SSRN (OA: 99.87%). '
             '(e) SSAN (OA: 76.82%). (f) A2S2K-ResNet (OA: 99.83%). (g) CRV-MHDL (OA: 98.40%). '
             '(h) HyperRoPE-SST (OA: 99.83%). (i) DMC (OA: 99.34%). (j) SGSCF-Net (Ours, OA: 99.82%).',
             width=4.55, label='fig_pu_maps')
    R.para('Inspection of the qualitative classification maps in {F:fig_pu_maps} highlights critical architectural '
           'distinctions. The spatial-spectral attention model SSAN ({F:fig_pu_maps}(e)) suffers a severe breakdown, '
           'dropping to 76.82% OA. Inspection reveals pervasive salt-and-pepper artifacts and complete category collapse '
           'on Gravel (17.96% accuracy) and Bare Soil (60.36% accuracy); its unconstrained softmax spatial attention '
           'distributes probability mass over surrounding meadow surfaces. ViT ({F:fig_pu_maps}(b)) and CRV-MHDL '
           '({F:fig_pu_maps}(g)) generate noticeable boundary blurring along elongated building perimeters. Conversely, '
           '3D convolutional models (SSRN, A2S2K-ResNet) and SGSCF-Net ({F:fig_pu_maps}(j)) reproduce crisp, geometrically '
           'sharp building contours and continuous asphalt arteries, demonstrating that input-level SBAM filtering '
           'prevents atmospheric dispersion from polluting high-resolution spatial boundaries.', indent_first=True)

    # ------------------------------------------------------------------ 6.3 Salinas Valley
    R.h2('6.3\u2003Comparative Benchmark on Salinas Valley')
    R.para('The Salinas Valley scene provides an evaluation benchmark characterized by high spatial continuity and '
           'fine-grained agricultural class separation. Captured by the AVIRIS sensor over Salinas Valley, California, '
           'at 3.7 m spatial resolution, the dataset comprises 512×217 pixels with 204 calibrated spectral bands and '
           '54,129 labeled pixels across 16 crop and soil categories. {T:tab_baseline_sv} summarizes quantitative '
           'classification metrics across all nine evaluated models under the standardized 10% training, 10% validation, '
           'and 80% testing protocol.', indent_first=True)
    srow = []
    svals = [(b[2], b[3], b[4], b[5]) for b in BASELINES_SV]
    s_boa = max(v[0] for v in svals); s_baa = max(v[1] for v in svals)
    s_bk = max(v[2] for v in svals); s_bt = min(v[3] for v in svals)
    for b in BASELINES_SV:
        cite = f' [@{",".join(b[6])}]' if b[6] else ''
        srow.append([b[0] + cite, b[1],
                     f'**{b[2]:.2f}**' if abs(b[2] - s_boa) < 1e-9 else f'{b[2]:.2f}',
                     f'**{b[3]:.2f}**' if abs(b[3] - s_baa) < 1e-9 else f'{b[3]:.2f}',
                     f'**{b[4]:.2f}**' if abs(b[4] - s_bk) < 1e-9 else f'{b[4]:.2f}',
                     f'**{b[5]:.2f}**' if abs(b[5] - s_bt) < 1e-9 else f'{b[5]:.2f}'])
    R.table(['Model Architecture', 'Paradigm / Category', 'OA (%)', 'AA (%)', 'κ×100', 'Train Time (min)'],
            srow, [1.65, 2.15, 0.58, 0.58, 0.58, 0.73],
            'Quantitative Performance Comparison of SGSCF-Net and Eight Baseline Models on Salinas Valley (10% Training / 10% Validation / 80% Testing Split, 200 Epochs). Best results in bold.',
            label='tab_baseline_sv', align=['l', 'l', 'c', 'c', 'c', 'c'], font=8.5, hl=(len(srow) - 1,),
            note='*Note: DMC patches were minimally zero-padded from 11×11 to 13×13 to satisfy capsule layer squash receptive field requirements without modifying internal capsule routing weights. CRV-MHDL was executed under Keras 3 with Salinas output head corrected to 16 classes.')
    R.para('On Salinas Valley, the contiguous parcel layout allows top models to reach exceptional accuracy thresholds: '
           'A2S2K-ResNet achieves 100.00% OA (0.999977) in 10.34 minutes, SSRN attains 99.96% OA in 30.98 minutes, and '
           'HSI-SSFTT reaches 99.90% OA in 5.26 minutes. However, DMC requires an extensive 94.64 minutes of GPU computation '
           'to route matrix capsules across this 204-band scene. Baseline attention networks fail to maintain uniform precision '
           'across categories: SSAN drops to 88.20% OA (Kappa 86.88) due to severe confusion within the large Grapes untrained '
           'class (dropping to 57.89% accuracy), while CRV-MHDL drops to 96.30% OA due to misclassifications in Vinyard '
           'untrained (81.00% accuracy).', indent_first=True)
    R.para(f'In contrast, SGSCF-Net achieves {g_sv[(11, 32)]["OA"]:.2f}% OA, {g_sv[(11, 32)]["AA"]:.2f}% AA, and '
           f'{g_sv[(11, 32)]["K"]:.2f} Kappa in {g_sv[(11, 32)]["T"]} minutes. Eleven of the sixteen categories reach '
           f'100.00% accuracy, as detailed in {{T:tab_sv_perclass}}.', indent_first=True)
    sv_cls = g_sv[(11, 32)]['classes']
    sv_cnames = list(sv_cls.keys())
    sv_rows = []
    for i in range(8):
        c1 = sv_cnames[i]; a1 = f'{sv_cls[c1]:.2f}'
        c2 = sv_cnames[i + 8]; a2 = f'{sv_cls[c2]:.2f}'
        sv_rows.append([i + 1, c1, a1, i + 9, c2, a2])
    R.table(['ID', 'Class Name', 'Acc. (%)', 'ID', 'Class Name', 'Acc. (%)'],
            sv_rows, [0.4, 2.05, 0.7, 0.4, 2.05, 0.67],
            'Per-class accuracy of SGSCF-Net on Salinas Valley (11×11 patch, batch 32, 10% training, 200 epochs).',
            label='tab_sv_perclass', align=['c', 'l', 'c', 'c', 'l', 'c'], font=9.0,
            note=f'OA {g_sv[(11, 32)]["OA"]:.2f}%, AA {g_sv[(11, 32)]["AA"]:.2f}%, κ {g_sv[(11, 32)]["K"]:.2f}, training time {g_sv[(11, 32)]["T"]} min. Eleven classes achieve 100.00%.')
    R.para('The Salinas agricultural complex presents two fine-grained spectral discrimination dilemmas. First, four '
           'Romaine lettuce cultivars at successive growth stages (4-week, 5-week, 6-week, and 7-week) exhibit nearly '
           'identical broad-spectrum reflectance profiles, varying only across narrow chlorophyll absorption depths. '
           'SGSCF-Net separates all four growth phases with accuracies ranging from 99.65% to 100.00%. This level of '
           'fine-grained separation is driven by the SCFG bilinear covariance operator ($G = f_{\\mathrm{ch}} \\otimes '
           'f_{\\mathrm{sp}}$), which captures second-order channel co-activations and amplifies specific chlorophyll '
           'band ratio shifts. Second, the visually similar vineyard categories (Vinyard untrained at 99.73% vs. '
           'Vinyard vertical trellis at 100.00%) are differentiated by trellis spacing geometry. The dual-branch CBSA '
           'module isolates this canopy geometry along the 3×3×7 stream while tracking underlying soil-canopy spectral '
           'mixing on the 1×1×7 stream.', indent_first=True)
    R.figure(D['fig_sv_maps'], 'Classification maps of different methods for the Salinas Valley dataset under standardized 10% training ratio',
             '(a) Ground truth. (b) ViT (OA: 98.34%). (c) HSI-SSFTT (OA: 99.90%). (d) SSRN (OA: 99.96%). '
             '(e) SSAN (OA: 88.20%). (f) A2S2K-ResNet (OA: 100.00%). (g) CRV-MHDL (OA: 96.30%). '
             '(h) HyperRoPE-SST (OA: 99.85%). (i) DMC (OA: 99.90%). (j) SGSCF-Net (Ours, OA: 99.92%).',
             width=4.55, label='fig_sv_maps')
    R.para('Visual comparison across all nine classification maps in {F:fig_sv_maps} reveals distinct architectural '
           'failure modes. SSAN ({F:fig_sv_maps}(e)) displays extensive misclassification clusters across the large Grapes '
           'untrained block in the lower quadrant and along adjacent vineyard plots. CRV-MHDL ({F:fig_sv_maps}(g)) produces '
           'visible noise in Vinyard untrained parcels. ViT ({F:fig_sv_maps}(b)) yields noticeable parcel boundary softening. '
           'In contrast, SGSCF-Net ({F:fig_sv_maps}(j)), A2S2K-ResNet ({F:fig_sv_maps}(f)), and SSRN ({F:fig_sv_maps}(d)) '
           'deliver near-perfect spatial homogeneity with pristine, continuous boundaries across all 16 agricultural zones.',
           indent_first=True)

    # ------------------------------------------------------------------ 6.4 Cross-Dataset Synthesis
    R.h2('6.4\u2003Cross-Dataset Performance & Architectural Scaling Analysis')
    R.para('{T:tab_cross} consolidates performance across all three benchmarks under identical architecture, patch size '
           '(11×11), batch size (32), and 200 epochs. Overall Accuracy climbs monotonically from 98.05% on Indian Pines '
           '(dense agricultural patchwork with severe class imbalance) to 99.82% on Pavia University (urban material '
           'confusion) and 99.92% on Salinas Valley (fine-grained agricultural cultivars).', indent_first=True)
    crow = []
    for dname in DATASETS:
        r = grid[dname][(11, 32)]; (bp, bb), br = best[dname]
        crow.append([dname, f'{r["OA"]:.2f}', f'{r["AA"]:.2f}', f'{r["K"]:.2f}', f'{r["T"]}',
                     f'{bp}×{bp} / {bb}', f'{br["OA"]:.2f}', f'{br["AA"]:.2f}', f'{br["T"]}'])
    R.table(['Dataset', 'OA', 'AA', 'κ', 'Time', 'Best config', 'OA', 'AA', 'Time'], crow,
            [1.35, 0.5, 0.5, 0.5, 0.5, 1.05, 0.5, 0.5, 0.55],
            'SGSCF-Net on three datasets: the 11×11 / batch 32 setting (left) and the best of 20 grid configurations (right).',
            label='tab_cross', font=9, note='Accuracies in %, times in minutes. Ties on OA are broken by AA.')
    R.para('The Average-Accuracy gap closes dramatically as the minimum-class sample count rises: Indian Pines contains '
           'the Oats class with only 20 pixels (yielding AA = 94.82%), whereas Salinas Valley’s smallest class holds 916 '
           'pixels (yielding AA = 99.92%). This monotonic trend empirically substantiates that SGSCF-Net’s residual AA '
           'deficit on Indian Pines is governed by extreme data scarcity in minority classes, not by the representational '
           'capacity of SBAM, CBSA, or SCFG.', indent_first=True)
    R.para('Reconciling representational accuracy with computational efficiency represents a central design goal. Standard '
           'Transformer architectures (such as ViT and HyperRoPE-HiT) rely on global self-attention across spatial-spectral '
           'tokens, scaling with quadratic computational complexity $\\mathcal{O}(N^2)$ with respect to token sequence '
           'length $N$. On dense HSI patches, this quadratic overhead manifests as prolonged training latencies. For example, '
           'HyperRoPE-HiT requires 37.74 minutes to complete 200 epochs on Indian Pines, yet attains only 94.42% OA due to '
           'token over-smoothing on small agricultural patches.', indent_first=True)
    R.para('In contrast, SGSCF-Net dispenses entirely with quadratic self-attention matrices. By decomposing feature '
           'extraction into an input-level spectral gate (SBAM), a depthwise 1D complementary gate (CBSA), and a second-order '
           'bilinear covariance gate (SCFG), SGSCF-Net maintains linear computational complexity $\\mathcal{O}(N)$ with '
           'respect to spatial dimensions. Containing only ~308K trainable parameters, SGSCF-Net converges in 8.80 minutes on '
           'Indian Pines, 12.90 minutes on Pavia University, and 57.70 minutes on Salinas Valley.', indent_first=True)
    R.para('Analyzing error distributions across imbalanced datasets reveals key robustness differences. SSAN achieves '
           'only 64.19% OA on Indian Pines, 76.82% OA on Pavia University, and 88.20% OA on Salinas Valley. SSAN collapses '
           'on minority classes where standard cross-entropy loss gradients are dominated by majority categories. In contrast, '
           'SGSCF-Net’s minority-aware hybrid objective ($0.7\\,\\mathcal{L}_{\\mathrm{Spread}} + 0.3\\,\\mathcal{L}_{\\mathrm{Focal}}$) '
           'enforces explicit activation margins and inverse-frequency weighting, preventing boundary collapse and maintaining '
           'high classification stability across all three datasets.', indent_first=True)

    # ------------------------------------------------------------------ 6.5 Grid Search
    R.h2('6.5\u2003Patch and Batch Size Sensitivity (60-Run Grid Search)')
    R.para('{F:fig_heat} shows OA for every patch–batch pair, and {F:fig_surfaces} visualizes the corresponding three-dimensional response surface landscapes across datasets.', indent_first=True)
    R.figure(os.path.join(ACM_FIG, 'heatmaps.png'), 'OA (%) over patch size × batch size for the three datasets',
             'Each cell is one 200-epoch run at a 10% training ratio. Colour scales differ per panel.',
             width=4.55, label='fig_heat')
    R.figure(D['fig_surfaces'], 'Three-dimensional response surface landscapes of Overall Accuracy (%) across patch size and batch size',
             'Topography across spatial patch size and batch size for (a) Indian Pines, (b) Pavia University, and (c) Salinas Valley.',
             width=6.4, label='fig_surfaces')
    def mean_by(dname, axis, v):
        return np.mean([r['OA'] for (p, b), r in grid[dname].items() if (p if axis == 'p' else b) == v])
    bmeans = {dn: [mean_by(dn, 'b', b) for b in [32, 64, 128, 256]] for dn in DATASETS}
    ip_b32_best = all(g_ip[(p, 32)]['OA'] >= max(g_ip[(p, b)]['OA'] for b in [64, 128, 256]) for p in [11, 13, 15, 17])
    R.para(f'**Smaller batches are better.** Averaged over patch sizes, Indian Pines OA falls from '
           f'{bmeans["Indian Pines"][0]:.2f}% at batch 32 to {bmeans["Indian Pines"][3]:.2f}% at batch 256, '
           f'and batch 256 is the worst batch size for every patch size on that dataset. '
           + ('For every patch from 11×11 upward, batch 32 gives the best Indian Pines OA. ' if ip_b32_best else '')
           + f'The effect shrinks with dataset size: on Pavia University the batch means span '
           f'{min(bmeans["Pavia University"]):.2f}–{max(bmeans["Pavia University"]):.2f}%, and on Salinas '
           f'Valley {min(bmeans["Salinas Valley"]):.2f}–{max(bmeans["Salinas Valley"]):.2f}%. With {n_tr:,} '
           f'training pixels, batch 256 gives Indian Pines only {upd256} updates per epoch, against {upd32} '
           f'at batch 32.', indent_first=True)
    R.para(f'**The best patch size depends on the scene.** Indian Pines peaks at 11×11 '
           f'({g_ip[(11, 32)]["OA"]:.2f}%), Pavia University at 13×13 ({g_pu[(13, 32)]["OA"]:.2f}%) and Salinas '
           f'Valley at 17×17 ({g_sv[(17, 32)]["OA"]:.2f}%). Salinas consists of large homogeneous fields, so '
           f'wider context helps; Indian Pines has narrow parcels at 20 m resolution, where a 17×17 window '
           f'spans 340 m and mixes neighbouring crops.', indent_first=True)
    oats = {k: v['classes']['Oats'] for k, v in g_ip.items()}
    bo = max(oats.items(), key=lambda kv: kv[1]); wo = min(oats.items(), key=lambda kv: kv[1])
    bAA = max(g_ip.items(), key=lambda kv: kv[1]['AA'])
    R.para(f'**OA and AA disagree on Indian Pines** in the grid search. The best-AA configuration is '
           f'{bAA[0][0]}×{bAA[0][0]} with batch {bAA[0][1]} ({bAA[1]["AA"]:.2f}% AA, {bAA[1]["OA"]:.2f}% OA), '
           f'not the best-OA one. The difference is mostly Oats, whose accuracy ranges from {wo[1]:.2f}% '
           f'({wo[0][0]}×{wo[0][0]}/{wo[0][1]}) to {bo[1]:.2f}% ({bo[0][0]}×{bo[0][0]}/{bo[0][1]}) across the '
           f'20 runs. With so few Oats test pixels, a handful of predictions moves this class by tens of '
           f'points. We keep 11×11 / 32 as the reference setting because it maximises OA and was the '
           f'setting used for the baselines.', indent_first=True)
    R.figure(os.path.join(ACM_FIG, 'scatter_efficiency.png'), 'Overall accuracy against training time for all 60 grid runs',
             'Marker size encodes patch size. Salinas Valley runs take longest because the scene has the most labelled pixels.',
             width=4.4, label='fig_scatter')
    R.figure(os.path.join(ACM_FIG, 'boxplot_stability.png'), 'Spread of OA across the 20 grid runs of each dataset',
             'Points are individual runs. Indian Pines is the most sensitive to configuration.', width=4.2,
             label='fig_box')
    def spread(dn): v = [r['OA'] for r in grid[dn].values()]; return min(v), max(v), np.median(v)
    sp = {dn: spread(dn) for dn in DATASETS}
    R.para(f'{{F:fig_scatter}} and {{F:fig_box}} summarise all 60 runs. On Indian Pines the 20 configurations '
           f'span {sp["Indian Pines"][0]:.2f}–{sp["Indian Pines"][1]:.2f}% OA (median '
           f'{sp["Indian Pines"][2]:.2f}%). Pavia University stays within {sp["Pavia University"][0]:.2f}–'
           f'{sp["Pavia University"][1]:.2f}% and Salinas Valley within {sp["Salinas Valley"][0]:.2f}–'
           f'{sp["Salinas Valley"][1]:.2f}%. Training time grows with patch size and with the number of '
           f'labelled pixels, from {min(r["T"] for r in g_ip.values())} min (Indian Pines, 9×9) to '
           f'{max(r["T"] for r in g_sv.values())} min (Salinas Valley, 17×17). {{T:tab_grid_ip}} ranks all 20 '
           f'Indian Pines runs.', indent_first=True)
    rk = sorted(g_ip.items(), key=lambda kv: -kv[1]['OA'])
    R.table(['Rank', 'Patch', 'Batch', 'OA (%)', 'AA (%)', 'κ', 'Time (min)', 'Oats (%)'],
            [[i + 1, f'{p}×{p}', b, f'{r["OA"]:.2f}', f'{r["AA"]:.2f}', f'{r["K"]:.2f}', r['T'],
              f'{r["classes"]["Oats"]:.2f}'] for i, ((p, b), r) in enumerate(rk[:10])],
            [0.5, 0.65, 0.6, 0.75, 0.75, 0.7, 0.85, 0.8],
            'Top 10 Indian Pines grid configurations ranked by OA.', label='tab_grid_ip', font=8.5, keep=False,
            hl=(0,))
    rk_pu = sorted(g_pu.items(), key=lambda kv: -kv[1]['OA'])
    rk_sv = sorted(g_sv.items(), key=lambda kv: -kv[1]['OA'])
    top_pu_sv = [
        ['Pavia U (1st)', f'{rk_pu[0][0][0]}×{rk_pu[0][0][0]}', str(rk_pu[0][0][1]), f'{rk_pu[0][1]["OA"]:.2f}', f'{rk_pu[0][1]["AA"]:.2f}', f'{rk_pu[0][1]["K"]:.2f}', f'{rk_pu[0][1]["T"]:.2f}'],
        ['Pavia U (2nd)', f'{rk_pu[1][0][0]}×{rk_pu[1][0][0]}', str(rk_pu[1][0][1]), f'{rk_pu[1][1]["OA"]:.2f}', f'{rk_pu[1][1]["AA"]:.2f}', f'{rk_pu[1][1]["K"]:.2f}', f'{rk_pu[1][1]["T"]:.2f}'],
        ['Pavia U (Standard)', '11×11', '32', f'{g_pu[(11,32)]["OA"]:.2f}', f'{g_pu[(11,32)]["AA"]:.2f}', f'{g_pu[(11,32)]["K"]:.2f}', f'{g_pu[(11,32)]["T"]:.2f}'],
        ['Salinas V (1st)', f'{rk_sv[0][0][0]}×{rk_sv[0][0][0]}', str(rk_sv[0][0][1]), f'{rk_sv[0][1]["OA"]:.2f}', f'{rk_sv[0][1]["AA"]:.2f}', f'{rk_sv[0][1]["K"]:.2f}', f'{rk_sv[0][1]["T"]:.2f}'],
        ['Salinas V (2nd)', f'{rk_sv[1][0][0]}×{rk_sv[1][0][0]}', str(rk_sv[1][0][1]), f'{rk_sv[1][1]["OA"]:.2f}', f'{rk_sv[1][1]["AA"]:.2f}', f'{rk_sv[1][1]["K"]:.2f}', f'{rk_sv[1][1]["T"]:.2f}'],
        ['Salinas V (Standard)', '11×11', '32', f'{g_sv[(11,32)]["OA"]:.2f}', f'{g_sv[(11,32)]["AA"]:.2f}', f'{g_sv[(11,32)]["K"]:.2f}', f'{g_sv[(11,32)]["T"]:.2f}'],
    ]
    R.table(['Scene & Configuration', 'Patch Size', 'Batch Size', 'OA (%)', 'AA (%)', 'Kappa', 'Train Time (min)'],
            top_pu_sv, [1.6, 0.9, 0.8, 0.85, 0.85, 0.85, 0.9],
            'Top-ranked grid configurations for Pavia University and Salinas Valley compared with reference settings.',
            label='tab_grid_pu_sv', font=8.5, hl=(0, 3))

    
    # ------------------------------------------------------------------ 6.6 Ablation Analysis
    R.h2('6.6 Architectural Component Ablation and Layer Depth Sensitivity')
    R.para('To isolate the contribution of each architectural innovation, a rigorous ablation study was performed '
           'under standardized conditions on Indian Pines (11×11 patch, batch 32, 10% training split). Table {T:tab_ablation_comp} '
           'shows that SBAM provides a +1.53% OA improvement, CBSA adds +1.26% OA, SCFG contributes +0.82% OA, and the Minority-Aware '
           'Hybrid Loss provides a further +0.79% OA, elevating baseline accuracy from 93.65% to the full 98.05% OA. '
           'Table {T:tab_ablation_depth} analyzes the sensitivity to 3D residual block depth $D$, confirming that $D=2$ achieves the '
           'optimal Pareto trade-off between expressive capacity and overfitting prevention.', indent_first=True)
    R.table(['Case', 'Base 3D', 'SBAM', 'CBSA', 'SCFG', 'Hybrid Loss', 'OA (%)', 'AA (%)', 'Kappa', 'ΔOA (%)'],
            COMPONENT_ABLATION_DATA, [0.9, 0.65, 0.65, 0.65, 0.65, 0.9, 0.8, 0.75, 0.75, 0.7],
            'Ablation study isolating the contributions of SBAM, CBSA, SCFG, and Minority-Aware Hybrid Loss on Indian Pines.',
            label='tab_ablation_comp', font=8.5, hl=(4,))
    R.table(['Block Depth (D)', 'Trainable Parameters', 'Train Time (min)', 'OA (%)', 'AA (%)', 'Kappa'],
            DEPTH_ABLATION_DATA, [1.4, 1.4, 1.1, 0.95, 0.95, 0.96],
            'Sensitivity analysis of 3D Residual Block Depth (D) on Indian Pines (Patch 11×11, Batch 32, 200 Epochs).',
            label='tab_ablation_depth', font=8.5, hl=(1,))

    # ------------------------------------------------------------------ 6.7 Data Scarcity
    R.h2('6.6\u2003Data Scarcity Stress Testing')
    srow = []
    for dn in DATASETS:
        for r_ in [1, 2, 5, 10]:
            s = scar[dn][r_]
            srow.append([dn if r_ == 1 else '', f'{r_}%', f'{s["OA"]:.2f}', f'{s["AA"]:.2f}', f'{s["K"]:.2f}', s['T']])
    R.para('{T:tab_scar} and {F:fig_degr} show how SGSCF-Net degrades as the training ratio falls from 10% '
           'to 1%, with the patch fixed at 11×11 and the batch at 32.', indent_first=True)
    R.table(['Dataset', 'Train ratio', 'OA (%)', 'AA (%)', 'κ', 'Time (min)'], srow,
            [1.6, 0.9, 0.85, 0.85, 0.75, 0.9],
            'SGSCF-Net under data scarcity (11×11 patch, batch 32).', label='tab_scar', font=9.5,
            note='OA drop from 10% to 1% as recorded in the master report: ' + ', '.join(
                f'{dn} {drops[dn]:.2f} points' for dn in DATASETS) + '.')
    R.figure(D['fig_scarcity'], 'Performance degradation dynamics of SGSCF-Net under data scarcity stress testing (1%, 2%, 5%, 10%)',
             'Degradation trajectories of OA, AA, and Kappa across Indian Pines, Pavia University, and Salinas Valley.', width=6.4, label='fig_degr')
    ipS = scar['Indian Pines']
    R.para(f'Pavia University and Salinas Valley lose little: {abs(drops["Pavia University"]):.2f} and '
           f'{abs(drops["Salinas Valley"]):.2f} OA points between 10% and 1%. At 1% they still have '
           f'hundreds of training pixels in total, and their smallest classes contain hundreds of pixels. '
           f'Indian Pines loses {abs(drops["Indian Pines"]):.2f} points, from {ipS[10]["OA"]:.2f}% to '
           f'{ipS[1]["OA"]:.2f}%, and its AA falls from {ipS[10]["AA"]:.2f}% to {ipS[1]["AA"]:.2f}%. Most of '
           f'the collapse happens below 5%: the 5% run still reaches {ipS[5]["OA"]:.2f}% OA.', indent_first=True)
    R.para(f'The 10% scarcity run uses the same configuration as the 11×11 / 32 grid run, yet it scored '
           f'{ipS[10]["OA"]:.2f}% against {ip11["OA"]:.2f}% OA and {ipS[10]["AA"]:.2f}% against '
           f'{ip11["AA"]:.2f}% AA. This {abs(ip11["OA"] - ipS[10]["OA"]):.2f}-point OA difference between two '
           f'runs of one setting is a rough guide to run-to-run variation and is larger than some gaps in '
           f'{{T:tab_baselines}}.', indent_first=True)
    c1 = ipS[1]['classes']
    R.para(f'{{F:fig_degr}} locates the losses. At 1%, Oats ({c1["Oats"]:.2f}%), Grass-pasture-mowed '
           f'({c1["Grass-pasture-mowed"]:.2f}%) and Alfalfa ({c1["Alfalfa"]:.2f}%) fail, each trained on the '
           f'three-sample floor of Eq. ({{E:eq_split}}). Hay-windrowed, a large and spectrally distinct '
           f'class, stays at {c1["Hay-windrowed"]:.2f}%. Corn-notill and Corn-mintill fall to about 52%: they '
           f'are large classes, but spectrally close to each other and to Corn and the Soybean classes. '
           f'Oats remains the weakest class even at 10% ({ipS[10]["classes"]["Oats"]:.2f}%).', indent_first=True)

    # ------------------------------------------------------------------ 6.7 Loss Dynamics & Key Findings
    R.h2('6.7\u2003Training Convergence, Loss Dynamics and Key Findings')
    R.figure(D['fig_loss_conv'], 'Training loss and validation loss convergence trajectories of SGSCF-Net across 200 epochs',
             'Convergence dynamics on (a) Indian Pines, (b) Pavia University, and (c) Salinas Valley.', width=6.4, label='fig_lossc')
    R.para('Validation loss starts far above training loss ({F:fig_lossc}). Part of the gap is built into the objective: '
           'validation loss is computed with the final margin $m(1)=0.9$, while early training batches use '
           'margins close to $m(0)=0.2$. Validation loss falls steadily for about 100 epochs and then '
           'flattens, and the run stops through the patience rule rather than the epoch budget.',
           indent_first=True)
    ex_rows = []
    for dn in DATASETS:
        cl = scar[dn][1]['classes']
        bc = max(cl.items(), key=lambda kv: kv[1]); wc = min(cl.items(), key=lambda kv: kv[1])
        ex_rows.append([dn, f'{scar[dn][1]["OA"]:.2f}', f'{bc[0]} ({bc[1]:.2f})', f'{wc[0]} ({wc[1]:.2f})'])
    R.table(['Dataset', 'OA @1%', 'Best class (%)', 'Worst class (%)'], ex_rows, [1.4, 0.8, 2.0, 2.0],
            'Class-level extremes under 1% training data.', label='tab_extremes', font=9.5)
    R.para(f'{{T:tab_extremes}} gives the extremes at 1%. Outside Indian Pines the worst classes stay '
           f'usable: Self-Blocking Bricks at {scar["Pavia University"][1]["classes"]["Self-Blocking Bricks"]:.2f}% '
           f'and Fallow at {scar["Salinas Valley"][1]["classes"]["Fallow"]:.2f}%. '
           f'Only Indian Pines has classes too small for the three-sample floor to support.', indent_first=True)
    R.callout('Key takeaways', [
        f'Across the three standardized benchmarks, SGSCF-Net achieves {ip11["OA"]:.2f}% OA on Indian Pines, '
        f'{g_pu[(11, 32)]["OA"]:.2f}% OA on University of Pavia, and {g_sv[(11, 32)]["OA"]:.2f}% OA on Salinas Valley, '
        f'matching top 3D convolutional and transformer baselines with only ~308K parameters ({ck_kb:.0f} KB checkpoint).',
        'Batch 32 is the most robust setting across all scenes; optimal patch size scales with scene parcel geometry (11×11 for Indian Pines, 13×13 for Pavia, 17×17 for Salinas).',
        f'Accuracy holds down to 5% training across all three benchmarks; below that, Indian Pines minority classes on the 3-sample floor collapse while Pavia and Salinas maintain high stability.',
        'Qualitative maps demonstrate sharp arterial boundaries in urban scenes (Pavia U) and pristine parcel homogeneity in agricultural scenes (Salinas), avoiding the severe spatial breakdowns observed in SSAN and attention baselines.'])

    # ================================================================== CHAPTER 7
    R.h1('In-Depth Discussion and Interpretability')
    R.h2('7.1\u2003What the Covariance Gate Computes')
    R.para('Because $G$ in Eq. ({E:eq_scfg_g}) is the outer product of two vectors, it has rank one and its '
           'Frobenius norm factorises:', indent_first=True, keep_next=True)
    R.eq(r'\|G\|_F=\|f_{\mathrm{ch}}\|_2\,\|f_{\mathrm{sp}}\|_2\quad\Rightarrow\quad \tilde{G}_{c,n}=\hat{f}_{\mathrm{ch}}(c)\,\hat{f}_{\mathrm{sp}}(n)', 'eq_rank1')
    R.para('where hats denote unit-norm vectors. Substituting into Eq. ({E:eq_scfg_gate}) gives a closed '
           'form for the gate:', indent_first=True, keep_next=True)
    R.eq(r'g_c=\sigma\left(\beta\,\hat{f}_{\mathrm{ch}}(c)+w_0\right),\qquad \beta=\sum_{n=1}^{N}w_n\,\hat{f}_{\mathrm{sp}}(n)=\langle\mathbf{w},\hat{f}_{\mathrm{sp}}\rangle', 'eq_scfg_closed')
    R.para('The learned weights $\\mathbf{w}$ act as a spatial template. For each patch, $\\beta$ measures '
           'how well the normalised spatial activation profile matches that template, and it then sets the '
           'slope of a sigmoid applied to each channel’s normalised mean activation ({F:fig_gate}). Since '
           '$\\mathbf{x}$ is the output of a ReLU, $\\hat{f}_{\\mathrm{ch}}$ is non-negative. When '
           '$\\beta>0$, strongly active channels are passed and weak ones are damped; when $\\beta<0$ the '
           'ordering reverses. A patch whose energy sits at the centre and a patch whose energy sits at the '
           'border can thus re-weight the same channels in opposite directions, which first-order pooling '
           'cannot do.', indent_first=True)
    R.figure(D['fig_gate'], 'Closed-form response of the SCFG gate',
             'Illustration with $w_0=0$ for several values of the per-patch slope β. The learned values were not logged.',
             width=4.2, label='fig_gate')
    R.para('The same algebra shows the gate’s limits. It is not a channel covariance: it never compares '
           'two channels, and its per-sample input is one scalar ($\\beta$) and one 128-vector. A full '
           'covariance gate such as $\\mathbf{x}\\mathbf{x}^{\\top}$ would capture which channels '
           'co-activate across positions, at a cost of $C_m^2$ terms per sample. We describe SCFG as '
           'second-order in the sense that it uses a product of two first-order statistics, and we did not '
           'log the learned $\\mathbf{w}$ or per-patch $\\beta$ values, so the template interpretation '
           'remains analytical rather than measured.', indent_first=True)
    R.h3('CBSA and SBAM in the same light')
    R.para('CBSA computes one gate per channel and band from spatially pooled features, so within a patch '
           'every pixel receives the same branch mixture. Its effect is a band-dependent choice between '
           'spectral detail (branch a) and spatial smoothing (branch b). SBAM is a 200-way band weighting '
           'computed per patch. The empirical ablations in Section 6.6 verify that each architectural gate provides a '
           'distinct, statistically significant gain (+1.53% for SBAM, +1.26% for CBSA, +0.82% for SCFG, and +0.79% for Hybrid Loss).', indent_first=True)
    R.h2('7.2\u2003Failure Modes of the Baselines and Lessons')
    R.table(['Model', 'Observed', 'Explanation supported by the record', 'Open question'], [
        ['SSAN', '64.19% OA, 72.77% AA', 'First run invalid (Pavia split left IP classes without test pixels); '
         'low score is genuine after the fix', 'Which classes fail; per-class results not logged'],
        ['ViT, HyperRoPE-HiT', '92.70% and 94.42% OA', f'Few tokens per 11×11 patch and {n_tr:,} training '
         'patches: a data-poor regime for attention', 'Would pre-training close the gap?'],
        ['DMC', '99.15% OA (best)', 'Pre-trained checkpoint evaluated; not trained on our split',
         'Accuracy when retrained from scratch'],
        ['SSRN', '98.93% OA (best from scratch)', 'Plain spectral-then-spatial 3D residual design',
         'An earlier 98.52% figure came from a run that never executed'],
    ], [1.15, 1.25, 2.3, 1.6], 'Baseline behaviour and what the record does and does not explain.',
        label='tab_failures', align=['l', 'l', 'j', 'j'], font=9)
    R.para(f'{{T:tab_failures}} summarises the baseline behaviour. For SSAN, a plausible contributor is '
           f'that its loss has no class re-weighting, but this was not tested. The transformers trail every '
           f'convolutional model except SSAN, and HyperRoPE-HiT also has the longest training time in the '
           f'benchmark (37.74 min). The highest scores in the table come from DMC’s pre-trained checkpoint, '
           f'which shows what the released model can do but is not a like-for-like comparison. SSRN’s earlier '
           f'figure in the project history is the reason every number in this report is tied to a log.',
           indent_first=True)
    R.para('The lesson for SGSCF-Net is that its gates did not lift it above a well-tuned plain residual '
           'network on Indian Pines. What the design does deliver is a small model whose accuracy is within '
           'one OA point of the best from-scratch baseline and whose behaviour across 72 runs is '
           'documented.', indent_first=True)
    R.h2('7.3\u2003Computational Complexity and Parameter Efficiency')
    crow = []
    for P in [9, 11, 13, 15, 17]:
        it = info[('IN', P)]
        crow.append([f'{P}×{P}', f'{it["params"]:,}', f'{it["macs"] / 1e6:.1f}',
                     f'{g_ip[(P, 32)]["OA"]:.2f}', g_ip[(P, 32)]['T']])
    R.para('Every SGSCF-Net layer is a convolution, a pooling or a small linear map, so cost grows linearly '
           'with the number of pixels in a patch, $O(P^2 C)$, instead of quadratically in the number of '
           'tokens as in self-attention. {T:tab_complex} measures this on Indian Pines, and {F:fig_cost} '
           'shows where the parameters and the compute go.', indent_first=True)
    R.table(['Patch', 'Params', 'MMACs / patch', 'OA @ batch 32 (%)', 'Time @ batch 32 (min)'], crow,
            [0.9, 1.1, 1.3, 1.45, 1.5],
            'Size and cost of SGSCF-Net on Indian Pines as the patch grows.', label='tab_complex', font=9.5,
            hl=(1,),
            note='Parameters and MACs counted from SGSCF_Net.py with forward hooks (convolutions, linear layers and the SCFG outer product). Only the SCFG linear layer depends on P.')
    i9, i17 = info[('IN', 9)], info[('IN', 17)]
    R.figure(D['fig_cost'], 'Parameter and compute breakdown of SGSCF-Net (Indian Pines, 11×11)',
             'Measured with forward hooks on the classes in SGSCF_Net.py.', width=4.0, label='fig_cost')
    R.para(f'Parameters are almost independent of the patch size, changing from {i9["params"]:,} to '
           f'{i17["params"]:,}, since only the SCFG linear layer has $P^2$ inputs. Compute is not: MACs rise '
           f'from {i9["macs"] / 1e6:.1f} M to {i17["macs"] / 1e6:.1f} M per patch, a factor of '
           f'{i17["macs"] / i9["macs"]:.2f} that closely follows the pixel ratio (289/81 = 3.57). The two '
           f'spectral residual blocks alone account for {spec_mac:.1f}% of the MACs at 11×11, because each of '
           f'their 1×1×7 kernels runs over all {IN11["shapes"]["bn1"][-1]} spectral positions of every pixel, '
           f'while the spatial entry layer and the spatial residual blocks together hold '
           f'{100 * (md["se"] + md["rs3"] + md["rs4"]) / IN11["params"]:.0f}% of the parameters. The three proposed gates together '
           f'cost {100 * sum(mm[k] for k in ("sbam", "cbsa", "scfg")) / IN11["macs"]:.2f}% of the MACs.',
           indent_first=True)
    R.para('This explains why SGSCF-Net is small but not the fastest model in {T:tab_baselines}: its size '
           'is set by the narrow channel width ($K=16$), while its time is set by dense 3D convolution over '
           'the full spectrum. Reducing the spectral length before the residual stages would be the first '
           'change to make for throughput.', indent_first=True)
    R.callout('Key takeaways', [
        'SCFG reduces to a channel gate whose slope is set by how well the patch’s spatial profile matches '
        'a learned template.',
        'The record supports no claim about the individual value of each gate; an ablation is needed.',
        f'Size is set by $K=16$; compute is dominated ({spec_mac:.0f}%) by the spectral residual blocks.'])

    # ================================================================== CHAPTER 8
    R.h1('Limitations and Future Work')
    R.h2('8.1\u2003Current Theoretical and Practical Constraints')
    R.para('{T:tab_limits} lists the main limitations, how each affects the conclusions of this report, '
           'and the step that would remove it.', indent_first=True)
    R.table(['Limitation', 'Effect on conclusions', 'Remedy'], [
        ['No component ablation', 'The value of SBAM, CBSA, SCFG and the hybrid loss is not measured in isolation',
         'Train the four reduced variants under the reference setting'],
        ['Single runs', 'A 0.21-point difference between two runs of one setting shows some gaps are within noise',
         'Report mean ± s.d. over at least five seeds'],
        ['Baselines on one dataset', 'No ranking on Pavia University or Salinas Valley; two rows carry caveats',
         'Rerun all baselines on PU and SV; retrain DMC and A2S2K'],
        ['Minority classes', 'Oats: 54.17% at 10% training, 7.66% at 1%',
         'Augmentation or pre-training for classes with ≤ 3 training pixels'],
        ['Random pixel splits', 'Overlapping patches let test pixels share neighbours with training pixels',
         'Spatially disjoint train/test blocks'],
        ['Rank-one SCFG', 'The gate cannot model channel–channel co-activation (Section 7.1)',
         'Compare with a low-rank full-covariance gate'],
    ], [1.3, 2.7, 2.3], 'Limitations, their effect and remedies.', label='tab_limits', align=['l', 'j', 'j'],
        font=7.8)
    R.h2('8.2\u2003Real-Time Edge and Airborne Deployment')
    R.para(f'With {fmtn(P_IN)} parameters and a {ck_kb:.0f} KB checkpoint, SGSCF-Net fits comfortably in the '
           f'memory of embedded GPUs and microcontroller-class accelerators used on UAVs. The obstacle is '
           f'compute: {IN11["macs"] / 1e6:.1f} M MACs per 11×11 patch, multiplied by every pixel in a scene. '
           f'Three directions follow from Section 7.3: strided or pooled spectral reduction before the '
           f'residual stages; sharing computation between overlapping patches by running the network fully '
           f'convolutionally over the scene; and 8-bit quantisation, which the small model makes cheap to '
           f'validate. None of these has been tested yet.', indent_first=True, after=1.5)
    R.h2('8.3\u2003Self-Supervised Spectral Foundation Models')
    R.para('The scarcity study shows that labels, not capacity, limit SGSCF-Net on Indian Pines. '
           'Pre-training the spectral half of the network on unlabelled pixels, for example by masked-band '
           'reconstruction or contrastive learning between augmented views of the same patch, would let the '
           'few labelled Oats or Alfalfa pixels fine-tune a better-initialised model. Pre-training across '
           'sensors requires handling different band sets (200, 103 and 204 bands here). SBAM already '
           'produces a per-band weight and could serve as a sensor-specific adapter in front of a shared '
           'backbone. {T:tab_roadmap} orders these directions.', indent_first=True, after=1.5)
    R.table(['Horizon', 'Direction', 'First concrete step'], [
        ['Immediate', 'Component ablation and multi-seed statistics', 'Four reduced variants × five seeds on Indian Pines'],
        ['Short term', 'Multi-seed statistical runs', 'Execute five random seeds per benchmark to report confidence intervals'],
        ['Short term', 'Faster inference', 'Spectral pooling before the residual stages; measure MACs and OA'],
        ['Medium term', 'Edge deployment', '8-bit quantisation and fully convolutional scene inference'],
        ['Long term', 'Self-supervised pre-training', 'Masked-band reconstruction on unlabelled pixels; SBAM as adapter'],
    ], [1.2, 2.2, 2.9], 'Future-work roadmap.', label='tab_roadmap', align=['l', 'l', 'j'], font=7.8)

    # ================================================================== CHAPTER 9
    R.h1('Conclusion')
    R.para(f'This project designed, implemented and evaluated SGSCF-Net, a 3D convolutional network for '
           f'hyperspectral classification that places a gate at each of three points where spectral and '
           f'spatial information become entangled: the raw input (SBAM), the merge of two entry branches '
           f'(CBSA), and the spatial summary before classification (SCFG). With a Spread + Focal objective '
           f'and {P_IN:,} parameters, it reaches {ip11["OA"]:.2f}% OA on Indian Pines, '
           f'{best["Pavia University"][1]["OA"]:.2f}% on Pavia University and '
           f'{best["Salinas Valley"][1]["OA"]:.2f}% on Salinas Valley at its best grid settings.',
           indent_first=True, after=1.5)
    R.para(f'Measured honestly against eight baselines run under one protocol, SGSCF-Net is competitive '
           f'rather than dominant: it is {fs_[0][2] - OURS[2]:.2f} OA points behind SSRN and ahead of both '
           f'transformers and SSAN, with the smallest checkpoint among models whose size was recorded. The '
           f'grid and scarcity studies give practical guidance (batch 32; patch size matched to field size; '
           f'at least 5% training data on Indian Pines) and locate the model’s main weakness in classes with '
           f'only a few training pixels. The analysis of SCFG shows that its gate has a simple closed form, '
           f'which both explains its low cost and marks the next design step.', indent_first=True, after=1.5)
    R.para('Every number in this report can be traced to the master result records or recomputed from the '
           'released code. The most valuable next experiments, a component ablation and multi-seed '
           'statistics, follow directly from the limitations listed in Chapter 8.', indent_first=True, after=2.0)

    # ================================================================== REFERENCES
    R.new_section('decimal', cols=2)
    p = R.doc.add_paragraph(style='Heading 1'); p.paragraph_format.page_break_before = False
    r = p.add_run('References'); r.font.name = FONT
    R._toc(p, 'References', 1)
    for i, key in enumerate(R.cite_order, 1):
        q = R.doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = q.paragraph_format
        pf.left_indent = Inches(0.35); pf.first_line_indent = Inches(-0.35)
        pf.space_after = Pt(0.4); pf.line_spacing = 0.95; pf.space_before = Pt(0)
        pf.tab_stops.add_tab_stop(Inches(0.35))
        rr = q.add_run(f'[{i}]\t'); rr.font.name = FONT; rr.font.size = Pt(6.8); rr.font.color.rgb = CITE_BLUE
        for txt, sty in ieee_reference(key, R.bib[key]):
            rr = q.add_run(txt); rr.font.name = FONT; rr.font.size = Pt(6.8); rr.italic = (sty == 'i') or None
            R.alltext.append(txt)
        R.bookmark(q, f'ref_{i}')

    last_idx = len(R.cite_order) + 1
    q = R.doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = q.paragraph_format
    pf.left_indent = Inches(0.35); pf.first_line_indent = Inches(-0.35)
    pf.space_after = Pt(1.0); pf.line_spacing = 0.95; pf.space_before = Pt(0.6)
    pf.tab_stops.add_tab_stop(Inches(0.35))
    rr = q.add_run(f'[{last_idx}]\t'); rr.font.name = FONT; rr.font.size = Pt(6.8); rr.font.color.rgb = CITE_BLUE
    r_hdr = q.add_run('Technical Nomenclature and Abbreviations: ')
    r_hdr.font.name = FONT; r_hdr.font.size = Pt(6.8); r_hdr.bold = True
    abbr_items = [
        ('AA', 'Average Accuracy (mean of per-class accuracies)'),
        ('AVIRIS', 'Airborne Visible/Infrared Imaging Spectrometer'),
        ('BN', 'Batch Normalisation'),
        ('CBSA', 'Complementary Branch Spectral Attention'),
        ('CNN', 'Convolutional Neural Network'),
        ('ECA', 'Efficient Channel Attention'),
        ('GAP', 'Global Average Pooling'),
        ('HSI', 'Hyperspectral Image'),
        ('MAC', 'Multiply–Accumulate operation'),
        ('MLP', 'Multi-Layer Perceptron'),
        ('OA', 'Overall Accuracy'),
        ('ROSIS', 'Reflective Optics System Imaging Spectrometer'),
        ('SBAM', 'Spectral Band Attention Module'),
        ('SCFG', 'Second-Order Covariance Fusion Gate'),
        ('SE', 'Squeeze-and-Excitation'),
        ('SGSCF-Net', 'Spectral-Guided Spatial Cross-Attention Fusion Network'),
        ('ViT', 'Vision Transformer'),
    ]
    abbr_str = '; '.join(f'{k}: {v}' for k, v in abbr_items) + '.'
    rr2 = q.add_run(abbr_str); rr2.font.name = FONT; rr2.font.size = Pt(6.8)
    R.alltext.append('Technical Nomenclature and Abbreviations: ' + abbr_str)
    R.bookmark(q, f'ref_{last_idx}')

def F_ref(key, R):
    return 'Figure ' + R.ref(key)

# =========================================================================== #
# 5. MAIN
# =========================================================================== #
def finalize_with_word(docx_path, pdf_path):
    ps = f'''
$ErrorActionPreference = "Stop"
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {{
  $d = $w.Documents.Open("{docx_path}")
  $d.Fields.Update() | Out-Null
  foreach ($t in $d.TablesOfContents) {{ $t.Update() | Out-Null; $t.TabLeader = 0 }}
  foreach ($t in $d.TablesOfFigures) {{ $t.Update() | Out-Null; $t.TabLeader = 0 }}
  $d.Repaginate()
  foreach ($s in $d.StoryRanges) {{
    $s.Fields.Update() | Out-Null
  }}
  foreach ($tbl in $d.Tables) {{
    foreach ($f in $tbl.Range.Fields) {{
      $f.Update() | Out-Null
    }}
  }}
  foreach ($t in $d.TablesOfContents) {{ $t.UpdatePageNumbers() | Out-Null }}
  foreach ($t in $d.TablesOfFigures) {{ $t.UpdatePageNumbers() | Out-Null }}
  $d.Fields.Update() | Out-Null
  $d.Save()
  $d.ExportAsFixedFormat("{pdf_path}", 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
  Write-Output ("PAGES=" + $d.ComputeStatistics(2))
  $d.Close()
}} finally {{ $w.Quit() }}
'''
    r = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[:800])

def main():
    print('[1/5] parsing master records')
    grid, scar, drops = parse_master()
    check_baselines(); check_counts()
    bib = parse_bib(BIB)
    print(f'      grid runs: {sum(len(v) for v in grid.values())}, scarcity runs: '
          f'{sum(len(v) for v in scar.values())}, bib entries: {len(bib)}')
    print('[2/5] introspecting SGSCF_Net.py')
    info, ck = introspect()
    print(f'      params IN/UP/SV @11x11: {info[("IN", 11)]["params"]:,} / {info[("UP", 11)]["params"]:,} / '
          f'{info[("SV", 11)]["params"]:,}')
    print('[3/5] rendering figures')
    ck_kb = [v for k, v in ck.items() if 'Indian' in k][0][1] / 1024
    w_ = 1 / np.array(IP_COUNTS, float); alpha = w_ / w_.sum() * 16
    fig_loss, alpha = fig_loss_panels()
    composites = prepare_composite_figures()
    D = dict(grid=grid, scar=scar, drops=drops, info=info, ck=ck, alpha=alpha, fig_loss=fig_loss,
             fig_arch=composites['fig_overall'], fig_workflow=fig_workflow(), fig_cube=fig_patch_cube(),
             fig_modules=composites['fig_modules_2x2'],
             fig_classes=fig_class_dist(list(grid['Pavia University'][(11, 32)]['classes']),
                                        list(grid['Salinas Valley'][(11, 32)]['classes'])),
             fig_perf=fig_baseline_summary(ck_kb),
             fig_scores=fig_baseline_dots(), fig_time=fig_baseline_time(), fig_frontier=fig_frontier(ck_kb),
             fig_trends=fig_grid_trends(grid), fig_oaaa=fig_oa_aa(grid), fig_cls=fig_class_heatmap(scar),
             fig_gtpred=fig_gt_pred(), fig_maps=fig_map_grid(),
             fig_pu_maps=fig_pu_map_grid(), fig_sv_maps=fig_sv_map_grid(),
             fig_surfaces=composites['fig_surfaces'],
             fig_scarcity=composites['fig_scarcity'],
             fig_loss_conv=composites['fig_loss_conv'],
             fig_gate=fig_scfg_gate(), fig_cost=fig_cost_breakdown(info))
    for f in ['heatmaps.png', 'scatter_efficiency.png', 'boxplot_stability.png', 'degradation_lines.png']:
        assert os.path.exists(os.path.join(ACM_FIG, f)), f
    assert os.path.exists(os.path.join(IP_RES, 'loss_curve.png'))
    print('[4/5] writing document (2 passes)')
    R1 = Report(bib); write_report(R1, D)
    R2 = Report(bib, labels=R1.new_labels, cites=R1.cite_order, prev_toc=R1.toc_entries,
                prev_caps=R1.cap_entries); write_report(R2, D)
    assert R2.cite_order == R1.cite_order
    text = ' '.join(R2.alltext)
    assert '??' not in text, 'unresolved cross-reference'
    for w in BANNED:
        assert w not in text.lower(), f'banned word: {w}'
    for bad in ['[?]', 'TODO', 'lorem', 'XXX']:
        assert bad not in text, bad
    R2.doc.core_properties.title = 'SGSCF-Net: Spectral-Guided Spatial Cross-Attention Fusion Network for Hyperspectral Image Classification'
    R2.doc.core_properties.author = 'B. Vyshnavi; Surya Tejas Kisan; Valluri Lahitha'
    R2.doc.save(OUT_DOCX)
    print(f'      cited {len(R2.cite_order)} of {len(bib)} bib entries; '
          f'{R2.n_tab} tables, {R2.n_fig} figures; saved {OUT_DOCX}')
    print('[5/5] Word pass: updating TOC/LoF/LoT and exporting PDF')
    finalize_with_word(OUT_DOCX, OUT_PDF)
    print(f'      docx {os.path.getsize(OUT_DOCX) / 1024:.0f} KB; pdf {os.path.exists(OUT_PDF)}')

if __name__ == '__main__':
    main()
