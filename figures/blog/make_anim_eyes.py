"""Animated 'eyes' implant (Qwen2.5-0.5B reads handwritten MNIST digits), drawn to match eyes.py.

Mechanism shown (finetuned_phase/eyes_and_finetune/eyes.py, EyeMachine):
  * an image in the prompt is <|img|> + 196 pixel tokens (14x14, gray 0..9, one digit per token) + <|/img|>
  * at every image-END token the implant's fixed reader looks at the window of the 198 tokens ending there, and its
    CNN turns the 196 pixel values into a digit class. That class belongs to that token position; nothing is written
    into the model there (the implant adds zero outside a call).
  * at a call <|see|>k<|/see|> the implant takes k and picks the class of the k-th image end (in the code a running
    sum over positions, which is the same thing as one hard-attention lookup back to that image-end token) and writes it.
  * then <tool_call>9*7</tool_call> and the calculator implant writes 63.
Drawn simplifications: <|see|> shown as <see>; the implant's box tokens (<|see_out|>..<|/see_out|>, <|box_start|>..)
and the prose of the answer ("Image 1 shows ...") are left out; the question line is shortened.

Real example: eyes dev question #100 (eyes.py data, random.Random(6)): 'a*b' with MNIST test images 1964 (a 9) and
2490 (a 7) -> 9*7 = 63. Pixels: 2x2 average pool, round(x/255*9), as eyes.py mnist().

Usage:
  python3 make_anim_eyes.py          # all frames -> figs/anim/frames_eyes, then GIF + MP4 + poster
  python3 make_anim_eyes.py keys     # a few key frames as PNG (layout check)
"""
import sys, os, gzip, random, subprocess, shutil
import numpy as np
import make_anim as A
from make_anim import (T, chip, BG, BORDER, INK, MUTE, LAYER, OUTER, G_DARK, G_MID, G_LINE, G_LIGHT, G_PALE,
                       N_OFF, N_LV, SANS, MONO, MID, SMALL)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'figs', 'anim')
FR = os.path.join(OUTD, 'frames_eyes')
FPS = 12
PX_W = 1600
MNIST = os.environ.get('MNIST_DIR', 'mnist')  # dir with t10k-images-idx3-ubyte.gz / t10k-labels-idx1-ubyte.gz (LeCun MNIST)
IDX = [1964, 2490]                    # eyes dev question #100: image 1 * image 2
W, H = 1800, 1680
DY = 110                              # model block of make_anim.py, moved down by DY
TOKF = 34
CHW = TOKF * 0.602
PAD, GAP_W, GAP_G, TH = 11, 14, 4, 60

# ---------------- data ----------------
_X = np.frombuffer(gzip.open(os.path.join(MNIST, 't10k-images-idx3-ubyte.gz')).read(), np.uint8, offset=16).reshape(-1, 28, 28)
_Y = np.frombuffer(gzip.open(os.path.join(MNIST, 't10k-labels-idx1-ubyte.gz')).read(), np.uint8, offset=8)
Q14 = [np.round(_X[i].astype(np.float32).reshape(14, 2, 14, 2).mean((1, 3)) / 255 * 9).astype(int) for i in IDX]
LAB = [int(_Y[i]) for i in IDX]
assert LAB == [9, 7]
a, b = LAB
ANS = a * b                           # 63

# ---------------- model block geometry (make_anim.py, shifted by DY) ----------------
CX, jx = A.CX, A.jx
wy = A.wy + DY
OY0, OY1 = A.OY0 + DY, A.OY1 + DY
PY0, PY1 = A.PY0 + DY, A.PY1 + DY
LAYERS = [(A.yL28 + DY, 'layer 24'), (A.yL27 + DY, 'layer 23'), (A.yL1 + DY, 'layer 1')]
MX0, MX1 = A.MX0, A.MX1
CS = 56
EYE_X = MX0 + 112
CNN_X0 = EYE_X + 104
CALC_X, CALC_Y = EYE_X, wy + 196
EG = (MX0 + 34, wy - 66, CNN_X0 + 160, wy + 100)        # eyes implant group box x0, y0, x1, y1
X_IN = EG[2] + 70                                       # input bus (tokens -> implants)
BUS_X = MX0 + 40                                        # calculator joins the write line here
NTB = (CX - 150, 136, 300, 70)                          # next-token box

# ---------------- token strip ----------------
YB = OY1 + 32                                           # the token rail
TY0 = YB + 34                                           # top of line 0 (image line)
CELL, CP = 12, 13
BWID = 14 * CP + 12                                     # pixel-token block (196 tokens drawn 14x14)
CHIP_Y0 = TY0 + (BWID - TH) / 2
TAG_H = 38
TAG_Y = CHIP_Y0 + TH + 12
LINE_Y = [CHIP_Y0, TY0 + BWID + 64, TY0 + BWID + 64 + 120, TY0 + BWID + 64 + 200]
XS = 215

# (text, who, line, glue, kind)   who 0 prompt / 1 model / 2 implant ; kind 'tok' | ('blk', k)
TOKS = []
for k in range(2):
    TOKS += [('Image', 0, 0, 0, 'tok'), (str(k + 1), 0, 0, 0, 'tok'), (':', 0, 0, 1, 'tok'), ('<img>', 0, 0, 0, 'tok'),
             ('', 0, 0, 0, ('blk', k)), ('</img>', 0, 0, 0, 'tok')]
TOKS += [(w, 0, 1, 0, 'tok') for w in ['What', 'is', 'image', '1', '*', 'image', '2']] + [('?', 0, 1, 1, 'tok')]
for k in range(2):
    TOKS += [('<see>', 1, 2, 0, 'tok'), (str(k + 1), 1, 2, 1, 'tok'), ('</see>', 1, 2, 1, 'tok'), (str(LAB[k]), 2, 2, 0, 'tok')]
TOKS += [('<tool_call>', 1, 3, 0, 'tok'), (str(a), 1, 3, 1, 'tok'), ('*', 1, 3, 1, 'tok'), (str(b), 1, 3, 1, 'tok'),
         ('</tool_call>', 1, 3, 1, 'tok')]
TOKS += [(d, 2, 3, i > 0, 'tok') for i, d in enumerate(str(ANS))]
TOKS += [(w, 1, 3, 0, 'tok') for w in ['So', 'the', 'answer', 'is']] + [(d, 1, 3, i > 0, 'tok') for i, d in enumerate(str(ANS))] + [('.', 1, 3, 1, 'tok')]

TPOS = []                                               # (x, y, w)
_x = [XS] * 4
_first = [True] * 4
for i, (s, who, li, glue, kind) in enumerate(TOKS):
    w = BWID if kind != 'tok' else len(s) * CHW + 2 * PAD
    if not _first[li]:
        gap = GAP_G if glue else GAP_W
        if kind != 'tok' or (i > 0 and TOKS[i - 1][4] != 'tok'):
            gap = 12
        if li == 0 and s == 'Image':
            gap = 40
        _x[li] += gap
    _first[li] = False
    y = TY0 if kind != 'tok' else LINE_Y[li]
    TPOS.append((_x[li], y, w))
    _x[li] += w
assert max(_x) < W - 50, _x
BLK = [i for i, t in enumerate(TOKS) if t[4] != 'tok']
IMG_S = [i - 1 for i in BLK]                            # <img>
IMG_E = [i + 1 for i in BLK]                            # </img>
SEE_CE = [i for i, t in enumerate(TOKS) if t[0] == '</see>']
LINE1_END = max(TPOS[i][0] + TPOS[i][2] for i, t in enumerate(TOKS) if t[2] == 1)
NPROMPT = max(i for i, t in enumerate(TOKS) if t[2] <= 1) + 1


def tok_cx(i):
    x, y, w = TPOS[i]
    return x + w / 2


def tag_box(k):
    s = f'image {k+1} → {LAB[k]}'
    w = 164
    cx = tok_cx(IMG_E[k])
    return cx - w / 2, TAG_Y, w, s


def ink(v):
    if v == 0:
        return '#eef0f2'
    t = (v - 1) / 8
    c0, c1 = np.array([0xb3, 0xb9, 0xc0]), np.array([0x11, 0x11, 0x11])
    return '#%02x%02x%02x' % tuple((c0 + (c1 - c0) * t).astype(int))


# ---------------- glyphs ----------------
def eye_chip(o, cx, cy, S, lit=False, op=1.0):
    h, P, st = S / 2, S * 0.2, S * 0.3
    o.append(f'<g opacity="{op}">')
    if lit:
        o.append(f'<rect x="{cx-h-20}" y="{cy-h-20}" width="{S+40}" height="{S+40}" rx="22" fill="{G_MID}" opacity="0.45" filter="url(#glow)"/>')
    pins = ''.join(f'M{cx+k*st} {cy-h}v-{P}M{cx+k*st} {cy+h}v{P}M{cx-h} {cy+k*st}h-{P}M{cx+h} {cy+k*st}h{P}' for k in (-1, 0, 1))
    o.append(f'<path d="{pins}" stroke="{G_DARK if lit else G_MID}" stroke-width="4" stroke-linecap="round" fill="none"/>')
    o.append(f'<rect x="{cx-h}" y="{cy-h}" width="{S}" height="{S}" rx="{S*0.13}" fill="{G_MID if lit else BG}" stroke="{G_DARK if lit else G_MID}" stroke-width="4"/>')
    d, e = S * 0.33, S * 0.22
    fg = BG if lit else G_MID
    o.append(f'<path d="M{cx-d} {cy}Q{cx} {cy-2*e} {cx+d} {cy}Q{cx} {cy+2*e} {cx-d} {cy}Z" fill="none" stroke="{fg}" stroke-width="3.5" stroke-linejoin="round"/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="{S*0.1}" fill="{fg}"/>')
    o.append('</g>')


def cnn_glyph(o, x0, cy, lit, op):
    hs = [72, 56, 40, 26, 14]
    x = x0
    o.append(f'<g opacity="{op}">')
    if lit:
        o.append(f'<rect x="{x0-14}" y="{cy-50}" width="{len(hs)*28+20}" height="100" rx="18" fill="{G_MID}" opacity="0.35" filter="url(#glow)"/>')
    for hh in hs:
        o.append(f'<rect x="{x}" y="{cy-hh/2}" width="16" height="{hh}" rx="3" fill="{G_MID if lit else BG}" stroke="{G_DARK if lit else G_MID}" stroke-width="3"/>')
        x += 28
    o.append('</g>')
    return x - 12


def along(pts, t):
    segs = [((x0, y0), (x1, y1), abs(x1 - x0) + abs(y1 - y0)) for (x0, y0), (x1, y1) in zip(pts, pts[1:])]
    L = sum(s[2] for s in segs) * max(0, min(1, t))
    for (x0, y0), (x1, y1), l in segs:
        if L <= l:
            f = L / l if l else 0
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        L -= l
    return pts[-1]


def path_up_eyes(i):         # token i -> rail -> bus -> eyes implant
    x = tok_cx(i)
    return [(x, TPOS[i][1] - 4 if TOKS[i][2] == 0 else LINE_Y[0] - 4), (x, YB), (X_IN, YB), (X_IN, wy), (EG[2], wy)]


def path_up_calc(i):
    x = tok_cx(i)
    return [(x, YB), (X_IN, YB), (X_IN, CALC_Y), (CALC_X + CS / 2 + 16, CALC_Y)]


def path_down(k):            # eyes implant -> bus -> rail -> image k end token
    x = tok_cx(IMG_E[k])
    return [(EG[2], wy), (X_IN, wy), (X_IN, YB), (x, YB), (x, CHIP_Y0)]


def path_write(which):
    if which == 'eyes':
        return [(EYE_X - CS / 2 - 14, wy), (jx + 24, wy)]
    return [(CALC_X - CS / 2 - 14, CALC_Y), (BUS_X, CALC_Y), (BUS_X, wy), (jx + 24, wy)]


def arc_pts(src, k):
    """hard-attention arrow: from the current token src (</see>) back to image k's end token (its tag).
    Routed through the gap between the question line and the model line, then up to the tag."""
    tx, ty, tw_, _ = tag_box(k)
    x0 = tok_cx(src)
    yg = (LINE_Y[1] + TH + LINE_Y[2]) / 2
    xu = min(max(tx + tw_ / 2, LINE1_END + 34), tx + tw_ - 22)
    return [(x0, LINE_Y[2] - 5), (x0, yg), (xu, yg), (xu, ty + TAG_H + 7)]


def poly_part(pts, p):
    segs = [abs(x1 - x0) + abs(y1 - y0) for (x0, y0), (x1, y1) in zip(pts, pts[1:])]
    L = sum(segs) * p
    out = [pts[0]]
    for (q0, q1), l in zip(zip(pts, pts[1:]), segs):
        if L >= l:
            out.append(q1); L -= l
        else:
            f = L / l if l else 0
            out.append((q0[0] + (q1[0] - q0[0]) * f, q0[1] + (q1[1] - q0[1]) * f)); break
    return out


def bez(P, t):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = P
    u = 1 - t
    return (u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3,
            u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3)


def bez_split(P, t):
    """first part [0, t] of a cubic bezier (de Casteljau)"""
    lerp = lambda p, q: (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
    p01, p12, p23 = lerp(P[0], P[1]), lerp(P[1], P[2]), lerp(P[2], P[3])
    p012, p123 = lerp(p01, p12), lerp(p12, p23)
    return [P[0], p01, p012, lerp(p012, p123)]


def dot(o, x, y, label=None):
    o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{24 if label else 20}" fill="{G_MID}" opacity="0.45" filter="url(#glow)"/>')
    if label:
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="19" fill="{G_MID}" stroke="{BG}" stroke-width="2.5"/>')
        T(o, x, y + 9, label, 26, BG, 700, 'middle', MONO)
    else:
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{G_MID}" stroke="{BG}" stroke-width="2.5"/>')


# ---------------- render ----------------
def render(st):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<defs><marker id="aI" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker>'
         f'<marker id="aG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="{G_MID}"/></marker>'
         f'<marker id="aGD" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.2" markerHeight="4.2" orient="auto"><path d="M0 0L10 5L0 10z" fill="{G_DARK}"/></marker>'
         f'<filter id="glow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="12"/></filter></defs>',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    T(o, 60, 72, 'Giving Qwen eyes', 46, INK, 700)
    T(o, 60, 116, 'the CNN reads each image as it streams by; later, the implant looks back to it', SMALL, MUTE)

    # next-token box
    nt = st['nextok']
    if nt is None:
        o.append(f'<rect x="{NTB[0]}" y="{NTB[1]}" width="{NTB[2]}" height="{NTB[3]}" rx="10" fill="{BG}" stroke="{BORDER}" stroke-width="2.5"/>')
        T(o, CX, NTB[1] + 46, 'next token', SMALL, MUTE, anchor='middle')
    else:
        s, g = nt
        bw = max(NTB[2], len(s) * (TOKF + 4) * 0.602 + 48)
        o.append(f'<rect x="{CX-bw/2}" y="{NTB[1]}" width="{bw}" height="{NTB[3]}" rx="10" fill="{G_LIGHT if g else BG}" stroke="{G_MID if g else INK}" stroke-width="{3 if g else 2.5}"/>')
        T(o, CX, NTB[1] + 48, s, TOKF + 4, G_DARK if g else INK, 700 if g else 400, 'middle', MONO)

    # ---- ONE MODEL ----
    o.append(f'<rect x="{A.OX0}" y="{OY0}" width="{A.OX1-A.OX0}" height="{OY1-OY0}" rx="14" fill="{OUTER}" stroke="{INK}" stroke-width="2.5"/>')
    T(o, A.OX0 + 36, OY0 + 50, 'ONE MODEL', SMALL, MUTE, 700, ls=3)
    o.append(f'<rect x="{A.LX0}" y="{PY0}" width="{A.LX1-A.LX0}" height="{PY1-PY0}" rx="12" fill="{BG}" stroke="{INK}" stroke-width="3"/>')
    o.append(f'<path d="M{CX} {PY0}V{NTB[1]+NTB[3]+12}" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
    T(o, A.LX0 + 32, PY0 + 60, 'LANGUAGE MODEL', MID, INK, 700, ls=3)
    T(o, A.LX0 + 32, PY0 + 102, 'Qwen2.5-0.5B · text only', SMALL, MUTE)
    yHead = A.yHead + DY
    o.append(f'<rect x="{A.BX0}" y="{yHead}" width="{A.BX1-A.BX0}" height="{A.HB}" rx="8" fill="{LAYER}" stroke="{INK}" stroke-width="2"/>')
    T(o, CX, yHead + 36, 'output head', SMALL, MUTE, anchor='middle')
    for li, (y, lab) in enumerate(LAYERS):
        o.append(f'<rect x="{A.BX0}" y="{y}" width="{A.BX1-A.BX0}" height="{A.LB}" rx="8" fill="{LAYER}" stroke="{INK}" stroke-width="2"/>')
        T(o, A.BX0 + 20, y + A.LB / 2 + 11, lab, SMALL, MUTE)
        lv = st['neur'][li]
        y0 = y + (A.LB - A.NROWS * A.NQ - (A.NP - A.NQ)) / 2
        for r in range(A.NROWS):
            for c in range(A.NCOLS):
                v = lv[r * A.NCOLS + c]
                o.append(f'<rect x="{A.NX0 + c*A.NP}" y="{y0 + r*A.NP:.1f}" width="{A.NQ}" height="{A.NQ}" rx="2.5" fill="{N_OFF if v < 0 else N_LV[v]}"/>')
    for k in range(3):
        o.append(f'<circle cx="{CX}" cy="{A.yL27 + DY + A.LB + 26 + k*16}" r="4" fill="{MUTE}"/>')
    o.append(f'<path d="M{jx} {LAYERS[0][0]}V{yHead+A.HB}" stroke="{INK}" stroke-width="2"/>')
    if st['plus_flash']:
        o.append(f'<circle cx="{jx}" cy="{wy}" r="30" fill="{G_MID}" opacity="0.4" filter="url(#glow)"/>')
        o.append(f'<circle cx="{jx}" cy="{wy}" r="19" fill="{G_MID}" stroke="{G_DARK}" stroke-width="3.5"/>')
        o.append(f'<path d="M{jx-10} {wy}h20M{jx} {wy-10}v20" stroke="{BG}" stroke-width="3.5" stroke-linecap="round"/>')
    else:
        o.append(f'<circle cx="{jx}" cy="{wy}" r="17" fill="{BG}" stroke="{G_MID}" stroke-width="3.5"/>')
        o.append(f'<path d="M{jx-9} {wy}h18M{jx} {wy-9}v18" stroke="{G_MID}" stroke-width="3.5" stroke-linecap="round"/>')

    # ---- implants panel ----
    o.append(f'<rect x="{MX0}" y="{PY0}" width="{MX1-MX0}" height="{PY1-PY0}" rx="10" fill="{G_PALE}" stroke="{G_LINE}" stroke-width="3"/>')
    T(o, MX0 + 28, PY0 + 60, 'IMPLANTS', MID, G_DARK, 700, ls=3)
    T(o, MX0 + 28, PY0 + 104, st['status'] or 'silent: adds exactly zero', SMALL, G_DARK,
      700 if st['status'] else 400, op=1 if st['status'] else 0.75)
    wact = st['wact']
    o.append(f'<path d="M{MX0} {wy}H{jx+24}" stroke="{G_MID}" stroke-width="4" marker-end="url(#aG)" opacity="{1 if wact else 0.5}"/>')
    T(o, (A.LX1 + MX0) / 2, wy - 20, 'write', SMALL, G_DARK, 600, 'middle', op=1 if wact else 0.6)
    o.append(f'<path d="M{EYE_X-CS/2-14} {wy}H{MX0}" stroke="{G_MID}" stroke-width="4" opacity="{1 if wact == "eyes" else 0.4}"/>')
    o.append(f'<path d="M{CALC_X-CS/2-14} {CALC_Y}H{BUS_X}V{wy}" fill="none" stroke="{G_MID}" stroke-width="4" opacity="{1 if wact == "calc" else 0.4}"/>')

    # eyes implant: chip + CNN, one group
    e_on, cnn_on = st['e_on'], st['cnn']
    o.append(f'<rect x="{EG[0]}" y="{EG[1]}" width="{EG[2]-EG[0]}" height="{EG[3]-EG[1]}" rx="14" fill="{BG}" '
             f'stroke="{G_MID if e_on else G_LINE}" stroke-width="{3 if e_on else 2.5}" opacity="{1 if e_on else 0.8}"/>')
    eye_chip(o, EYE_X, wy, CS, lit=e_on, op=1 if e_on else 0.55)
    T(o, EYE_X, wy + 76, 'eyes', SMALL, G_DARK, 700 if e_on else 600, 'middle', op=1 if e_on else 0.6)
    xe = cnn_glyph(o, CNN_X0, wy, cnn_on, 1 if (cnn_on or e_on) else 0.5)
    o.append(f'<path d="M{CNN_X0-6} {wy}H{EYE_X+CS/2+16}" stroke="{G_MID}" stroke-width="3" marker-end="url(#aG)" opacity="{0.9 if cnn_on else 0.4}"/>')
    T(o, (CNN_X0 + xe) / 2, wy + 76, 'CNN', SMALL, G_DARK, 700 if cnn_on else 600, 'middle', op=1 if cnn_on else 0.6)

    # calculator
    c_on = st['c_on']
    chip(o, CALC_X, CALC_Y, CS, lit=c_on, op=1 if c_on else 0.55)
    T(o, CALC_X, CALC_Y + 72, 'calculator', SMALL, G_DARK, 700 if c_on else 600, 'middle', op=1 if c_on else 0.6)
    if c_on and st['calc_txt']:
        T(o, CALC_X + 70, CALC_Y - 22, st['calc_txt'], TOKF, G_DARK, 700, font=MONO)

    # ---- token rail + input bus ----
    xr1 = TPOS[IMG_E[1]][0] + TPOS[IMG_E[1]][2]
    o.append(f'<path d="M{XS} {YB}H{xr1}" stroke="{BORDER}" stroke-width="3"/>')
    o.append(f'<path d="M{CX} {YB}V{PY1+8}" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
    bus_on = st['bus']
    for which, pts in (('eyes', [(X_IN, YB), (X_IN, wy), (EG[2] + 10, wy)]), ('calc', [(X_IN, CALC_Y), (CALC_X + CS / 2 + 16, CALC_Y)])):
        on = bus_on == which
        d = 'M' + 'L'.join(f'{x:.1f} {y:.1f}' for x, y in pts)
        if on:
            o.append(f'<path d="{d}" stroke="{G_MID}" stroke-width="12" fill="none" opacity="0.3" filter="url(#glow)"/>')
        o.append(f'<path d="{d}" fill="none" stroke="{G_MID}" stroke-width="{4 if on else 3}" marker-end="url(#aG)" opacity="{1 if on else 0.45}"/>')
    o.append(f'<circle cx="{X_IN}" cy="{YB}" r="5" fill="{G_MID}" opacity="{1 if bus_on else 0.5}"/>')

    # ---- tokens ----
    T(o, 60, CHIP_Y0 + 41, 'prompt', SMALL, MUTE, 600)
    if st['n'] > NPROMPT:
        T(o, 60, LINE_Y[2] + 41, 'model', SMALL, MUTE, 600)
    win = st['window']
    if win is not None:
        x0 = TPOS[IMG_S[win]][0]
        x1 = TPOS[IMG_E[win]][0] + TPOS[IMG_E[win]][2]
        o.append(f'<rect x="{x0-10}" y="{TY0-10}" width="{x1-x0+20}" height="{BWID+20}" rx="14" fill="{G_PALE}" stroke="{G_MID}" stroke-width="3.5"/>')
    for i in range(st['n']):
        s, who, li, _, kind = TOKS[i]
        x, y, w = TPOS[i]
        if kind != 'tok':
            k = kind[1]
            rows = st['rows'][k]
            hot = win == k
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{w}" rx="9" fill="{BG}" stroke="{G_MID if hot else INK}" stroke-width="{3 if hot else 2}"/>')
            q = Q14[k]
            for r in range(rows):
                for c in range(14):
                    o.append(f'<rect x="{x + 6 + c*CP}" y="{y + 6 + r*CP}" width="{CELL}" height="{CELL}" rx="1.5" fill="{ink(q[r, c])}"/>')
            if st['ticker'] is not None and st['ticker'][0] == k and rows < 14:
                r = st['ticker'][1]
                o.append(f'<rect x="{x + 3}" y="{y + 3 + r*CP}" width="{14*CP + 5}" height="{CP + 5}" rx="3" fill="none" stroke="{INK}" stroke-width="2"/>')
                T(o, x + w / 2, y + w + 36, ' '.join(str(v) for v in q[r]), 22, INK, 600, 'middle', MONO)
            elif rows == 14:
                T(o, x + w / 2, y + w + 34, '196 pixel tokens', 23, MUTE, 600, 'middle')
            continue
        g = who == 2
        fresh = i == st['n'] - 1 and st['new'] == 0
        q_ = st['query'] == i
        if g:
            fill, stroke, col, sw = G_LIGHT, G_MID, G_DARK, 3
        elif q_:
            fill, stroke, col, sw = G_PALE, G_MID, G_DARK, 3.5
        else:
            fill, stroke, col, sw = BG, INK, INK, (3 if fresh else 2)
        if fresh and g:
            o.append(f'<rect x="{x-8:.1f}" y="{y-8}" width="{w+16:.1f}" height="{TH+16}" rx="14" fill="{G_MID}" opacity="0.4" filter="url(#glow)"/>')
        o.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{TH}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        T(o, x + w / 2, y + 41, s, TOKF, col, 700 if (g or q_) else 400, 'middle', MONO)

    # tags pinned on the image-end tokens
    for k in range(st['tags']):
        tx, ty, tw_, s = tag_box(k)
        glow = st['tag_glow'] == k
        dim = st['tag_dim'] == k
        cx = tx + tw_ / 2
        op = 0.45 if dim else 1
        if glow:
            o.append(f'<rect x="{tx-10}" y="{ty-8}" width="{tw_+20}" height="{TAG_H+16}" rx="20" fill="{G_MID}" opacity="0.45" filter="url(#glow)"/>')
        o.append(f'<g opacity="{op}"><path d="M{cx} {CHIP_Y0+TH}V{ty}" stroke="{G_MID}" stroke-width="3"/>')
        o.append(f'<rect x="{tx}" y="{ty}" width="{tw_}" height="{TAG_H}" rx="{TAG_H/2}" fill="{G_MID if glow else G_LIGHT}" stroke="{G_DARK if glow else G_MID}" stroke-width="2.5"/>')
        T(o, cx, ty + 27, s, 23, BG if glow else G_DARK, 700, 'middle')
        o.append('</g>')

    # hard-attention arrow
    if st['arc'] is not None:
        src, k, p = st['arc']
        P = arc_pts(src, k)
        Q = poly_part(P, p)
        d = 'M' + 'L'.join(f'{x:.1f} {y:.1f}' for x, y in Q)
        o.append(f'<path d="{d}" fill="none" stroke="{BG}" stroke-width="13" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>')
        o.append(f'<path d="{d}" fill="none" stroke="{G_DARK}" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"'
                 f'{" marker-end=" + chr(34) + "url(#aGD)" + chr(34) if p >= 0.97 else ""}/>')
        if st['arc_lab'] and p >= 1:
            mx, my = P[2]
            if mx + 175 < W - 10:
                T(o, mx + 16, my + 9, 'looks back', 26, G_DARK, 700)
            else:
                T(o, mx - 16, my - 14, 'looks back', 26, G_DARK, 700, 'end')

    for (x, y, lab) in st['dots']:
        dot(o, x, y, lab)
    T(o, W / 2, H - 30, st['cap'], SMALL, MUTE, anchor='middle')
    o.append('</svg>')
    return '\n'.join(o)


# ---------------- timeline ----------------
def ease(t):
    return t * t * (3 - 2 * t)


def build():
    rng = random.Random(7)
    NN = A.NROWS * A.NCOLS
    frames = []
    neur = [[-1] * NN for _ in LAYERS]

    def fire():
        for li in range(len(LAYERS)):
            neur[li] = [(rng.choice([0, 1, 1, 2, 2]) if rng.random() < 0.32 else -1) for _ in range(NN)]

    def decay():
        for li in range(len(LAYERS)):
            nv = []
            for v in neur[li]:
                if v >= 0:
                    v = v - 1 if rng.random() < 0.45 else v
                elif rng.random() < 0.025:
                    v = 0
                nv.append(v)
            neur[li] = nv

    st = dict(n=0, new=9, nextok=None, rows=[0, 0], ticker=None, window=None, tags=0, tag_glow=None, tag_dim=None,
              arc=None, arc_lab=False, query=None, dots=[], bus=None, e_on=False, cnn=False, c_on=False, calc_txt=None,
              wact=None, plus_flash=False, status=None, cap='')

    def snap(key=None, tick=True):
        if tick:
            decay()
        st['new'] += 1
        d = dict(st)
        d['rows'] = list(st['rows']); d['dots'] = list(st['dots']); d['neur'] = [list(x) for x in neur]; d['key'] = key
        frames.append(d)

    def add_tok(i, nf=1, **kw):
        fire()
        st.update(n=i + 1, new=0, **kw)
        snap(tick=False)
        for _ in range(nf - 1):
            snap()

    for _ in range(3):
        snap()
    # ---- 1) prefill: two images stream in as pixel tokens ----
    for k in range(2):
        first = k == 0
        st.update(cap=f'the prompt streams in: image {k+1} is 196 pixel tokens (14×14, gray level 0–9)')
        for i in range(IMG_S[k] - 2, IMG_S[k] + 1):
            add_tok(i)
        st['n'] = BLK[k] + 1
        for r in range(1, 15):
            st['rows'][k] = r
            st['ticker'] = (k, r - 1)
            fire(); st['new'] = 0
            snap(tick=False)
        st['ticker'] = None
        add_tok(IMG_E[k], 2)
        # image end: the reader's window over the 198 tokens -> CNN
        st.update(window=k, bus='eyes', e_on=True, status=f'image end: run the CNN',
                  cap=f'at </img>, the implant’s CNN reads the 196 pixel tokens before it')
        P = path_up_eyes(IMG_E[k])
        for p in ((0.2, 0.45, 0.7, 0.95) if first else (0.35, 0.7, 1.0)):
            st['dots'] = [along(P, p) + (None,)]
            snap()
        st.update(dots=[], cnn=True, status=f'CNN: image {k+1} → {LAB[k]}')
        for _ in range(4 if first else 3):
            snap('k_cnn' if first else None)
        # the result stays at that token position
        st.update(cap=f'its answer stays at that token: image {k+1} → {LAB[k]}  (nothing is written into Qwen)')
        P = path_down(k)
        for p in ((0.15, 0.4, 0.65, 0.9) if first else (0.25, 0.6, 0.95)):
            st['dots'] = [along(P, p) + (str(LAB[k]),)]
            snap()
        st.update(dots=[], tags=k + 1, tag_glow=k, cnn=False, bus=None, status=None)
        for _ in range(3):
            snap()
        st.update(window=None, e_on=False, tag_glow=None)
        for _ in range(6 if first else 5):
            snap('k_tag' if first else None)
    # ---- 2) the question ----
    st.update(cap='then the question; the implant stays silent')
    for i in range(IMG_E[1] + 1, NPROMPT):
        add_tok(i)
    for _ in range(4):
        snap()
    # ---- 3) <see>k</see>: hard attention back to image k's end token ----
    for k in range(2):
        first = k == 0
        i0 = SEE_CE[k] - 2
        st.update(cap=f'the model asks to see image {k+1}')
        for i in (i0, i0 + 1, i0 + 2):
            add_tok(i, 2, nextok=(TOKS[i][0], False))
        st.update(bus='eyes', e_on=True, query=i0 + 1, status=f'read <see>{k+1}</see>',
                  cap=f'the implant reads the index {k+1} …')
        P = path_up_eyes(i0 + 1)
        for p in ((0.25, 0.55, 0.85) if not first else (0.4, 0.9)):
            st['dots'] = [along(P, p) + (None,)]
            snap()
        st.update(dots=[], bus=None, status=f'look back to image {k+1}',
                  cap=f'… and looks back along the prompt to image {k+1}’s end token' if not first else
                  f'… and looks back to image {k+1}’s end token')
        steps = (0.2, 0.4, 0.6, 0.8, 1.0) if not first else (0.35, 0.7, 1.0)
        for p in steps:
            st['arc'] = (SEE_CE[k], k, p)
            snap()
        st.update(tag_glow=k, tag_dim=1 - k, arc_lab=True)
        for _ in range(7 if not first else 3):
            snap('poster' if not first else None)
        # picked value travels to the implant, which writes it
        st.update(cap=f'it picks up the {LAB[k]} pinned there and writes it', status=f'write {LAB[k]}', bus='eyes')
        tx = tok_cx(IMG_E[k])
        P = [(tx, CHIP_Y0), (tx, YB), (X_IN, YB), (X_IN, wy), (EG[2], wy)]
        for p in ((0.2, 0.5, 0.8, 1.0) if not first else (0.4, 0.9)):
            st['dots'] = [along(P, p) + (str(LAB[k]),)]
            snap()
        st.update(bus=None, wact='eyes')
        P = path_write('eyes')
        for p in ((0.15, 0.5, 0.85) if not first else (0.3, 0.8)):
            st['dots'] = [along(P, p) + (str(LAB[k]),)]
            snap()
        st.update(dots=[], plus_flash=True)
        add_tok(SEE_CE[k] + 1, 1, nextok=(str(LAB[k]), True))
        st.update(plus_flash=False)
        for _ in range(7 if not first else 3):
            snap('k_wrote' if not first else None)
        st.update(arc=None, arc_lab=False, tag_glow=None, tag_dim=None, query=None, e_on=False, wact=None, status=None)
    # ---- 4) the calculator ----
    ic = SEE_CE[1] + 2
    st.update(cap=f'then the calculator implant does the math')
    for i in range(ic, ic + 5):
        add_tok(i, 2 if i in (ic, ic + 4) else 1, nextok=(TOKS[i][0], False))
    st.update(bus='calc', c_on=True, status=f'read {a}*{b}')
    P = path_up_calc(ic + 2)
    for p in (0.4, 0.9):
        st['dots'] = [along(P, p) + (None,)]
        snap()
    st.update(dots=[], bus=None, calc_txt=f'{a}*{b} = {ANS}', status=f'compute {ANS}', cap=f'{a}*{b} = {ANS}, written one digit per step')
    snap('k_calc'); snap()
    st['wact'] = 'calc'
    for j, d in enumerate(str(ANS)):
        st['status'] = f'write {d}'
        P = path_write('calc')
        for p in (0.25, 0.7):
            st['dots'] = [along(P, p) + (d,)]
            snap()
        st.update(dots=[], plus_flash=True)
        add_tok(ic + 5 + j, 1, nextok=(d, True))
        st['plus_flash'] = False
        snap()
    st.update(c_on=False, wact=None, calc_txt=None, status=None)
    for i in range(ic + 5 + len(str(ANS)), len(TOKS)):
        add_tok(i, 1, nextok=(TOKS[i][0], False))
    st.update(nextok=None, cap='Qwen reads text only. The implant reads the pixels.')
    for _ in range(14):
        snap('k_end')
    return frames


def main():
    keys_only = len(sys.argv) > 1 and sys.argv[1] == 'keys'
    os.makedirs(FR, exist_ok=True)
    frames = build()
    print(len(frames), 'frames', f'{len(frames)/FPS:.1f}s', 'labels', LAB, 'answer', ANS)
    if keys_only:
        seen = set()
        for i, f in enumerate(frames):
            k = f['key']
            if k and k not in seen:
                seen.add(k)
                svg = os.path.join(FR, k + '.svg')
                open(svg, 'w').write(render(f))
                subprocess.run(['rsvg-convert', '-w', str(PX_W), '-b', 'white', svg, '-o', os.path.join(FR, k + '.png')], check=True)
                print(k, i)
        for name, i in (('k_stream', 12), ('k_arc', next(j for j, f in enumerate(frames) if f['key'] == 'poster') - 6)):
            svg = os.path.join(FR, name + '.svg')
            open(svg, 'w').write(render(frames[i]))
            subprocess.run(['rsvg-convert', '-w', str(PX_W), '-b', 'white', svg, '-o', os.path.join(FR, name + '.png')], check=True)
            print(name, i)
        return
    for f in os.listdir(FR):
        os.remove(os.path.join(FR, f))
    svg = os.path.join(FR, 'tmp.svg')
    done_poster = False
    for i, fr in enumerate(frames):
        open(svg, 'w').write(render(fr))
        subprocess.run(['rsvg-convert', '-w', str(PX_W), '-b', 'white', svg, '-o', os.path.join(FR, f'f{i:04d}.png')], check=True)
        if fr['key'] == 'poster' and not done_poster:
            subprocess.run(['rsvg-convert', '-w', '2600', '-b', 'white', svg, '-o', os.path.join(OUTD, 'eyes_poster.png')], check=True)
            done_poster = True
    os.remove(svg)
    gif = os.path.join(OUTD, 'eyes.gif')
    pat = os.path.join(FR, 'f%04d.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', pat, '-vf',
                    'split[a][b];[a]palettegen=max_colors=96:stats_mode=full:reserve_transparent=0[p];'
                    '[b][p]paletteuse=dither=none:diff_mode=rectangle', '-loop', '0', gif], check=True)
    if shutil.which('gifsicle'):
        subprocess.run(['gifsicle', '-O3', '-b', gif], check=True)
    mp4 = os.path.join(OUTD, 'eyes.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-stream_loop', '1', '-framerate', str(FPS), '-i', pat,
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '18', '-preset', 'slow', '-movflags', '+faststart', mp4], check=True)
    for p in (gif, mp4):
        print(p, f'{os.path.getsize(p)/1e6:.2f} MB')


if __name__ == '__main__':
    main()
