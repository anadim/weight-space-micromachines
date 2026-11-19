"""Animated mistake-detector implant (GSM8K test, frozen Qwen3-0.6B-Base), drawn to match make_anim.py / make_anim_eyes.py.

Mechanism shown (micromachines/rethink_circuit_v2.py, intervene='hybrid', detector results/rethink_v2/detector_hybrid.pt):
  * the "read point" is the first position where the host's own next token is '####' (about to commit the final answer).
  * there the detector (a small ReLU MLP over the host's hidden states at layers 12,14,...,24) computes a gate score
    sB = logit P(right after a rethink) - logit P(right now). If sB > thB = 1.51 the implant writes
    "Hmm, wait. Let me rethink this.\n" in place of '####' and goes silent; the frozen model continues.
    (The same MLP has a third head for a rethink AFTER the answer; it is not the route used in these two examples.)
  * the meter shows sB (the gate score: "will a rethink help?"); the small line under it is the MLP's P(right now) head.

Real data: figs/anim/detector_trace.json (default) or figs/anim/detector_v2_trace.json (argument v2)
  beat 1: GSM8K test problem 1160 (plain arm, hybrid run): fire_b=1, phrase at generated token 29, first answer wrong,
          final answer 1125 correct. Without the implant (none arm) the model commits '#### 1925'.
  beat 2: GSM8K test problem 462: fire_b=0, the implant stays silent, '#### 71' correct.
  Texts / token pieces from <notes>/release_full_outputs/rethink_v2/final_eval/test_plain_{hybrid,none}_s*.json.
  Scores (sB, P(right now)) were NOT logged by the runs; recomputed once on A100 node with the fitted detector on the same
  prefixes (figs/anim/anim_scores_job.py). 1160: sB 1.96 > 1.51 (fires, as logged); 462: sB -3.26 (silent, as logged).
Drawn simplifications: pure-space tokens (Qwen writes ' ' before a number) are drawn as a gap, newlines as line breaks;
the question is shown as plain text instead of token chips; the few-shot prefix of the prompt is not shown.

Usage:
  python3 make_anim_detector.py          # all frames -> figs/anim/frames_detector, then GIF + MP4 + poster
  python3 make_anim_detector.py v2       # v2: beat 1 = GSM8K test 755, a REASONING error (misread quantity), not an
                                         #     arithmetic slip -> detector_v2.gif/.mp4/_poster.png, frames_detector_v2
  v2 data (detector_v2_trace.json): problem 755, fire_b=1, phrase at generated token 36; first attempt
  'On Saturday, Joey played 2 * 1 = 2 matches. In total ... 2 + 1 + 2 = 5' (every step right, but it doubles Friday,
  not Monday); none arm commits '#### 5'; after the rethink '2 * 2 = 4', '#### 7' (gold 7). sB 2.05 > 1.51 (recomputed,
  figs/anim/anim_scores_job_v2.py), P(right now) 30.1%.
  python3 make_anim_detector.py keys     # a few key frames as PNG (layout check)
"""
import sys, os, json, random, subprocess, shutil
from PIL import ImageFont
import make_anim as A
from make_anim import (T, chip, esc, BG, BORDER, INK, MUTE, LAYER, OUTER, G_DARK, G_MID, G_LINE, G_LIGHT, G_PALE,
                       N_OFF, N_LV, SANS, MONO, MID, SMALL)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'figs', 'anim')
FPS = 12
PX_W = 1600
RED = '#b4232a'
W = 1800
DY = 110                                      # model block of make_anim.py, moved down by DY
TOKF = 30
CHW = TOKF * 0.602
PAD, GAP_W, GAP_G, TH, LP = 10, 13, 3, 52, 66
FD = os.path.join(HERE, 'fonts')
_F = {}


def tw(s, size, weight=400):
    k = (size, weight)
    if k not in _F:
        _F[k] = ImageFont.truetype(os.path.join(FD, 'AvenirNextBold.ttf' if weight >= 700 else
                                                'AvenirNextDemiBold.ttf' if weight >= 600 else 'AvenirNextRegular.ttf'), size)
    return _F[k].getlength(s)


def wrap(s, size, wmax, weight=400):
    lines, cur = [], ''
    for wd in s.split(' '):
        t = (cur + ' ' + wd) if cur else wd
        if tw(t, size, weight) > wmax and cur:
            lines.append(cur); cur = wd
        else:
            cur = t
    return lines + [cur]


# ---------------- model block geometry (make_anim.py shifted by DY; middle layer bar relabelled) ----------------
CX, jx = A.CX, A.jx
OY0, OY1 = A.OY0 + DY, A.OY1 + DY
PY0, PY1 = A.PY0 + DY, A.PY1 + DY
yHead = A.yHead + DY
wy = A.wy + DY
LB = A.LB
yTop = A.yL28 + DY
yMid = yTop + LB + 50
yBot = yMid + LB + 50
NX0 = A.BX0 + 214
NCOLS = int((A.BX1 - 18 - NX0 + (A.NP - A.NQ)) // A.NP)
NN = A.NROWS * NCOLS
MX0, MX1 = A.MX0, A.MX1
NTB = (CX - 150, 136, 300, 70)
CS = 56
WR_X = MX0 + 130                              # writer chip (on the write line)
DET_X, DET_Y = MX0 + 130, yMid + LB / 2       # detector glyph, level with the read layer bar
MTR_X0, MTR_X1 = MX0 + 262, MX1 - 40          # meter
MTR_Y = DET_Y
XS = 230                                      # token text start
XMAX = W - 60
TOK_Y0 = OY1 + 46


def defs():
    return (f'<defs><marker id="aI" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker>'
            f'<marker id="aG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="{G_MID}"/></marker>'
            f'<filter id="glow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="12"/></filter></defs>')


def mlp_glyph(o, cx, cy, lit, op):
    hs = [64, 80, 80, 56, 30]
    x0 = cx - (len(hs) * 26 - 10) / 2
    o.append(f'<g opacity="{op}">')
    if lit:
        o.append(f'<rect x="{x0-16}" y="{cy-52}" width="{len(hs)*26+22}" height="104" rx="18" fill="{G_MID}" opacity="0.35" filter="url(#glow)"/>')
    x = x0
    for hh in hs:
        o.append(f'<rect x="{x}" y="{cy-hh/2}" width="16" height="{hh}" rx="3" fill="{G_MID if lit else BG}" stroke="{G_DARK if lit else G_MID}" stroke-width="3"/>')
        x += 26
    o.append('</g>')
    return x0, x - 10


def neuron_glyph(o, cx, cy, lit, op):
    o.append(f'<g opacity="{op}">')
    if lit:
        o.append(f'<circle cx="{cx}" cy="{cy}" r="44" fill="{G_MID}" opacity="0.4" filter="url(#glow)"/>')
    for k in (-1, 0, 1):
        o.append(f'<path d="M{cx-58} {cy+k*18}L{cx-26} {cy+k*8}" stroke="{G_DARK if lit else G_MID}" stroke-width="3" stroke-linecap="round"/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="27" fill="{G_MID if lit else BG}" stroke="{G_DARK if lit else G_MID}" stroke-width="4"/>')
    o.append(f'<path d="M{cx-12} {cy+9}H{cx-2}L{cx+4} {cy-9}H{cx+13}" stroke="{BG if lit else G_MID}" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    o.append('</g>')
    return cx - 58, cx + 27


def dot(o, x, y):
    o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="20" fill="{G_MID}" opacity="0.45" filter="url(#glow)"/>')
    o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{G_MID}" stroke="{BG}" stroke-width="2.5"/>')


def mark(o, x, y, ok, r=20):
    c = G_MID if ok else RED
    o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>')
    if ok:
        o.append(f'<path d="M{x-9} {y+1}l6 6l12 -13" stroke="{BG}" stroke-width="4.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        o.append(f'<path d="M{x-7} {y-7}l14 14M{x+7} {y-7}l-14 14" stroke="{BG}" stroke-width="4.5" stroke-linecap="round"/>')


def along(pts, t):
    segs = [((x0, y0), (x1, y1), abs(x1 - x0) + abs(y1 - y0)) for (x0, y0), (x1, y1) in zip(pts, pts[1:])]
    L = sum(s[2] for s in segs) * max(0, min(1, t))
    for (x0, y0), (x1, y1), l in segs:
        if L <= l:
            f = L / l if l else 0
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        L -= l
    return pts[-1]


def layout_tokens(pieces, who, x0, y0, xmax=XMAX, start_line=True):
    """pieces: raw token strings; who: list (1 model, 2 implant). Returns list of (text, who, x, y, w, src_index).
    pure-space tokens -> gap, '\n' -> line break after the token."""
    out = []
    x, y = x0, y0
    gap_next, first = False, True
    for k, (p, wh) in enumerate(zip(pieces, who)):
        if p in ('<|endoftext|>', '<|im_end|>'):
            continue
        brk = '\n' in p
        s = p.replace('\n', '')
        lead = s.startswith(' ')
        s = s.strip(' ')
        if not s:
            gap_next = gap_next or lead or p == ' '
            if brk and not first:
                x, y, first = x0, y + LP, True
            continue
        w = len(s) * CHW + 2 * PAD
        if not first:
            x += GAP_W if (lead or gap_next) else GAP_G
        if x + w > xmax and not first:
            x, y = x0, y + LP
        out.append((s, wh, x, y, w, k))
        x += w
        first, gap_next = False, False
        if brk:
            x, y, first = x0, y + LP, True
    return out


def draw_tokens(o, items, n, new, op=1.0, hl=None):
    o.append(f'<g opacity="{op:.2f}">')
    for j in range(n):
        s, who, x, y, w, _ = items[j]
        g = who == 2
        fresh = j == n - 1 and new == 0
        if g:
            fill, stroke, col, sw = G_LIGHT, G_MID, G_DARK, 3
        elif hl == j:
            fill, stroke, col, sw = G_PALE, G_MID, G_DARK, 3.5
        else:
            fill, stroke, col, sw = BG, INK, INK, (3 if fresh else 2)
        if fresh and g:
            o.append(f'<rect x="{x-8:.1f}" y="{y-8}" width="{w+16:.1f}" height="{TH+16}" rx="14" fill="{G_MID}" opacity="0.4" filter="url(#glow)"/>')
        o.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{TH}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        T(o, x + w / 2, y + 36, s, TOKF, col, 700 if (g or hl == j) else 400, 'middle', MONO)
    o.append('</g>')


def draw_nextok(o, nt, struck=None):
    if nt is None:
        o.append(f'<rect x="{NTB[0]}" y="{NTB[1]}" width="{NTB[2]}" height="{NTB[3]}" rx="10" fill="{BG}" stroke="{BORDER}" stroke-width="2.5"/>')
        T(o, CX, NTB[1] + 46, 'next token', SMALL, MUTE, anchor='middle')
    else:
        s, g = nt
        o.append(f'<rect x="{NTB[0]}" y="{NTB[1]}" width="{NTB[2]}" height="{NTB[3]}" rx="10" fill="{G_LIGHT if g else BG}" stroke="{G_MID if g else INK}" stroke-width="{3 if g else 2.5}"/>')
        T(o, CX, NTB[1] + 48, s, TOKF + 8, G_DARK if g else INK, 700 if g else 400, 'middle', MONO)
    if struck:
        s, lab = struck
        bw = len(s) * CHW + 2 * PAD
        bx, by = NTB[0] + NTB[2] + 34, NTB[1] + 9
        o.append(f'<rect x="{bx}" y="{by}" width="{bw:.1f}" height="{TH}" rx="8" fill="{LAYER}" stroke="{MUTE}" stroke-width="2"/>')
        T(o, bx + bw / 2, by + 36, s, TOKF, MUTE, 400, 'middle', MONO)
        o.append(f'<path d="M{bx-8} {by+TH/2}h{bw+16:.1f}" stroke="{G_MID}" stroke-width="5" stroke-linecap="round"/>')
        T(o, bx + bw + 20, by + 36, lab, SMALL - 2, G_DARK, 600)


def draw_model(o, st, model_name, layer_labels, read_layer_idx):
    o.append(f'<rect x="{A.OX0}" y="{OY0}" width="{A.OX1-A.OX0}" height="{OY1-OY0}" rx="14" fill="{OUTER}" stroke="{INK}" stroke-width="2.5"/>')
    T(o, A.OX0 + 36, OY0 + 50, 'ONE MODEL', SMALL, MUTE, 700, ls=3)
    o.append(f'<rect x="{A.LX0}" y="{PY0}" width="{A.LX1-A.LX0}" height="{PY1-PY0}" rx="12" fill="{BG}" stroke="{INK}" stroke-width="3"/>')
    o.append(f'<path d="M{CX} {PY0}V{NTB[1]+NTB[3]+12}" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
    T(o, A.LX0 + 32, PY0 + 60, 'LANGUAGE MODEL', MID, INK, 700, ls=3)
    T(o, A.LX0 + 32, PY0 + 102, model_name, SMALL, MUTE)
    o.append(f'<rect x="{A.BX0}" y="{yHead}" width="{A.BX1-A.BX0}" height="{A.HB}" rx="8" fill="{LAYER}" stroke="{INK}" stroke-width="2"/>')
    T(o, CX, yHead + 36, 'output head', SMALL, MUTE, anchor='middle')
    for li, (y, lab) in enumerate(zip((yTop, yMid, yBot), layer_labels)):
        rd = li == read_layer_idx and st['read']
        if rd:
            o.append(f'<rect x="{A.BX0-8}" y="{y-8}" width="{A.BX1-A.BX0+16}" height="{LB+16}" rx="12" fill="{G_MID}" opacity="0.3" filter="url(#glow)"/>')
        o.append(f'<rect x="{A.BX0}" y="{y}" width="{A.BX1-A.BX0}" height="{LB}" rx="8" fill="{G_PALE if rd else LAYER}" stroke="{G_MID if rd else INK}" stroke-width="{3 if rd else 2}"/>')
        T(o, A.BX0 + 20, y + LB / 2 + 11, lab, SMALL, G_DARK if rd else MUTE, 700 if rd else 400)
        lv = st['neur'][li]
        y0 = y + (LB - A.NROWS * A.NQ - (A.NP - A.NQ)) / 2
        for r in range(A.NROWS):
            for c in range(NCOLS):
                v = lv[r * NCOLS + c]
                o.append(f'<rect x="{NX0 + c*A.NP}" y="{y0 + r*A.NP:.1f}" width="{A.NQ}" height="{A.NQ}" rx="2.5" fill="{N_OFF if v < 0 else N_LV[v]}"/>')
    for yb in (yTop, yMid):
        for k in range(3):
            o.append(f'<circle cx="{CX}" cy="{yb + LB + 9 + k*16}" r="4" fill="{MUTE}"/>')
    o.append(f'<path d="M{jx} {yTop}V{yHead+A.HB}" stroke="{INK}" stroke-width="2"/>')
    if st['plus_flash']:
        o.append(f'<circle cx="{jx}" cy="{wy}" r="30" fill="{G_MID}" opacity="0.4" filter="url(#glow)"/>')
        o.append(f'<circle cx="{jx}" cy="{wy}" r="19" fill="{G_MID}" stroke="{G_DARK}" stroke-width="3.5"/>')
        o.append(f'<path d="M{jx-10} {wy}h20M{jx} {wy-10}v20" stroke="{BG}" stroke-width="3.5" stroke-linecap="round"/>')
    else:
        o.append(f'<circle cx="{jx}" cy="{wy}" r="17" fill="{BG}" stroke="{G_MID}" stroke-width="3.5"/>')
        o.append(f'<path d="M{jx-9} {wy}h18M{jx} {wy-9}v18" stroke="{G_MID}" stroke-width="3.5" stroke-linecap="round"/>')
    # token input arrow (tokens -> model)
    o.append(f'<path d="M{CX} {TOK_Y0-6}V{PY1+8}" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')


def draw_meter(o, st, title, lo, hi, thr, thr_lab, fmt, ends=('no', 'yes')):
    """horizontal meter in the implants panel. st['meter'] = current value or None."""
    v = st['meter']
    on = v is not None and v > thr
    x0, x1, y = MTR_X0, MTR_X1, MTR_Y
    h = 30
    X = lambda u: x0 + (x1 - x0) * (min(hi, max(lo, u)) - lo) / (hi - lo)
    act = v is not None
    T(o, x0, y - h / 2 - 18, title, SMALL - 2, G_DARK, 700 if act else 600, op=1 if act else 0.6)
    o.append(f'<rect x="{x0}" y="{y-h/2}" width="{x1-x0}" height="{h}" rx="{h/2}" fill="{BG}" stroke="{G_LINE}" stroke-width="2.5"/>')
    if act:
        xv = X(v)
        if on:
            o.append(f'<rect x="{x0}" y="{y-h/2-6}" width="{xv-x0:.1f}" height="{h+12}" rx="{h/2+6}" fill="{G_MID}" opacity="0.35" filter="url(#glow)"/>')
        o.append(f'<rect x="{x0+3}" y="{y-h/2+3}" width="{max(0, xv-x0-6):.1f}" height="{h-6}" rx="{(h-6)/2}" fill="{G_MID if on else N_LV[1]}"/>')
    xt = X(thr)
    o.append(f'<path d="M{xt:.1f} {y-h/2-10}V{y+h/2+10}" stroke="{INK}" stroke-width="3.5" stroke-linecap="round"/>')
    T(o, xt, y + h / 2 + 38, thr_lab, 23, INK, 600, 'middle')
    T(o, x0 + 2, y + h / 2 + 38, ends[0], 23, MUTE, 600)
    T(o, x1 - 2, y + h / 2 + 38, ends[1], 23, MUTE, 600, 'end')
    if act and st['mval']:
        T(o, x1, y - h / 2 - 18, fmt(v), SMALL, G_DARK if on else INK, 700, 'end', MONO)


# ---------------- data ----------------
V2 = 'v2' in sys.argv[1:]
NAME = 'detector_v2' if V2 else 'detector'
K1 = '755' if V2 else '1160'
TR = json.load(open(os.path.join(OUTD, NAME + '_trace.json')))
EX = TR['examples']
PHRASE_N = 9                                   # 'Hmm', ',', ' wait', '.', ' Let', ' me', ' rethink', ' this', '.\n'


def beat_data(key, fired):
    e = EX[key]
    pcs = e['gen_pieces']
    cut = e['cut']
    who = [2 if (fired and cut <= k < cut + PHRASE_N) else 1 for k in range(len(pcs))]
    qlines = wrap(e['question'], 30, XMAX - XS)
    ytok = TOK_Y0 + 8 + len(qlines) * 40 + 34
    items = layout_tokens(pcs, who, XS, ytok)
    read_j = next(j for j, it in enumerate(items) if it[5] >= cut) - 1   # last token before the read point
    return dict(e=e, items=items, qlines=qlines, ytok=ytok, read_j=read_j, fired=fired, pcs=pcs)


B = [beat_data(K1, True), beat_data('462', False)]
assert ''.join(B[0]['pcs'][B[0]['e']['cut']:B[0]['e']['cut'] + PHRASE_N]) == 'Hmm, wait. Let me rethink this.\n'
assert B[0]['e']['fires_B'] and not B[1]['e']['fires_B']
assert B[0]['e']['host_top'] == '####' and B[1]['e']['host_top'] == '####'
H = int(max(b['items'][-1][3] for b in B) + TH + 64 + 70 + 60)
LAB = ['layer 28', 'layers 12–24', 'layer 1']


def render(st):
    b = B[st['beat']]
    e = b['e']
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">', defs(),
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    T(o, 60, 72, 'A mistake-detector implant', 46, INK, 700)
    T(o, 60, 116, f'GSM8K test problem {st["prob"]} · frozen Qwen3-0.6B-Base', SMALL, MUTE)
    draw_nextok(o, st['nextok'], st['struck'])
    draw_model(o, st, 'Qwen3-0.6B-Base · frozen', LAB, 1)

    # ---- implant panel ----
    o.append(f'<rect x="{MX0}" y="{PY0}" width="{MX1-MX0}" height="{PY1-PY0}" rx="10" fill="{G_PALE}" stroke="{G_LINE}" stroke-width="3"/>')
    T(o, MX0 + 28, PY0 + 60, 'MISTAKE DETECTOR', MID, G_DARK, 700, ls=3)
    T(o, MX0 + 28, PY0 + 104, st['status'] or 'silent: adds exactly zero', SMALL, G_DARK,
      700 if st['status'] else 400, op=1 if st['status'] else 0.75)
    wact = st['wact']
    o.append(f'<path d="M{MX0} {wy}H{jx+24}" stroke="{G_MID}" stroke-width="4" marker-end="url(#aG)" opacity="{1 if wact else 0.5}"/>')
    o.append(f'<path d="M{WR_X-CS/2-14} {wy}H{MX0}" stroke="{G_MID}" stroke-width="4" opacity="{1 if wact else 0.45}"/>')
    T(o, (A.LX1 + MX0) / 2, wy - 20, 'write', SMALL, G_DARK, 600, 'middle', op=1 if wact else 0.6)
    chip(o, WR_X, wy, CS, lit=st['wr_on'], op=1 if st['wr_on'] else 0.55)
    T(o, WR_X + CS / 2 + 30, wy + 11, 'rethink writer', SMALL, G_DARK, 700 if st['wr_on'] else 600, op=1 if st['wr_on'] else 0.6)
    # read line: layers 12-24 -> detector
    rd = st['read']
    gx0, gx1 = mlp_glyph(o, DET_X, DET_Y, st['det_on'], 1 if (st['det_on'] or rd) else 0.55)
    rl = f'M{A.BX1} {DET_Y}H{gx0-12}'
    if rd:
        o.append(f'<path d="{rl}" stroke="{G_MID}" stroke-width="12" fill="none" opacity="0.3" filter="url(#glow)"/>')
    o.append(f'<path d="{rl}" stroke="{G_MID}" stroke-width="{4 if rd else 3}" fill="none" marker-end="url(#aG)" opacity="{1 if rd else 0.45}"/>')
    T(o, (A.LX1 + MX0) / 2, DET_Y - 20, 'read', SMALL, G_DARK, 600, 'middle', op=1 if rd else 0.6)
    T(o, DET_X, DET_Y + 84, 'detector', SMALL, G_DARK, 700 if st['det_on'] else 600, 'middle', op=1 if st['det_on'] else 0.6)
    o.append(f'<path d="M{gx1+8} {DET_Y}H{MTR_X0-14}" stroke="{G_MID}" stroke-width="3" marker-end="url(#aG)" opacity="{0.9 if st["det_on"] else 0.4}"/>')
    # gate: detector -> writer
    gon = st['gate']
    gp = f'M{DET_X} {DET_Y-56}V{wy+CS/2+22}'
    if gon:
        o.append(f'<path d="{gp}" stroke="{G_MID}" stroke-width="12" fill="none" opacity="0.3" filter="url(#glow)"/>')
    o.append(f'<path d="{gp}" stroke="{G_MID}" stroke-width="{4 if gon else 3}" fill="none" marker-end="url(#aG)" opacity="{1 if gon else 0.4}" stroke-dasharray="{"none" if gon else "7 7"}"/>')
    T(o, DET_X + 16, (DET_Y - 56 + wy + CS / 2 + 22) / 2 + 10, 'gate', 24, G_DARK, 600, op=1 if gon else 0.55)
    draw_meter(o, st, 'will a rethink help?', -5, 5, e['thB'], f'threshold {e["thB"]:.2f}', lambda v: f'{v:+.2f}')
    if st['pnow']:
        T(o, MTR_X0, MTR_Y + 104, f'its other head, P(right now): {100*e["p_correct_now"]:.1f}%', 24, MUTE, 600)

    # ---- question + tokens ----
    qop = st['qop']
    o.append(f'<g opacity="{qop:.2f}">')
    T(o, 60, TOK_Y0 + 38, 'question', SMALL, MUTE, 600)
    for k, l in enumerate(b['qlines']):
        T(o, XS, TOK_Y0 + 38 + k * 40, l, 30, INK, 400)
    o.append('</g>')
    if st['n'] > 0:
        o.append(f'<g opacity="{qop:.2f}">')
        T(o, 60, b['ytok'] + 36, 'model', SMALL, MUTE, 600)
        o.append('</g>')
    draw_tokens(o, b['items'], st['n'], st['new'], qop, st['hl'])
    if st['final']:
        s, who, x, y, w, _ = b['items'][-1]
        o.append(f'<g opacity="{qop:.2f}">')
        mark(o, x + w + 36, y + TH / 2, True)
        T(o, x + w + 68, y + 36, 'correct', SMALL, G_DARK, 700)
        if st['final'] == 2 and b['fired']:
            yy = y + LP + 36
            mark(o, XS + 18, yy - 10, False, 16)
            nn = ''.join(e['none_pieces']).replace('<|endoftext|>', '').strip().split('\n')[-1]
            T(o, XS + 46, yy, f'without the implant the same model commits “{nn}”', SMALL - 2, RED, 600)
        o.append('</g>')
    for (x, y) in st['dots']:
        dot(o, x, y)
    T(o, W / 2, H - 30, st['cap'], SMALL, MUTE, anchor='middle')
    o.append('</svg>')
    return '\n'.join(o)


# ---------------- timeline ----------------
def ease(t):
    return t * t * (3 - 2 * t)


def make_neur(seed=7):
    rng = random.Random(seed)
    neur = [[-1] * NN for _ in range(3)]

    def fire():
        for li in range(3):
            neur[li] = [(rng.choice([0, 1, 1, 2, 2]) if rng.random() < 0.32 else -1) for _ in range(NN)]

    def decay():
        for li in range(3):
            nv = []
            for v in neur[li]:
                if v >= 0:
                    v = v - 1 if rng.random() < 0.45 else v
                elif rng.random() < 0.025:
                    v = 0
                nv.append(v)
            neur[li] = nv
    return neur, fire, decay


def build():
    neur, fire, decay = make_neur()
    frames = []
    st = dict(beat=0, prob=int(K1), n=0, new=9, nextok=None, struck=None, read=False, det_on=False, gate=False, wr_on=False,
              wact=False, plus_flash=False, meter=None, mval=False, pnow=False, status=None, qop=0.0, hl=None, final=0,
              dots=[], cap='')

    def snap(key=None, tick=True):
        if tick:
            decay()
        st['new'] += 1
        d = dict(st); d['dots'] = list(st['dots']); d['neur'] = [list(x) for x in neur]; d['key'] = key
        frames.append(d)

    def tok(j, nf=1, **kw):
        fire()
        st.update(n=j + 1, new=0, **kw)
        snap(tick=False)
        for _ in range(nf - 1):
            snap()

    for bi, b in enumerate(B):
        e, items, rj = b['e'], b['items'], b['read_j']
        first = bi == 0
        st.update(beat=bi, prob=int([K1, '462'][bi]), n=0, nextok=None, struck=None, final=0, hl=None, meter=None,
                  mval=False, pnow=False, qop=0.0,
                  cap='the question goes in; the frozen model writes its solution' if first else
                  'an easier problem: the same implant, attached the same way')
        for q in (0.25, 0.5, 0.75, 1.0):
            st['qop'] = q; snap()
        for _ in range(2):
            snap()
        # 1) the model writes up to the read point
        for j in range(rj + 1):
            tok(j, 1, nextok=(items[j][0], False))
        # 2) read point: the host's next token would be '####'
        st.update(nextok=('####', False), hl=rj,
                  cap='about to commit the final answer: the model’s own next token is ####')
        for _ in range(7 if first else 4):
            snap('k_hash' if first else None)
        # 3) the detector reads layers 12-24
        st.update(read=True, status='read layers 12–24', cap='the implant reads the model’s hidden state at layers 12–24 …')
        P = [(A.BX1, DET_Y), (MX0 + 70, DET_Y)]
        for p in ((0.2, 0.5, 0.8, 1.0) if first else (0.35, 0.75, 1.0)):
            st['dots'] = [along(P, p)]; snap()
        st.update(dots=[], det_on=True, status='detector: score this answer',
                  cap='… and its small detector scores the answer about to be committed')
        nf = 11 if first else 7
        for f in range(nf):
            u = ease((f + 1) / nf)
            st.update(meter=-5 + (e['sB'] + 5) * u, mval=True)
            snap('k_meter' if (first and f == nf - 1) else None)
        st['pnow'] = True
        if b['fired']:
            st.update(status='gate open: rethink before ####',
                      cap=f'score {e["sB"]:+.2f} is above the threshold {e["thB"]:.2f}: a rethink should help')
            for _ in range(9):
                snap()
            # 4) gate -> writer -> write the phrase instead of '####'
            st.update(gate=True, wr_on=True, cap='so instead of ####, the implant writes a rethink phrase')
            Pg = [(DET_X, DET_Y - 56), (DET_X, wy + CS / 2 + 16)]
            for p in (0.35, 0.75, 1.0):
                st['dots'] = [along(Pg, p)]; snap()
            st.update(dots=[], wact=True, read=False, det_on=False, hl=None)
            j0 = rj + 1
            Pw = [(WR_X - CS / 2 - 14, wy), (jx + 24, wy)]
            for k in range(PHRASE_N):
                j = j0 + k
                txt = items[j][0]
                st['status'] = f'write “{txt}”'
                for p in ((0.2, 0.55, 0.9) if k == 0 else (0.5,)):
                    st['dots'] = [along(Pw, p)]; snap()
                st.update(dots=[], plus_flash=True, struck=('####', 'replaced') if k < 4 else None)
                tok(j, 2 if k == 0 else 1, nextok=(txt, True))
                st['plus_flash'] = False
                if k == 0:
                    for _ in range(4):
                        snap('poster')
            st.update(gate=False, wr_on=False, wact=False, struck=None, meter=None, mval=False, pnow=False, status=None,
                      cap='then it goes silent; the frozen model rethinks on its own')
            for _ in range(3):
                snap()
            # 5) the model continues to the corrected answer (2 tokens per frame)
            rest = list(range(j0 + PHRASE_N, len(items)))
            for c in range(0, len(rest), 2):
                jj = rest[c:c + 2]
                fire()
                st.update(n=jj[-1] + 1, new=0, nextok=(items[jj[-1]][0], False))
                snap(tick=False)
            st.update(nextok=None, final=1, cap='… and reaches the right answer, ' + ('7' if V2 else '1125'))
            for _ in range(6):
                snap()
            st.update(final=2, cap=('every sum was right, the plan was not: it doubled Friday, not Monday' if V2 else
                          'the detector caught the slip 5168 − 1907 − 2136 = 1925 before it was committed'))
            for _ in range(20 if V2 else 26):
                snap('k_end')
        else:
            st.update(cap=f'score {e["sB"]:+.2f} is far below the threshold: no rethink, the implant stays silent')
            for _ in range(11):
                snap('k_quiet')
            st.update(read=False, det_on=False, hl=None, status=None)
            for j in range(rj + 1, len(items)):
                tok(j, 1, nextok=(items[j][0], False))
            st.update(nextok=None, final=1, meter=None, mval=False, pnow=False, cap='the model commits its own answer, 71: correct, and nothing was added')
            for _ in range(15 if V2 else 18):
                snap('k_end2')
        # fade out
        for q in (0.75, 0.5, 0.25, 0.0):
            st['qop'] = q; snap()
        st.update(n=0, final=0)
    return frames


def render_all(frames, name, fr_dir, poster_key='poster'):
    os.makedirs(fr_dir, exist_ok=True)
    for f in os.listdir(fr_dir):
        os.remove(os.path.join(fr_dir, f))
    svg = os.path.join(fr_dir, 'tmp.svg')
    done = False
    for i, fr in enumerate(frames):
        open(svg, 'w').write(render_fn[0](fr))
        subprocess.run(['rsvg-convert', '-w', str(PX_W), '-b', 'white', svg, '-o', os.path.join(fr_dir, f'f{i:04d}.png')], check=True)
        if fr['key'] == poster_key and not done:
            subprocess.run(['rsvg-convert', '-w', '2600', '-b', 'white', svg, '-o', os.path.join(OUTD, f'{name}_poster.png')], check=True)
            done = True
    os.remove(svg)
    gif = os.path.join(OUTD, f'{name}.gif')
    pat = os.path.join(fr_dir, 'f%04d.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', pat, '-vf',
                    'split[a][b];[a]palettegen=max_colors=96:stats_mode=full:reserve_transparent=0[p];'
                    '[b][p]paletteuse=dither=none:diff_mode=rectangle', '-loop', '0', gif], check=True)
    if shutil.which('gifsicle'):
        subprocess.run(['gifsicle', '-O3', '-b', gif], check=True)
    mp4 = os.path.join(OUTD, f'{name}.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-stream_loop', '1', '-framerate', str(FPS), '-i', pat,
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '18', '-preset', 'slow', '-movflags', '+faststart', mp4], check=True)
    for p in (gif, mp4):
        print(p, f'{os.path.getsize(p)/1e6:.2f} MB')


def keys(frames, fr_dir, extra=()):
    os.makedirs(fr_dir, exist_ok=True)
    seen = set()
    picks = []
    for i, f in enumerate(frames):
        k = f['key']
        if k and k not in seen:
            seen.add(k); picks.append((k, i if not k.startswith('k_end') else i + 10))
    picks += list(extra)
    for k, i in picks:
        i = min(i, len(frames) - 1)
        svg = os.path.join(fr_dir, k + '.svg')
        open(svg, 'w').write(render_fn[0](frames[i]))
        subprocess.run(['rsvg-convert', '-w', str(PX_W), '-b', 'white', svg, '-o', os.path.join(fr_dir, k + '.png')], check=True)
        os.remove(svg)
        print(k, i)


render_fn = [render]

if __name__ == '__main__':
    frames = build()
    print(len(frames), 'frames', f'{len(frames)/FPS:.1f}s', 'H', H, [(b['e']['sB'], b['read_j']) for b in B])
    fr_dir = os.path.join(OUTD, 'frames_' + NAME)
    if len(sys.argv) > 1 and sys.argv[1] == 'keys':
        keys(frames, fr_dir, extra=[('k_gen', 20)])
    else:
        render_all(frames, NAME, fr_dir)
