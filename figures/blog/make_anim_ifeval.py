"""Animated rule-keeper implant (IFEval, frozen Qwen3-0.6B, held-out prompt key 3615).
The no-comma keeper reads the prompt tokens; at "doesn't use commas" it latches. While the assistant writes, it adds a
solved veto vector to the last hidden state (every step) that pushes every comma token below the model's other choices.
It writes no text: when the host's top choice is ',' the host simply takes its next-best token.
Same palette / type / helpers as make_anim.py.

Real data: figs/anim/ifeval_trace.json (per-step trace from A40 pod, same bs16 batch as the reported held-out run):
  host top-3 WITHOUT the veto (same hidden state) and the token chosen WITH it. Comma would have won at steps 6 and 9.

Usage:
  python3 make_anim_ifeval.py          # all frames -> figs/anim/frames_ifeval, then GIF + MP4 + poster
  python3 make_anim_ifeval.py keys     # a few key frames as PNG (layout check)
Needs: rsvg-convert, ffmpeg, gifsicle (optional), PIL (text measuring for the end panels).
"""
import sys, os, json, random, subprocess, shutil
from PIL import ImageFont
import make_anim as A
from make_anim import (T, chip, esc, BG, BORDER, INK, MUTE, LAYER, OUTER, G_DARK, G_MID, G_LINE, G_LIGHT, G_PALE,
                       N_OFF, N_LV, SANS, MONO, MID, SMALL)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'figs', 'anim')
FR = os.path.join(OUTD, 'frames_ifeval')
FPS = 12
PX_W = 1600
RED, RED_PALE = '#b4232a', '#fbe9ea'          # used only for the failing commas / the cross
W, H = 1800, 1630
DY = 330                                       # the model block of make_anim.py, moved down by DY
TOKF = 34
CHW = TOKF * 0.602
PAD, GAP_W, GAP_G, TH, LP = 12, 14, 4, 60, 76  # token box padding, gaps, height, line pitch

TR = json.load(open(os.path.join(OUTD, 'ifeval_trace.json')))
IMP, BASE = TR['implant'], TR['base']
assert IMP['text'].startswith('Camilla is a pretty girl with a smile.')
STEPS = [s for s in IMP['steps'] if s['s'] != '<|im_end|>']
VETO = [s['step'] for s in STEPS if s['comma_would_win']]          # [6, 9]

# ---------------- tokens ----------------
pt = TR['prompt_tokens']
P0 = pt.index('Write'); P1 = pt.index('<|im_end|>')
PROMPT = pt[P0:P1]                                                  # Write a r iddle about Cam illa that doesn 't use commas .
INSTR = set(range(PROMPT.index(' doesn'), PROMPT.index(' commas') + 1))
LATCH = PROMPT.index(' commas')
TOKS = [(t, 0) for t in PROMPT] + [(s['s'], 1) for s in STEPS]       # (raw token, 0 prompt / 1 model)
NP_ = len(PROMPT)

SX, XMAX = 96, W - 70
TY0 = A.OY1 + DY + 64
TPOS = []
x, line = SX + 150, 0                                                # room for the 'user' label
for i, (t, who) in enumerate(TOKS):
    s = t.lstrip(' ')
    w = len(s) * CHW + 2 * PAD
    glue = not t.startswith(' ')
    if i == NP_:                                                     # assistant starts a new line
        x, line = SX + 150, 1
    elif i > 0:
        x += GAP_G if glue else GAP_W
    if x + w > XMAX:
        x, line = SX + 150, line + 1
    TPOS.append((x, TY0 + line * LP, w, s))
    x += w
NLINES = line + 1

# ---------------- top-3 card ----------------
CX = A.CX
NTB = (CX - 140, 190, 280, 70)                                       # next-token box (absolute)
CARD = (700, 128, 500, 330)                                          # x, y, w, h
ROW0, RP = CARD[1] + 70, 60


def draw_card(o, rows, op, title, foot):
    """rows: list of dict(s, y (row float), op, strike (0..1), pick, gone)"""
    if op <= 0:
        return
    x0, y0, w, h = CARD
    o.append(f'<g opacity="{op:.2f}">')
    o.append(f'<path d="M{NTB[0]+NTB[2]} {NTB[1]+NTB[3]/2}H{x0}" stroke="{BORDER}" stroke-width="2.5" stroke-dasharray="6 6"/>')
    o.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="12" fill="{BG}" stroke="{INK}" stroke-width="2.5"/>')
    T(o, x0 + 24, y0 + 44, title, SMALL - 2, MUTE, 600)
    o.append(f'<clipPath id="cc"><rect x="{x0}" y="{y0+56}" width="{w}" height="{h-60}"/></clipPath><g clip-path="url(#cc)">')
    for r in rows:
        yy = ROW0 + r['y'] * RP
        ro = r.get('op', 1)
        if ro <= 0:
            continue
        rank = r['y'] + 1
        o.append(f'<g opacity="{ro:.2f}">')
        if 0.5 <= rank <= 3.5 and abs(rank - round(rank)) < 0.05:
            T(o, x0 + 30, yy + 38, f'{int(round(rank))}', SMALL, MUTE, 600)
        s = r['s'].lstrip(' ')
        bw = max(len(s), 1) * CHW + 2 * PAD
        bx = x0 + 74 + r.get('dx', 0)
        pick = r.get('pick', 0)
        fill, stroke, col, sw = (G_PALE, G_MID, G_DARK, 3.5) if pick else (BG, INK, INK, 2)
        if r.get('gone'):
            fill, stroke, col = LAYER, MUTE, MUTE
        o.append(f'<rect x="{bx}" y="{yy}" width="{bw:.1f}" height="50" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        T(o, bx + bw / 2, yy + 36, s, TOKF - 2, col, 700 if pick else 400, 'middle', MONO)
        if r.get('lab'):
            T(o, bx + bw + 22, yy + 36, r['lab'], SMALL - 2, r.get('labc', MUTE), 600)
        st = r.get('strike', 0)
        if st > 0:
            o.append(f'<path d="M{bx-10} {yy+25}h{(bw+20)*st:.1f}" stroke="{G_MID}" stroke-width="5" stroke-linecap="round"/>')
        o.append('</g>')
    o.append('</g>')
    if foot:
        T(o, x0 + 24, y0 + h - 18, foot, SMALL - 4, G_DARK, 600)
    o.append('</g>')


NAMES = ['no-comma implant', 'quote', 'lowercase', 'repeat', 'forbid words', 'two answers', 'paragraphs', 'nth word']


def render_main(o, st):
    # next token box
    nt = st['nextok']
    if nt is None:
        o.append(f'<rect x="{NTB[0]}" y="{NTB[1]}" width="{NTB[2]}" height="{NTB[3]}" rx="10" fill="{BG}" stroke="{BORDER}" stroke-width="2.5"/>')
        T(o, CX, NTB[1] + 46, 'next token', SMALL, MUTE, anchor='middle')
    else:
        s, vg = nt
        o.append(f'<rect x="{NTB[0]}" y="{NTB[1]}" width="{NTB[2]}" height="{NTB[3]}" rx="10" fill="{BG}" stroke="{G_MID if vg else INK}" stroke-width="{3.5 if vg else 2.5}"/>')
        T(o, CX, NTB[1] + 48, s.lstrip(' '), TOKF + 4, G_DARK if vg else INK, 700 if vg else 400, 'middle', MONO)
    draw_card(o, st['rows'], st['card'], st['ctitle'], st['cfoot'])

    o.append(f'<g transform="translate(0,{DY})">')
    o.append(f'<rect x="{A.OX0}" y="{A.OY0}" width="{A.OX1-A.OX0}" height="{A.OY1-A.OY0}" rx="14" fill="{OUTER}" stroke="{INK}" stroke-width="2.5"/>')
    T(o, A.OX0 + 36, A.OY0 + 50, 'ONE MODEL', SMALL, MUTE, 700, ls=3)
    # ---- language model (as make_anim.py) ----
    o.append(f'<rect x="{A.LX0}" y="{A.PY0}" width="{A.LX1-A.LX0}" height="{A.PY1-A.PY0}" rx="12" fill="{BG}" stroke="{INK}" stroke-width="3"/>')
    o.append(f'<path d="M{CX} {A.PY0}V{NTB[1]+NTB[3]+12-DY}" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
    T(o, A.LX0 + 32, A.PY0 + 60, 'LANGUAGE MODEL', MID, INK, 700, ls=3)
    T(o, A.LX0 + 32, A.PY0 + 102, 'Qwen3-0.6B · frozen', SMALL, MUTE)
    o.append(f'<rect x="{A.BX0}" y="{A.yHead}" width="{A.BX1-A.BX0}" height="{A.HB}" rx="8" fill="{LAYER}" stroke="{INK}" stroke-width="2"/>')
    T(o, CX, A.yHead + 36, 'output head', SMALL, MUTE, anchor='middle')
    for li, (y, lab) in enumerate(A.LAYERS):
        o.append(f'<rect x="{A.BX0}" y="{y}" width="{A.BX1-A.BX0}" height="{A.LB}" rx="8" fill="{LAYER}" stroke="{INK}" stroke-width="2"/>')
        T(o, A.BX0 + 20, y + A.LB / 2 + 11, lab, SMALL, MUTE)
        lv = st['neur'][li]
        y0 = y + (A.LB - A.NROWS * A.NQ - (A.NP - A.NQ)) / 2
        for r in range(A.NROWS):
            for c in range(A.NCOLS):
                v = lv[r * A.NCOLS + c]
                o.append(f'<rect x="{A.NX0 + c*A.NP}" y="{y0 + r*A.NP:.1f}" width="{A.NQ}" height="{A.NQ}" rx="2.5" fill="{N_OFF if v < 0 else N_LV[v]}"/>')
    for k in range(3):
        o.append(f'<circle cx="{CX}" cy="{A.yL27 + A.LB + 26 + k*16}" r="4" fill="{MUTE}"/>')
    jx, wy = A.jx, A.wy
    o.append(f'<path d="M{jx} {A.yL28}V{A.yHead+A.HB}" stroke="{INK}" stroke-width="2"/>')
    if st['plus_flash']:
        o.append(f'<circle cx="{jx}" cy="{wy}" r="30" fill="{G_MID}" opacity="0.4" filter="url(#glow)"/>')
        o.append(f'<circle cx="{jx}" cy="{wy}" r="19" fill="{G_MID}" stroke="{G_DARK}" stroke-width="3.5"/>')
        o.append(f'<path d="M{jx-10} {wy}h20M{jx} {wy-10}v20" stroke="{BG}" stroke-width="3.5" stroke-linecap="round"/>')
    else:
        o.append(f'<circle cx="{jx}" cy="{wy}" r="17" fill="{BG}" stroke="{G_MID}" stroke-width="3.5"/>')
        o.append(f'<path d="M{jx-9} {wy}h18M{jx} {wy-9}v18" stroke="{G_MID}" stroke-width="3.5" stroke-linecap="round"/>')

    # ---- micromachines panel ----
    MX0, MX1 = A.MX0, A.MX1
    latched, writing = st['latched'], st['writing']
    o.append(f'<rect x="{MX0}" y="{A.PY0}" width="{MX1-MX0}" height="{A.PY1-A.PY0}" rx="10" fill="{G_PALE}" stroke="{G_LINE}" stroke-width="3"/>')
    T(o, MX0 + 28, A.PY0 + 60, 'RULE-KEEPER IMPLANTS', MID, G_DARK, 700, ls=3)
    if writing:
        T(o, MX0 + 28, A.PY0 + 104, 'veto commas, every step', SMALL, G_DARK, 700)
    elif latched:
        T(o, MX0 + 28, A.PY0 + 104, 'no comma: latched', SMALL, G_DARK, 700)
    else:
        T(o, MX0 + 28, A.PY0 + 104, 'silent: adds exactly zero', SMALL, G_DARK, op=0.75)
    cx0, cy0 = A.CHIPS[0]
    o.append(f'<path d="M{MX0} {wy}H{jx+24}" stroke="{G_MID}" stroke-width="4" marker-end="url(#aG)" opacity="{1 if writing else 0.45}"/>')
    if writing:
        o.append(f'<path d="M{cx0 - A.CS/2 - 14} {wy}H{MX0}" stroke="{G_MID}" stroke-width="4"/>')
    T(o, (A.LX1 + MX0) / 2, wy - 20, 'veto', SMALL, G_DARK, 700 if writing else 600, 'middle', op=1 if writing else 0.6)
    for i, ((cx, cy), n) in enumerate(zip(A.CHIPS, NAMES)):
        lit = i == 0 and latched
        chip(o, cx, cy, A.CS, lit=lit, op=1 if lit else 0.4)
        T(o, cx, cy + 70, n, SMALL, G_DARK, 700 if lit else 600, 'middle', op=1 if lit else 0.5)
    if latched:                                                       # 'latched' badge on the chip
        bx, by = cx0 + 34, cy0 - 62
        o.append(f'<rect x="{bx}" y="{by}" width="112" height="34" rx="17" fill="{G_MID}"/>')
        T(o, bx + 56, by + 24, 'latched', 22, BG, 700, 'middle')
    if st['pulse'] is not None:
        px = (cx0 - A.CS / 2 - 14) + (jx + 24 - (cx0 - A.CS / 2 - 14)) * st['pulse']
        o.append(f'<circle cx="{px:.1f}" cy="{wy}" r="20" fill="{G_MID}" opacity="0.45" filter="url(#glow)"/>')
        o.append(f'<circle cx="{px:.1f}" cy="{wy}" r="10" fill="{G_MID}" stroke="{BG}" stroke-width="2.5"/>')
    o.append('</g>')

    # ---- token strip + input bus ----
    ybus, yb_top = A.OY1 + DY + 30, A.PY1 + DY + 8
    o.append(f'<path d="M{CX} {TY0-12}V{yb_top}" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
    on = st['bus']
    bus = f'M{CX} {ybus}H{cx0}V{yb_top}'
    if on:
        o.append(f'<path d="{bus}" stroke="{G_MID}" stroke-width="12" fill="none" opacity="0.35" filter="url(#glow)"/>')
    o.append(f'<path d="{bus}" stroke="{G_MID}" stroke-width="{4 if on else 3}" fill="none" opacity="{1 if on else 0.45}" marker-end="url(#aG)"/>')
    if on and st['inp'] is not None:
        L1_, L2_ = cx0 - CX, ybus - yb_top
        d = st['inp'] * (L1_ + L2_)
        px, py = (CX + d, ybus) if d < L1_ else (cx0, ybus - (d - L1_))
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="20" fill="{G_MID}" opacity="0.45" filter="url(#glow)"/>')
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="10" fill="{G_MID}" stroke="{BG}" stroke-width="2.5"/>')
    T(o, SX, TY0 + 40, 'user', SMALL, MUTE, 600)
    if st['n'] > NP_:
        T(o, SX, TY0 + LP + 40, 'model', SMALL, MUTE, 600)
    for i in range(st['n']):
        x, y, w, s = TPOS[i]
        who = TOKS[i][1]
        ins = who == 0 and i in INSTR and st['hl']
        fresh = i == st['n'] - 1 and st['new'] == 0
        vg = who == 1 and (i - NP_) in VETO and st['vmark']
        if ins:
            fill, stroke, col, sw = G_LIGHT, G_MID, G_DARK, 3
        else:
            fill, stroke, col, sw = BG, INK, INK, (3 if fresh else 2)
        o.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{TH}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        T(o, x + w / 2, y + 42, s, TOKF, col, 700 if ins else 400, 'middle', MONO)
        if vg:
            o.append(f'<path d="M{x+3:.1f} {y+TH+9}h{w-6:.1f}" stroke="{G_MID}" stroke-width="5" stroke-linecap="round"/>')
    if st['hl'] and st['n'] <= NP_:                                  # bracket only while the prompt is read
        a = TPOS[min(INSTR)][0]
        b = TPOS[max(INSTR)][0] + TPOS[max(INSTR)][2]
        yb = TY0 + TH + 10
        o.append(f'<path d="M{a+4:.1f} {yb}v10H{b-4:.1f}v-10" fill="none" stroke="{G_MID}" stroke-width="4"/>')
    T(o, W / 2, H - 34, st['cap'], SMALL, MUTE, anchor='middle')


# ---------------- end panels ----------------
FD = os.path.join(HERE, 'fonts')
_F = {}


def tw(s, size, weight):
    k = (size, weight)
    if k not in _F:
        _F[k] = ImageFont.truetype(os.path.join(FD, 'AvenirNextBold.ttf' if weight >= 700 else
                                                'AvenirNextDemiBold.ttf' if weight >= 600 else 'AvenirNextRegular.ttf'), size)
    return _F[k].getlength(s)


def flow(o, segs, x0, y0, wmax, size, lh):
    """segs: list of (text, kind) kind in {'n','comma','veto'}; word-wrapped; '\n' forces a break. Returns last y."""
    words, cur = [], []
    for t, k in segs:                             # split into words, keeping the styled pieces
        parts = t.replace('\n', ' \n ').split(' ')
        for j, p in enumerate(parts):
            if j > 0 and cur:
                words.append(cur); cur = []
            if p == '\n':
                words.append([('\n', 'n')]); continue
            if p:
                cur.append((p, k))
    if cur:
        words.append(cur)
    sty = {'n': (INK, 400), 'comma': (RED, 700), 'veto': (G_DARK, 700)}
    sp = tw(' ', size, 400)
    x, y = x0, y0
    for wd in words:
        if wd[0][0] == '\n':
            x, y = x0, y + lh; continue
        wl = sum(tw(p, size, sty[k][1]) for p, k in wd)
        if x > x0 and x + wl > x0 + wmax:
            x, y = x0, y + lh
        for p, k in wd:
            c, wt = sty[k]
            pw = tw(p, size, wt)
            if k in ('comma', 'veto'):
                x += 5
                f_, s_ = (RED_PALE, RED) if k == 'comma' else (G_LIGHT, G_MID)
                o.append(f'<rect x="{x-5:.1f}" y="{y-size*0.9:.1f}" width="{pw+10:.1f}" height="{size*1.25:.1f}" rx="8" fill="{f_}" stroke="{s_}" stroke-width="2.5"/>')
            T(o, x, y, p, size, c, wt)
            x += pw + (5 if k in ('comma', 'veto') else 0)
        x += sp
    return y


def mark(o, x, y, ok, r=22):
    c = G_MID if ok else RED
    o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>')
    if ok:
        o.append(f'<path d="M{x-10} {y+1}l7 7l13 -15" stroke="{BG}" stroke-width="5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        o.append(f'<path d="M{x-8} {y-8}l16 16M{x+8} {y-8}l-16 16" stroke="{BG}" stroke-width="5" stroke-linecap="round"/>')


def render_end(o):
    y = 360
    T(o, 70, y, 'prompt:', SMALL, MUTE, 600)
    xx = 70 + tw('prompt: ', 30, 600) + 6
    pre, ins = 'Write a riddle about Camilla that ', 'doesn’t use commas'
    T(o, xx, y, pre, 34, INK, 400)
    T(o, xx + tw(pre, 34, 400), y, ins, 34, G_DARK, 700)
    T(o, xx + tw(pre, 34, 400) + tw(ins, 34, 700), y, '.', 34, INK, 400)
    PW, top, bot = 815, 420, 1370
    for k, (x0, ok) in enumerate(((70, False), (W - 70 - PW, True))):
        fill, stroke = (G_PALE, G_MID) if ok else (BG, INK)
        o.append(f'<rect x="{x0}" y="{top}" width="{PW}" height="{bot-top}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        T(o, x0 + 36, top + 62, 'WITH IMPLANT' if ok else 'WITHOUT IMPLANT', MID - 2, G_DARK if ok else INK, 700, ls=3)
        if ok:
            segs = []
            for s in STEPS:
                segs.append((s['s'], 'veto' if s['step'] in VETO else 'n'))
            segs = _merge(segs)
        else:
            txt = '\n'.join(l.rstrip() for l in BASE['text'].split('\n'))
            segs = []
            for j, piece in enumerate(txt.split(',')):
                if j:
                    segs.append((',', 'comma'))
                segs.append((piece, 'n'))
        flow(o, segs, x0 + 36, top + 156, PW - 72, 46, 78)
        mark(o, x0 + 62, bot - 70, ok, 26)
        T(o, x0 + 104, bot - 57, 'IFEval: pass · no commas' if ok else 'IFEval: fail · 2 commas', SMALL + 6, G_DARK if ok else RED, 700)
    T(o, W / 2, 1450, 'same frozen model, same greedy decoding · green: where a comma was vetoed', SMALL, MUTE, anchor='middle')


def _merge(segs):
    """join BPE pieces into text; veto pieces stay separate segments (the leading space is moved out)."""
    out = []
    for t, k in segs:
        if k == 'veto':
            if t.startswith(' '):
                out.append((' ', 'n')); t = t[1:]
            out.append((t, 'veto'))
        elif out and out[-1][1] == 'n':
            out[-1] = (out[-1][0] + t, 'n')
        else:
            out.append((t, 'n'))
    return out


def render(st):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<defs><marker id="aI" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker>'
         f'<marker id="aG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="{G_MID}"/></marker>'
         f'<filter id="glow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="12"/></filter></defs>',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    T(o, 60, 72, 'A rule-keeper implant', 46, INK, 700)
    T(o, 60, 116, 'IFEval · frozen Qwen3-0.6B · held-out prompt', SMALL, MUTE)
    e = st.get('end', 0)
    if e < 1:
        o.append(f'<g opacity="{1-e:.2f}">'); render_main(o, st); o.append('</g>')
    if e > 0:
        o.append(f'<g opacity="{e:.2f}">'); render_end(o); o.append('</g>')
    o.append('</svg>')
    return '\n'.join(o)


# ---------------- timeline ----------------
def ease(t):
    return t * t * (3 - 2 * t)


def build():
    rng = random.Random(7)
    NN = A.NROWS * A.NCOLS
    frames = []
    neur = [[-1] * NN for _ in A.LAYERS]

    def fire():
        for li in range(len(A.LAYERS)):
            neur[li] = [(rng.choice([0, 1, 1, 2, 2]) if rng.random() < 0.32 else -1) for _ in range(NN)]

    def decay():
        for li in range(len(A.LAYERS)):
            nv = []
            for v in neur[li]:
                if v >= 0:
                    v = v - 1 if rng.random() < 0.45 else v
                elif rng.random() < 0.025:
                    v = 0
                nv.append(v)
            neur[li] = nv

    st = dict(n=0, new=9, nextok=None, latched=False, writing=False, pulse=None, plus_flash=False, bus=False, inp=None,
              hl=False, vmark=True, rows=[], card=0, ctitle='', cfoot='', end=0,
              cap='the implant reads the prompt tokens, like the model does')

    def snap(key=None):
        d = dict(st); d['rows'] = [dict(r) for r in st['rows']]; d['neur'] = [list(x) for x in neur]; d['key'] = key; frames.append(d)

    for _ in range(3):
        decay(); snap()
    # 1) prompt streams in
    for i in range(NP_):
        for f in range(2):
            if f == 0:
                fire(); st.update(n=i + 1, new=0)
                if i in INSTR:
                    st.update(hl=True, bus=True)
            else:
                decay(); st['new'] += 1
            snap()
        if i == LATCH:
            for p in (0.25, 0.55, 0.85):
                decay(); st.update(inp=p); snap()
            st.update(latched=True, inp=None, cap='it sees “doesn’t use commas” and latches on')
            for _ in range(9):
                decay(); st['new'] += 1; snap()
            st.update(bus=False)
    st.update(hl=True)
    # 2) generation
    st.update(writing=True, cap='while the model writes, the implant adds a veto vector at every step')
    for k, s in enumerate(STEPS):
        i = NP_ + k
        if k in VETO:
            host = s['top3_host']; used = s['top3_used']
            assert ',' in host[0][0]
            first = k == VETO[0]
            st.update(cap='the frozen model’s top choice is a comma' if first else 'a comma wins again')
            rows = [dict(s=host[0][0], y=0), dict(s=host[1][0], y=1), dict(s=host[2][0], y=2),
                    dict(s=used[2][0], y=3, op=0)]
            st.update(rows=rows, ctitle='output head: top 3', cfoot='')
            for c in (0.35, 0.7, 1):
                decay(); st.update(card=c); snap()
            for f in range(7 if first else 3):
                decay(); snap()
            # veto pulse travels to the junction
            st.update(cap='the implant’s veto pushes every comma below the other choices')
            for p in (0.15, 0.45, 0.75, 1.0):
                decay(); st.update(pulse=p); snap()
            st.update(pulse=None, plus_flash=True)
            for sk in (0.5, 1):
                decay(); rows[0].update(strike=sk, gone=True); snap()
            st.update(plus_flash=False)
            nm = 9 if first else 7
            for f in range(nm):                         # comma steps aside, then sinks below the list
                u = (f + 1) / nm
                a_, b_ = ease(min(1, u / 0.35)), ease(max(0, (u - 0.35) / 0.65))
                rows[0].update(dx=220 * a_, y=3.2 * b_, op=1 - 0.45 * b_, lab='vetoed', labc=G_DARK)
                rows[1].update(y=1 - b_); rows[2].update(y=2 - b_)
                rows[3].update(y=2, op=max(0, (u - 0.7) / 0.3))
                decay(); snap()
            rows[1].update(pick=1, lab='picked', labc=G_DARK)
            st.update(
                      cap='the model picks its own next-best token: “' + s['s'].strip() + '”')
            fire(); st.update(n=i + 1, new=0, nextok=(s['s'], True))
            snap('poster' if first else None)
            for f in range(12 if first else 6):
                decay(); st['new'] += 1; snap()
            for c in (0.6, 0.25, 0):
                decay(); st.update(card=c); snap()
            st.update(cap='while the model writes, the implant adds a veto vector at every step')
            continue
        for f in range(2):
            if f == 0:
                fire(); st.update(n=i + 1, new=0, nextok=(s['s'], False))
            else:
                decay(); st['new'] += 1
            snap()
    st.update(cap='done: no commas in the answer')
    for _ in range(8):
        decay(); st['new'] += 1; snap()
    # 3) end panels
    for e in (0.25, 0.5, 0.75, 1):
        st.update(end=e); snap()
    for _ in range(40):
        snap('end')
    return frames


def main():
    keys_only = len(sys.argv) > 1 and sys.argv[1] == 'keys'
    os.makedirs(FR, exist_ok=True)
    frames = build()
    print(len(frames), 'frames', f'{len(frames)/FPS:.1f}s', 'lines', NLINES, 'veto steps', VETO)
    if keys_only:
        picks = {'ki_latch': next(i for i, f in enumerate(frames) if f['latched']) + 6,
                 'ki_card': next(i for i, f in enumerate(frames) if f['card'] == 1) + 2,
                 'ki_poster': next(i for i, f in enumerate(frames) if f['key'] == 'poster'),
                 'ki_slide': next(i for i, f in enumerate(frames) if f['key'] == 'poster') - 3,
                 'ki_gen': next(i for i, f in enumerate(frames) if f['key'] == 'poster') + 60,
                 'ki_end': len(frames) - 1}
        for name, i in picks.items():
            svg = os.path.join(FR, name + '.svg')
            open(svg, 'w').write(render(frames[i]))
            subprocess.run(['rsvg-convert', '-w', str(PX_W), '-b', 'white', svg, '-o', os.path.join(FR, name + '.png')], check=True)
        print('keys', picks)
        return
    for f in os.listdir(FR):
        os.remove(os.path.join(FR, f))
    svg = os.path.join(FR, 'tmp.svg')
    for i, fr in enumerate(frames):
        open(svg, 'w').write(render(fr))
        subprocess.run(['rsvg-convert', '-w', str(PX_W), '-b', 'white', svg, '-o', os.path.join(FR, f'f{i:04d}.png')], check=True)
        if fr['key'] == 'poster':
            subprocess.run(['rsvg-convert', '-w', '2600', '-b', 'white', svg, '-o', os.path.join(OUTD, 'ifeval_poster.png')], check=True)
    os.remove(svg)
    gif = os.path.join(OUTD, 'ifeval.gif')
    pat = os.path.join(FR, 'f%04d.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', pat, '-vf',
                    'split[a][b];[a]palettegen=max_colors=96:stats_mode=full:reserve_transparent=0[p];'
                    '[b][p]paletteuse=dither=none:diff_mode=rectangle', '-loop', '0', gif], check=True)
    if shutil.which('gifsicle'):
        subprocess.run(['gifsicle', '-O3', '-b', gif], check=True)
    mp4 = os.path.join(OUTD, 'ifeval.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-stream_loop', '1', '-framerate', str(FPS), '-i', pat,
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '18', '-preset', 'slow', '-movflags', '+faststart', mp4], check=True)
    for p in (gif, mp4):
        print(p, f'{os.path.getsize(p)/1e6:.2f} MB')


if __name__ == '__main__':
    main()
