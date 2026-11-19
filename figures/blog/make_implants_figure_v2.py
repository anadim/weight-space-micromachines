"""Implants figure v2: the implant set the post presents now (v4_live3), grouped by what they read / write.
Same SVG style as make_implants_figure.py (chips, palette, Avenir Next).
  read tokens  -> write tokens : calculator, eyes (CNN, trained), 14 IFEval rules, 8 BBH tools
  read hidden  -> write tokens : mistake detector (layers 12-24, trained heads), topic blocker (probe, layer 16)
  read tokens  -> write hidden : latent write (layers <= 12)
Usage: python3 make_implants_figure_v2.py && rsvg-convert -w 1600 -b white figs/0_implants_v2.svg -o figs/0_implants_v2.png
"""
import html, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figs', '0_implants_v2.svg')
BG, BORDER, INK, MUTE, LAYER, OUTER = '#ffffff', '#cfd5dc', '#222222', '#6b7280', '#f1f3f5', '#fafbfc'
G_DARK, G_MID, G_LINE, G_LIGHT, G_PALE = '#03623f', '#118659', '#b2dac4', '#cee7d9', '#f3f9f5'
B_DARK, B_MID, B_LINE, B_LIGHT, B_PALE = '#1d4f73', '#2f74a8', '#bcd4e8', '#dbe8f4', '#f1f6fb'
P_DARK, P_MID, P_LINE, P_LIGHT, P_PALE = '#4b2f7f', '#6d4fa3', '#d3c8ea', '#e6dff4', '#f6f3fb'
TR = '#374151'   # 'trained' badge
SANS = "'Avenir Next','Helvetica Neue',Helvetica,Arial,sans-serif"
MONO = "'SF Mono',Menlo,Consolas,monospace"
BIG, MID, LAB, SMALL, TINY = 54, 30, 28, 23, 19
esc = html.escape
W, H = 1600, 1610
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">', '<defs>']
for nm, c, s in (('aI', INK, 6), ('aG', G_MID, 5), ('aB', B_MID, 5), ('aP', P_MID, 5)):
    o.append(f'<marker id="{nm}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="{s}" markerHeight="{s}" orient="auto"><path d="M0 0L10 5L0 10z" fill="{c}"/></marker>')
o.append(f'</defs><rect width="{W}" height="{H}" fill="{BG}"/>')
A = o.append


def T(x, y, s, size=MID, fill=INK, weight=400, anchor='start', font=SANS, ls=0):
    sp = f' letter-spacing="{ls}"' if ls else ''
    A(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{sp}>{esc(s)}</text>')


def chip(cx, cy, S, col=G_MID, stack=0):
    h, P, st = S / 2, S * 0.2, S * 0.3
    for k in range(stack, 0, -1):          # stacked copies behind = a block of many implants
        dx = k * 9
        A(f'<rect x="{cx-h+dx}" y="{cy-h-dx}" width="{S}" height="{S}" rx="{S*0.13}" fill="{BG}" stroke="{col}" stroke-width="3" opacity="{0.9-0.25*k}"/>')
    pins = ''.join(f'M{cx+k*st} {cy-h}v-{P}M{cx+k*st} {cy+h}v{P}M{cx-h} {cy+k*st}h-{P}M{cx+h} {cy+k*st}h{P}' for k in (-1, 0, 1))
    A(f'<path d="{pins}" stroke="{col}" stroke-width="4" stroke-linecap="round" fill="none"/>')
    A(f'<rect x="{cx-h}" y="{cy-h}" width="{S}" height="{S}" rx="{S*0.13}" fill="{BG}" stroke="{col}" stroke-width="4"/>')
    d = S * 0.36
    A(f'<rect x="{cx-d/2}" y="{cy-d/2}" width="{d}" height="{d}" rx="4" fill="{col}"/>')


def badge(x, y, s='trained'):
    """small dark pill; (x, y) = left, vertical centre"""
    w = len(s) * TINY * 0.56 + 18
    A(f'<rect x="{x}" y="{y-14}" width="{w:.0f}" height="28" rx="14" fill="{TR}"/>')
    T(x + w / 2, y + 7, s, TINY, BG, 600, 'middle')
    return w


T(48, 84, 'One model, many implants', BIG, INK, 700)

# ---------------- frame ----------------
OX0, OX1, OY0, OY1 = 32, 1568, 250, 1300
LX0, LX1 = OX0 + 28, 560
PX0, PX1 = 622, 1500
CH = 594                      # channel between model and panels
CX = (LX0 + LX1) / 2
PY0, PY1 = 300, 1268

A(f'<rect x="{CX-130}" y="132" width="260" height="66" rx="10" fill="{BG}" stroke="{INK}" stroke-width="2.5"/>')
T(CX, 176, 'next token', MID, INK, anchor='middle')
A(f'<rect x="{OX0}" y="{OY0}" width="{OX1-OX0}" height="{OY1-OY0}" rx="14" fill="{OUTER}" stroke="{INK}" stroke-width="2.5"/>')
T(OX0 + 28, OY0 + 36, 'ONE MODEL', SMALL, MUTE, 700, ls=3)

# ---------------- language model ----------------
A(f'<rect x="{LX0}" y="{PY0}" width="{LX1-LX0}" height="{PY1-PY0}" rx="12" fill="{BG}" stroke="{INK}" stroke-width="3"/>')
A(f'<path d="M{CX} {PY0}V210" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
T(LX0 + 26, PY0 + 50, 'LANGUAGE MODEL', MID, INK, 700, ls=3)
T(LX0 + 26, PY0 + 86, 'small Qwen · own weights frozen', SMALL, MUTE)
BX0, BX1 = LX0 + 26, LX1 - 26


def bar(y0, y1, label, fill=LAYER, stroke=INK, col=MUTE, weight=400, sw=2):
    A(f'<rect x="{BX0}" y="{y0}" width="{BX1-BX0}" height="{y1-y0}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    T((BX0 + BX1) / 2, (y0 + y1) / 2 + 10, label, LAB, col, weight, 'middle')


def dots(y):
    for k in range(3):
        A(f'<circle cx="{CX}" cy="{y + k*16}" r="4" fill="{MUTE}"/>')


yH0, yH1 = 420, 468
wy = 512                     # token-write junction
yA0, yA1 = 552, 600          # layer 28
yM0, yM1 = 690, 860          # layers 12-24 (read by detector / blocker)
yR = 820                     # hidden read arrow
yW0, yW1 = 1062, 1158        # layers 1-12 (latent write)
yE0, yE1 = 1196, 1244        # embeddings
bar(yH0, yH1, 'output head')
A(f'<path d="M{CX} {yA0}V{yH1}" stroke="{INK}" stroke-width="2"/>')
A(f'<circle cx="{CX}" cy="{wy}" r="17" fill="{BG}" stroke="{G_MID}" stroke-width="3.5"/>')
A(f'<path d="M{CX-9} {wy}h18M{CX} {wy-9}v18" stroke="{G_MID}" stroke-width="3.5" stroke-linecap="round"/>')
bar(yA0, yA1, 'layer 28')
dots(yA1 + 28)
A(f'<rect x="{BX0}" y="{yM0}" width="{BX1-BX0}" height="{yM1-yM0}" rx="8" fill="{B_PALE}" stroke="{B_MID}" stroke-width="2.5"/>')
T(CX, yM0 + 52, 'layers 12–24', LAB, B_DARK, 600, 'middle')
# layer 16 tick inside the block
A(f'<path d="M{BX0+30} {yM0+112}H{BX1-30}" stroke="{B_MID}" stroke-width="2" stroke-dasharray="6 6"/>')
A(f'<rect x="{CX-70}" y="{yM0+96}" width="140" height="32" fill="{B_PALE}"/>')
T(CX, yM0 + 120, 'layer 16', SMALL, B_DARK, 400, 'middle')
dots(yM1 + 52)
A(f'<rect x="{BX0}" y="{yW0}" width="{BX1-BX0}" height="{yW1-yW0}" rx="8" fill="{P_PALE}" stroke="{P_MID}" stroke-width="2.5"/>')
T(CX, (yW0 + yW1) / 2 + 10, 'layers 1–12', LAB, P_DARK, 600, 'middle')
bar(yE0, yE1, 'embeddings')

# ---------------- panel A: tokens -> tokens ----------------
aY0, aY1 = PY0, 702
A(f'<rect x="{PX0}" y="{aY0}" width="{PX1-PX0}" height="{aY1-aY0}" rx="10" fill="{G_PALE}" stroke="{G_LINE}" stroke-width="3"/>')
T(PX0 + 26, aY0 + 46, 'READ TOKENS → WRITE TOKENS', SMALL + 1, G_DARK, 700, ls=2)
cw = (PX1 - PX0) / 4
items = [('calculator', 'exact + − × ÷', 0, False),
         ('eyes', 'CNN on pixel tokens', 0, True),
         ('IFEval rules', '×14, no fine-tune', 2, False),
         ('BBH tools', '×8', 2, False)]
cyA = aY0 + 142
for i, (n, sub, stk, tr) in enumerate(items):
    cx = PX0 + cw * (i + 0.5)
    chip(cx, cyA, 56, G_MID, stk)
    if tr:
        badge(cx + 40, cyA - 40)
    T(cx, cyA + 84, n, LAB, G_DARK, 700, 'middle')
    T(cx, cyA + 116, sub, SMALL, G_DARK, 400, 'middle')
T(PX0 + 26, aY1 - 66, 'BBH tools: track · order · interval · walk · truth ·', SMALL, G_DARK)
T(PX0 + 26, aY1 - 34, 'expr · calc · bool', SMALL, G_DARK)
# token write: panel A -> junction under the output head
A(f'<path d="M{PX0} {wy}H{CX+24}" stroke="{G_MID}" stroke-width="4" marker-end="url(#aG)"/>')

# ---------------- panel B: hidden -> tokens ----------------
bY0, bY1 = 730, 958
A(f'<rect x="{PX0}" y="{bY0}" width="{PX1-PX0}" height="{bY1-bY0}" rx="10" fill="{B_PALE}" stroke="{B_LINE}" stroke-width="3"/>')
T(PX0 + 26, bY0 + 46, 'READ HIDDEN STATES → WRITE TOKENS', SMALL + 1, B_DARK, 700, ls=2)
half = (PX1 - PX0) / 2
for k, (n, l1, l2) in enumerate([('mistake detector', '3 probe heads, layers 12–24', '→ “Hmm, wait…”'),
                                  ('topic blocker', 'probe at layer 16', '→ decline + end reply')]):
    cx, cy = PX0 + 70 + k * half, bY0 + 118
    chip(cx, cy, 56, B_MID)
    tx = cx + 62
    T(tx, cy - 22, n, LAB, B_DARK, 700)
    badge(cx - 46, cy + 62)
    T(tx, cy + 12, l1, SMALL, B_DARK)
    T(tx, cy + 44, l2, SMALL, B_DARK, 600)
# read: layers 12-24 -> panel B
A(f'<path d="M{BX1} {yR}H{PX0-6}" stroke="{B_MID}" stroke-width="3.5" stroke-dasharray="8 7" marker-end="url(#aB)"/>')
# write: panel B -> the same token-write line
A(f'<path d="M{PX0} {bY0+30}H{CH}V{wy+8}" fill="none" stroke="{B_MID}" stroke-width="4" marker-end="url(#aB)"/>')

# ---------------- panel C: tokens -> hidden ----------------
cY0, cY1 = 986, PY1
A(f'<rect x="{PX0}" y="{cY0}" width="{PX1-PX0}" height="{cY1-cY0}" rx="10" fill="{P_PALE}" stroke="{P_LINE}" stroke-width="3"/>')
T(PX0 + 26, cY0 + 46, 'READ TOKENS → WRITE HIDDEN STATES', SMALL + 1, P_DARK, 700, ls=2)
cx, cy = PX0 + 70, (yW0 + yW1) / 2
chip(cx, cy, 56, P_MID)
T(cx + 62, cy - 22, 'latent write', LAB, P_DARK, 700)
T(cx + 62, cy + 12, 'writes “There are 161 pens.” as activations,', SMALL, P_DARK)
T(cx + 62, cy + 44, 'not text (layer ≤ 12)', SMALL, P_DARK)
T(cx + 62, cy + 76, '→ the model uses it as its own thought', SMALL, P_DARK, 600)
A(f'<path d="M{PX0} {cy}H{BX1+14}" stroke="{P_MID}" stroke-width="4" marker-end="url(#aP)"/>')

# ---------------- token strip + token bus ----------------
TY, TH = OY1 + 52, 62
toks = [('…', 0), ('2595 + 3075 =', 1), (' 5670', 2), (' cookies', 1), ('…', 1)]
ws = [len(s) * 18 + 28 for s, _ in toks]
gaps = [10, 30, 30, 10]
x = 60
pos = []
for i, ((s, who), w) in enumerate(zip(toks, ws)):
    fill, stroke, col = [(BG, BORDER, MUTE), (BG, INK, INK), (G_LIGHT, G_MID, G_DARK)][who]
    A(f'<rect x="{x:.1f}" y="{TY}" width="{w:.1f}" height="{TH}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{3 if who == 2 else 2.2}"/>')
    T(x + w / 2, TY + 42, s.strip(), MID, col, 600 if who == 2 else 400, 'middle', MONO)
    pos.append((x, x + w))
    x += w + (gaps[i] if i < len(gaps) else 0)
A(f'<path d="M{CX} {TY-10}V{PY1+8}" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
# the same tokens also feed panels A and C (bus up the right edge)
bx, by = 1534, OY1 - 14
A(f'<path d="M{CX} {by}H{bx}V{aY0+150}H{PX1+8}" fill="none" stroke="{INK}" stroke-width="2.5" marker-end="url(#aI)"/>')
A(f'<path d="M{bx} {(yW0+yW1)/2}H{PX1+8}" fill="none" stroke="{INK}" stroke-width="2.5" marker-end="url(#aI)"/>')
A(f'<circle cx="{bx}" cy="{(yW0+yW1)/2}" r="5" fill="{INK}"/>')
A(f'<circle cx="{CX}" cy="{by}" r="5" fill="{INK}"/>')


def step(i0, i1, line, green=False):
    a, b = pos[i0][0], pos[i1][1]
    col = G_MID if green else INK
    A(f'<path d="M{a+4} {TY+TH+12}v10H{b-4}v-10" fill="none" stroke="{col}" stroke-width="2.5"/>')
    T((a + b) / 2, TY + TH + 58, line, LAB, G_DARK if green else INK, 600 if green else 400, 'middle')


step(1, 1, 'model')
step(2, 2, 'implant', True)
step(3, 4, 'model')

# ---------------- legend + footnote ----------------
LGX, LGY = 905, TY + 22
chip(LGX, LGY, 30, MUTE)
T(LGX + 32, LGY + 8, 'compiled circuit, written by Claude', SMALL, INK)
bw = badge(LGX - 17, LGY + 56)
T(LGX - 17 + bw + 12, LGY + 63, 'has a trained part (CNN, probes)', SMALL, INK)
FY = H - 82
T(48, FY, 'Implants never touch the model’s own weights; their readers and write directions are fitted to the host model.', SMALL - 2, MUTE)
T(48, FY + 30, 'Only the BBH tools and eyes also use a small fine-tune of the model (a few hundred examples at most) to', SMALL - 2, MUTE)
T(48, FY + 60, 'teach it the call format. Everything else runs on the untouched model.', SMALL - 2, MUTE)
A('</svg>')
open(OUT, 'w').write('\n'.join(o))
png = OUT.replace('.svg', '.png')
subprocess.run(['rsvg-convert', '-w', '1600', '-b', 'white', OUT, '-o', png], check=True)
print('wrote', png)
