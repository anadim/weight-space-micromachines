"""Implants figure: one model = frozen language model + micromachine chips. Big type.
Nine chips read the tokens and write into the top of the model (just under the output head);
rethink (dashed, partly future work) reads layer 28 directly.
Usage: python3 figures/make_implants_figure.py figures/0_implants.svg B && rsvg-convert -w 2600 figures/0_implants.svg -o figures/0_implants.png
  A: rethink sits LEFT of the model, short straight wire from layer 28.
  B (chosen): rethink sits between the model and the panel, under the write arrow.
"""
import sys, html

OUT = sys.argv[1]
VAR = sys.argv[2] if len(sys.argv) > 2 else 'B'
BG, BORDER, INK, MUTE, LAYER, OUTER = '#ffffff', '#cfd5dc', '#222222', '#6b7280', '#f1f3f5', '#fafbfc'
G_DARK, G_MID, G_LINE, G_LIGHT, G_PALE = '#03623f', '#118659', '#b2dac4', '#cee7d9', '#f3f9f5'
B_DARK, B_MID, B_LINE, B_PALE = '#1d4f73', '#2f74a8', '#bcd4e8', '#f1f6fb'
SANS = "'Avenir Next','Helvetica Neue',Helvetica,Arial,sans-serif"
MONO = "'SF Mono',Menlo,Consolas,monospace"
BIG, MID, SMALL = 64, 34, 27
esc = html.escape
W, H = 1960, 1250
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
     f'<defs><marker id="aI" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker>'
     f'<marker id="aG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="{G_MID}"/></marker>'
     f'<marker id="aB" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="{B_MID}"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="{BG}"/>']
A = o.append


def T(x, y, s, size=MID, fill=INK, weight=400, anchor='start', font=SANS, ls=0):
    sp = f' letter-spacing="{ls}"' if ls else ''
    A(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{sp}>{esc(s)}</text>')


def chip(cx, cy, S, dashed=False, col=None):
    col = col or G_MID
    h, P, st = S / 2, S * 0.2, S * 0.3
    pins = ''.join(f'M{cx+k*st} {cy-h}v-{P}M{cx+k*st} {cy+h}v{P}M{cx-h} {cy+k*st}h-{P}M{cx+h} {cy+k*st}h{P}' for k in (-1, 0, 1))
    A(f'<path d="{pins}" stroke="{col}" stroke-width="4" stroke-linecap="round" fill="none"/>')
    da = ' stroke-dasharray="7 6"' if dashed else ''
    A(f'<rect x="{cx-h}" y="{cy-h}" width="{S}" height="{S}" rx="{S*0.13}" fill="{BG}" stroke="{col}" stroke-width="4"{da}/>')
    d = S * 0.36
    A(f'<rect x="{cx-d/2}" y="{cy-d/2}" width="{d}" height="{d}" rx="4" fill="{col}"/>')


T(90, 120, 'One model with ten implants', BIG, INK, 700)

OX0, OX1, OY0, OY1 = 60, 1900, 292, 980
if VAR == 'A':
    RX0, RX1 = OX0 + 36, OX0 + 266            # rethink group (left)
    LX0, LX1 = RX1 + 64, 1060                  # language model
    MX0, MX1 = 1150, OX1 - 36                  # micromachines panel
else:
    LX0, LX1 = OX0 + 40, 900
    RX0, RX1 = 950, 1220
    MX0, MX1 = 1260, OX1 - 36
LCX = (LX0 + LX1) / 2
CX = LCX  # the main vertical flow goes through the language model

# next token, above the LM
A(f'<rect x="{CX-140}" y="164" width="280" height="72" rx="10" fill="{BG}" stroke="{INK}" stroke-width="2.5"/>')
T(CX, 212, 'next token', MID, INK, anchor='middle')

A(f'<rect x="{OX0}" y="{OY0}" width="{OX1-OX0}" height="{OY1-OY0}" rx="14" fill="{OUTER}" stroke="{INK}" stroke-width="2.5"/>')
T(OX0 + 36, OY0 + 50, 'ONE MODEL', SMALL, MUTE, 700, ls=3)

PY0, PY1 = OY0 + (34 if VAR == 'A' else 74), OY1 - 34
# ---------------- language model (the main thing) ----------------
A(f'<rect x="{LX0}" y="{PY0}" width="{LX1-LX0}" height="{PY1-PY0}" rx="12" fill="{BG}" stroke="{INK}" stroke-width="3"/>')
A(f'<path d="M{CX} {PY0}V248" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
T(LX0 + 32, PY0 + 60, 'LANGUAGE MODEL', MID, INK, 700, ls=3)
T(LX0 + 32, PY0 + 102, 'frozen, never trained', SMALL, MUTE)
BX0, BX1, BH = LX0 + 32, LX1 - 32, 48


def bar(y, label, fill=LAYER, stroke=INK, col=MUTE, weight=400):
    A(f'<rect x="{BX0}" y="{y}" width="{BX1-BX0}" height="{BH}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    T((BX0 + BX1) / 2, y + 34, label, SMALL, col, weight, 'middle')


yHead = PY0 + 140
bar(yHead, 'output head')
yGap = yHead + BH + 40          # write point sits in the middle of this gap
yL28 = yHead + BH + 80
bar(yL28, 'layer 28')
yL27 = yL28 + BH + 18
bar(yL27, 'layer 27')
# vertical dots
for k in range(3):
    A(f'<circle cx="{(BX0+BX1)/2}" cy="{yL27 + BH + 30 + k*18}" r="4" fill="{MUTE}"/>')
yL1 = yL27 + BH + 96
bar(yL1, 'layer 1')
wy = yHead + BH + 40            # write junction y
jx = (BX0 + BX1) / 2
# junction: small green plus-circle on the residual path
A(f'<path d="M{jx} {yL28}V{yHead+BH}" stroke="{INK}" stroke-width="2"/>')
A(f'<circle cx="{jx}" cy="{wy}" r="17" fill="{BG}" stroke="{G_MID}" stroke-width="3.5"/>')
A(f'<path d="M{jx-9} {wy}h18M{jx} {wy-9}v18" stroke="{G_MID}" stroke-width="3.5" stroke-linecap="round"/>')

# ---------------- micromachines panel ----------------
A(f'<rect x="{MX0}" y="{PY0}" width="{MX1-MX0}" height="{PY1-PY0}" rx="10" fill="{G_PALE}" stroke="{G_LINE}" stroke-width="3"/>')
T(MX0 + 28, PY0 + 60, 'MICROMACHINES', MID, G_DARK, 700, ls=3)
T(MX0 + 28, PY0 + 102, 'written, not trained · read tokens', SMALL, G_DARK)
names = ['calculator', 'sum', 'expression', 'truth tracker', 'navigator',
         'boolean', 'schedule', 'swap tracker', 'deduction']
cols = 3
cw = (MX1 - MX0 - 20) / cols
gy0 = PY0 + 196
rh = 142
for i, n in enumerate(names):
    r, c = divmod(i, cols)
    cx = MX0 + 10 + cw * (c + 0.5)
    cy = gy0 + r * rh
    chip(cx, cy, 56)
    T(cx, cy + 74, n, SMALL, G_DARK, 600, 'middle')

# write arrow: panel -> junction (just under the output head)
A(f'<path d="M{MX0} {wy}H{jx+24}" stroke="{G_MID}" stroke-width="4" marker-end="url(#aG)"/>')
wlx = (LX1 + MX0) / 2 if VAR == 'A' else (RX0 + RX1) / 2
A(f'<rect x="{wlx-44}" y="{wy-50}" width="88" height="34" fill="{OUTER}"/>')
T(wlx, wy - 22, 'write', SMALL, G_DARK, 600, 'middle')

# ---------------- rethink: reads layer 28 ----------------
ry = yL28 + BH / 2
if VAR == 'A':
    gx0, gx1 = RX0, RX1
    A(f'<rect x="{gx0}" y="{ry-130}" width="{gx1-gx0}" height="300" rx="10" fill="{G_PALE}" stroke="{G_LINE}" stroke-width="3" stroke-dasharray="10 8"/>')
    rcx = (gx0 + gx1) / 2
    chip(rcx, ry, 56, dashed=True)
    T(rcx, ry - 74, 'rethink', SMALL + 3, G_DARK, 700, 'middle')
    T(rcx, ry + 82, 'reads', SMALL, G_DARK, 400, 'middle')
    T(rcx, ry + 116, 'layer 28', SMALL, G_DARK, 400, 'middle')
    A(f'<path d="M{BX0} {ry}H{rcx+28+14}" stroke="{G_MID}" stroke-width="3.5" stroke-dasharray="8 7" marker-end="url(#aG)"/>')
else:
    rcx = (RX0 + RX1) / 2
    A(f'<rect x="{RX0}" y="{wy+18}" width="{RX1-RX0}" height="{ry+170-wy-18}" rx="10" fill="{B_PALE}" stroke="{B_LINE}" stroke-width="3"/>')
    chip(rcx, ry + 10, 56, dashed=True, col=B_MID)
    A(f'<path d="M{BX1} {ry+10}H{rcx-28-14}" stroke="{B_MID}" stroke-width="3.5" stroke-dasharray="8 7" marker-end="url(#aB)"/>')
    T(rcx, ry + 94, 'rethink', SMALL + 3, B_DARK, 700, 'middle')
    T(rcx, ry + 130, 'reads layer 28', SMALL - 2, B_DARK, 400, 'middle')

# ---------------- token strip ----------------
TY, TH = OY1 + 86, 70
toks = [('…', 0), ('2595 + 3075 =', 1), (' 5670', 2), (' cookies', 1), ('…', 1)]
ws = [len(s) * 20.4 + 32 for s, _ in toks]
gaps = [10, 36, 36, 10]
x = (W if VAR == 'A' else 2 * CX) / 2 - (sum(ws) + sum(gaps)) / 2
pos = []
for i, ((s, who), w) in enumerate(zip(toks, ws)):
    fill, stroke, col = [(BG, BORDER, MUTE), (BG, INK, INK), (G_LIGHT, G_MID, G_DARK)][who]
    A(f'<rect x="{x:.1f}" y="{TY}" width="{w:.1f}" height="{TH}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{3 if who == 2 else 2.2}"/>')
    T(x + w / 2, TY + 47, s.strip(), MID, col, 600 if who == 2 else 400, 'middle', MONO)
    pos.append((x, x + w))
    x += w + (gaps[i] if i < len(gaps) else 0)
# tokens feed up into the language model
A(f'<path d="M{CX} {TY-12}V{PY1+8}" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')


def step(i0, i1, line, green=False):
    a, b = pos[i0][0], pos[i1][1]
    col = G_MID if green else INK
    A(f'<path d="M{a+4} {TY+TH+14}v10H{b-4}v-10" fill="none" stroke="{col}" stroke-width="2.5"/>')
    T((a + b) / 2, TY + TH + 74, line, MID, G_DARK if green else INK, 600 if green else 400, 'middle')

step(1, 1, 'model')
step(2, 2, 'implant', True)
step(3, 4, 'model')
A('</svg>')
open(OUT, 'w').write('\n'.join(o))
print('wrote', OUT)
