"""Animated micromachine figure: the frozen LM generates tokens (grey neuron flicker per step);
when it writes '=', the calculator chip lights up, reads the tokens, adds, and writes one digit per
step into the residual stream just under the output head. Then it goes silent (adds exactly zero).
Same palette / type as make_implants_figure.py (variant B), rethink chip omitted.

Usage:
  python3 make_anim.py            # all frames -> figs/anim/frames, then GIF + MP4 + poster
  python3 make_anim.py keys       # only a few key frames as PNG (layout check)
Needs: rsvg-convert, ffmpeg, gifsicle (optional).
"""
import sys, os, html, random, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'figs', 'anim')
FR = os.path.join(OUTD, 'frames')
FPS = 12
PX_W = 1400  # rendered width in px

BG, BORDER, INK, MUTE, LAYER, OUTER = '#ffffff', '#cfd5dc', '#222222', '#6b7280', '#f1f3f5', '#fafbfc'
G_DARK, G_MID, G_LINE, G_LIGHT, G_PALE = '#03623f', '#118659', '#b2dac4', '#cee7d9', '#f3f9f5'
N_OFF, N_LV = '#e1e4e8', ['#bfc5cc', '#959ca6', '#6b7280']   # neuron squares: off, 3 grey levels
SANS = "'Avenir Next','Helvetica Neue',Helvetica,Arial,sans-serif"
MONO = "'SF Mono',Menlo,Consolas,monospace"
MID, SMALL, TOKF = 36, 30, 36
esc = html.escape
W, H = 1800, 1130

# ---------------- geometry ----------------
OX0, OX1, OY0, OY1 = 60, 1740, 140, 850
LX0, LX1 = OX0 + 40, 880
MX0, MX1 = 1000, OX1 - 36
PY0, PY1 = OY0 + 74, OY1 - 36
CX = (LX0 + LX1) / 2
BX0, BX1 = LX0 + 32, LX1 - 32
HB, LB = 52, 64                        # head bar / layer bar heights
yHead = PY0 + 130
wy = yHead + HB + 40                   # write junction (the plus-circle)
yL28 = yHead + HB + 80
yL27 = yL28 + LB + 14
yL1 = yL27 + LB + 84
jx = (BX0 + BX1) / 2
LAYERS = [(yL28, 'layer 28'), (yL27, 'layer 27'), (yL1, 'layer 1')]
NQ, NP, NC = 14, 19, 0                 # neuron square size, pitch
NX0 = BX0 + 160
NCOLS = int((BX1 - 18 - NX0 + (NP - NQ)) // NP)
NROWS = 2

names = ['calculator', 'sum', 'expression', 'truth tracker', 'navigator',
         'boolean', 'schedule', 'swap tracker', 'deduction']
cw = (MX1 - MX0 - 20) / 3
gy0, rh = wy, 142                      # calculator row sits on the write line
CHIPS = [(MX0 + 10 + cw * (c + 0.5), gy0 + r * rh) for r, c in (divmod(i, 3) for i in range(9))]
CS = 56
calc_x, calc_y = CHIPS[0]
PULSE_X0, PULSE_X1 = calc_x - CS / 2 - 14, jx + 24

# ---------------- tokens ----------------
# (text, who, line, glue)  who: 0 prompt, 1 model, 2 implant; glue = digit/punct that hugs the previous token
def num(s, who, line):
    return [(d, who, line, k > 0) for k, d in enumerate(s)]
L1 = ([('Sam', 0, 0, 0), ('baked', 0, 0, 0)] + num('2595', 0, 0) +
      [('cookies', 0, 0, 0), ('on', 0, 0, 0), ('Monday', 0, 0, 0), ('and', 0, 0, 0)] + num('3075', 0, 0) +
      [('on', 0, 0, 0), ('Tuesday', 0, 0, 0), ('.', 0, 0, 1)])
L2 = num('2595', 1, 1) + [('+', 1, 1, 0)] + num('3075', 1, 1) + [('=', 1, 1, 0)] + \
     num('5670', 2, 1) + [('cookies', 1, 1, 0), ('.', 1, 1, 1)]
TOKS = L1 + L2
CHW = TOKF * 0.602
PAD, GAP_W, GAP_G, TH = 12, 14, 4, 62
TY = [OY1 + 56, OY1 + 56 + TH + 18]
pos = []
widths = []
for li in (0, 1):
    xs, x = [], 0
    for t in TOKS:
        if t[2] != li:
            continue
        w = len(t[0]) * CHW + 2 * PAD
        if xs:
            x += GAP_G if t[3] else GAP_W
        xs.append((x, w))
        x += w
    widths.append(x)
    pos.append(xs)
SX = (W - max(widths)) / 2
TPOS = []                               # absolute (x, y, w) per token index
k = [0, 0]
for t in TOKS:
    li = t[2]
    x, w = pos[li][k[li]]
    k[li] += 1
    TPOS.append((SX + x, TY[li], w))
EXPR = [i for i, t in enumerate(TOKS) if t[2] == 1][:10]   # '2595 + 3075 =' (what the calculator reads)


def T(o, x, y, s, size=MID, fill=INK, weight=400, anchor='start', font=SANS, ls=0, op=1):
    sp = f' letter-spacing="{ls}"' if ls else ''
    oo = f' opacity="{op}"' if op != 1 else ''
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
             f'fill="{fill}" text-anchor="{anchor}"{sp}{oo}>{esc(s)}</text>')


def chip(o, cx, cy, S, lit=False, op=1.0):
    h, P, st = S / 2, S * 0.2, S * 0.3
    o.append(f'<g opacity="{op}">')
    if lit:
        o.append(f'<rect x="{cx-h-20}" y="{cy-h-20}" width="{S+40}" height="{S+40}" rx="22" fill="{G_MID}" opacity="0.45" filter="url(#glow)"/>')
    pins = ''.join(f'M{cx+k*st} {cy-h}v-{P}M{cx+k*st} {cy+h}v{P}M{cx-h} {cy+k*st}h-{P}M{cx+h} {cy+k*st}h{P}' for k in (-1, 0, 1))
    o.append(f'<path d="{pins}" stroke="{G_DARK if lit else G_MID}" stroke-width="4" stroke-linecap="round" fill="none"/>')
    o.append(f'<rect x="{cx-h}" y="{cy-h}" width="{S}" height="{S}" rx="{S*0.13}" fill="{G_MID if lit else BG}" stroke="{G_DARK if lit else G_MID}" stroke-width="4"/>')
    d = S * 0.36
    o.append(f'<rect x="{cx-d/2}" y="{cy-d/2}" width="{d}" height="{d}" rx="4" fill="{BG if lit else G_MID}"/>')
    o.append('</g>')


def render(st):
    """st: n (tokens shown), new (age in frames of the newest token), nextok (index shown in head box or None),
    neur (dict layer-> list of levels -1..2), calc (None|'read'|'compute'|'write'), pulse (0..1 or None),
    plus_flash (bool), silent_note (bool)"""
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<defs><marker id="aI" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker>'
         f'<marker id="aG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="{G_MID}"/></marker>'
         f'<filter id="glow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="12"/></filter></defs>',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    calc = st['calc']

    # next token box above the LM
    nt = st['nextok']
    if nt is None:
        o.append(f'<rect x="{CX-130}" y="30" width="260" height="70" rx="10" fill="{BG}" stroke="{BORDER}" stroke-width="2.5"/>')
        T(o, CX, 76, 'next token', SMALL, MUTE, anchor='middle')
    else:
        s, who = TOKS[nt][0], TOKS[nt][1]
        g = who == 2
        o.append(f'<rect x="{CX-130}" y="30" width="260" height="70" rx="10" fill="{G_LIGHT if g else BG}" stroke="{G_MID if g else INK}" stroke-width="{3 if g else 2.5}"/>')
        T(o, CX, 78, s, TOKF + 4, G_DARK if g else INK, 700 if g else 400, 'middle', MONO)

    o.append(f'<rect x="{OX0}" y="{OY0}" width="{OX1-OX0}" height="{OY1-OY0}" rx="14" fill="{OUTER}" stroke="{INK}" stroke-width="2.5"/>')
    T(o, OX0 + 36, OY0 + 50, 'ONE MODEL', SMALL, MUTE, 700, ls=3)

    # ---- language model ----
    o.append(f'<rect x="{LX0}" y="{PY0}" width="{LX1-LX0}" height="{PY1-PY0}" rx="12" fill="{BG}" stroke="{INK}" stroke-width="3"/>')
    o.append(f'<path d="M{CX} {PY0}V112" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
    T(o, LX0 + 32, PY0 + 60, 'LANGUAGE MODEL', MID, INK, 700, ls=3)
    T(o, LX0 + 32, PY0 + 102, 'Qwen3-0.6B · frozen', SMALL, MUTE)
    o.append(f'<rect x="{BX0}" y="{yHead}" width="{BX1-BX0}" height="{HB}" rx="8" fill="{LAYER}" stroke="{INK}" stroke-width="2"/>')
    T(o, CX, yHead + 36, 'output head', SMALL, MUTE, anchor='middle')
    for li, (y, lab) in enumerate(LAYERS):
        o.append(f'<rect x="{BX0}" y="{y}" width="{BX1-BX0}" height="{LB}" rx="8" fill="{LAYER}" stroke="{INK}" stroke-width="2"/>')
        T(o, BX0 + 20, y + LB / 2 + 11, lab, SMALL, MUTE)
        lv = st['neur'][li]
        y0 = y + (LB - NROWS * NQ - (NP - NQ)) / 2
        for r in range(NROWS):
            for c in range(NCOLS):
                v = lv[r * NCOLS + c]
                col = N_OFF if v < 0 else N_LV[v]
                o.append(f'<rect x="{NX0 + c*NP}" y="{y0 + r*NP:.1f}" width="{NQ}" height="{NQ}" rx="2.5" fill="{col}"/>')
    for k in range(3):
        o.append(f'<circle cx="{CX}" cy="{yL27 + LB + 26 + k*16}" r="4" fill="{MUTE}"/>')
    o.append(f'<path d="M{jx} {yL28}V{yHead+HB}" stroke="{INK}" stroke-width="2"/>')
    if st['plus_flash']:
        o.append(f'<circle cx="{jx}" cy="{wy}" r="30" fill="{G_MID}" opacity="0.4" filter="url(#glow)"/>')
        o.append(f'<circle cx="{jx}" cy="{wy}" r="19" fill="{G_MID}" stroke="{G_DARK}" stroke-width="3.5"/>')
        o.append(f'<path d="M{jx-10} {wy}h20M{jx} {wy-10}v20" stroke="{BG}" stroke-width="3.5" stroke-linecap="round"/>')
    else:
        o.append(f'<circle cx="{jx}" cy="{wy}" r="17" fill="{BG}" stroke="{G_MID}" stroke-width="3.5"/>')
        o.append(f'<path d="M{jx-9} {wy}h18M{jx} {wy-9}v18" stroke="{G_MID}" stroke-width="3.5" stroke-linecap="round"/>')

    # ---- micromachines panel ----
    o.append(f'<rect x="{MX0}" y="{PY0}" width="{MX1-MX0}" height="{PY1-PY0}" rx="10" fill="{G_PALE}" stroke="{G_LINE}" stroke-width="3"/>')
    T(o, MX0 + 28, PY0 + 60, 'MICROMACHINES', MID, G_DARK, 700, ls=3)
    if calc:
        x = MX0 + 28
        for k, wd in enumerate(['read', 'compute', 'write']):
            on = wd == calc
            T(o, x, PY0 + 104, wd, SMALL, G_DARK, 700 if on else 400, op=1 if on else 0.38)
            x += len(wd) * 17 + 18
            if k < 2:
                T(o, x, PY0 + 104, '→', SMALL, G_DARK, op=0.38)
                x += 50
    else:
        T(o, MX0 + 28, PY0 + 104, 'silent: adds exactly zero', SMALL, G_DARK, op=0.75)
    # write arrow (the calculator's own segment only drawn when it writes)
    wa = 1.0 if calc else 0.55
    o.append(f'<path d="M{MX0} {wy}H{jx+24}" stroke="{G_MID}" stroke-width="4" marker-end="url(#aG)" opacity="{wa}"/>')
    if calc:
        o.append(f'<path d="M{PULSE_X0} {wy}H{MX0}" stroke="{G_MID}" stroke-width="4"/>')
    wlx = (LX1 + MX0) / 2
    T(o, wlx, wy - 20, 'write', SMALL, G_DARK, 600, 'middle', op=1 if calc == 'write' else 0.6)
    for i, ((cx, cy), n) in enumerate(zip(CHIPS, names)):
        lit = i == 0 and bool(calc)
        chip(o, cx, cy, CS, lit=lit, op=1 if lit else 0.4)
        T(o, cx, cy + 70, n, SMALL, G_DARK, 700 if lit else 600, 'middle', op=1 if lit else 0.5)
    if st['pulse'] is not None:
        px = PULSE_X0 + (PULSE_X1 - PULSE_X0) * st['pulse']
        o.append(f'<circle cx="{px:.1f}" cy="{wy}" r="20" fill="{G_MID}" opacity="0.45" filter="url(#glow)"/>')
        o.append(f'<circle cx="{px:.1f}" cy="{wy}" r="10" fill="{G_MID}" stroke="{BG}" stroke-width="2.5"/>')

    # ---- token strip ----
    o.append(f'<path d="M{CX} {TY[0]-12}V{PY1+8}" stroke="{INK}" stroke-width="3" marker-end="url(#aI)"/>')
    # input bus: the same tokens also feed the micromachines panel
    ybus, mxc = OY1 + 26, CHIPS[0][0]
    on = calc == 'read'
    bus = f'M{CX} {ybus}H{mxc}V{PY1+8}'
    if on:
        o.append(f'<path d="{bus}" stroke="{G_MID}" stroke-width="12" fill="none" opacity="0.35" filter="url(#glow)"/>')
    o.append(f'<path d="{bus}" stroke="{G_MID}" stroke-width="{4 if on else 3}" fill="none" opacity="{1 if on else 0.45}" marker-end="url(#aG)"/>')
    if on and st.get('inp') is not None:
        L1_, L2_ = mxc - CX, ybus - (PY1 + 8)
        d = st['inp'] * (L1_ + L2_)
        px, py = (CX + d, ybus) if d < L1_ else (mxc, ybus - (d - L1_))
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="20" fill="{G_MID}" opacity="0.45" filter="url(#glow)"/>')
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="10" fill="{G_MID}" stroke="{BG}" stroke-width="2.5"/>')
    # clear the arrow behind the first strip line
    for i in range(st['n']):
        s, who, li, _ = TOKS[i]
        x, y, w = TPOS[i]
        g = who == 2
        fresh = i == st['n'] - 1 and st['new'] == 0
        fill, stroke, col = (G_LIGHT, G_MID, G_DARK) if g else (BG, INK, INK)
        if fresh and g:
            o.append(f'<rect x="{x-8:.1f}" y="{y-8}" width="{w+16:.1f}" height="{TH+16}" rx="14" fill="{G_MID}" opacity="0.4" filter="url(#glow)"/>')
        sw = 3 if g else (3 if fresh else 2)
        o.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{TH}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        T(o, x + w / 2, y + 43, s, TOKF, col, 700 if g else 400, 'middle', MONO)
    if calc == 'read':
        a, b = TPOS[EXPR[0]][0], TPOS[EXPR[-1]][0] + TPOS[EXPR[-1]][2]
        yb = TY[1] + TH + 10
        o.append(f'<path d="M{a+4:.1f} {yb}v10H{b-4:.1f}v-10" fill="none" stroke="{G_MID}" stroke-width="4"/>')
    T(o, W / 2, H - 22, 'the implant writes 5670; the model never knows it’s there', SMALL, MUTE, anchor='middle')
    o.append('</svg>')
    return '\n'.join(o)


# ---------------- timeline ----------------
def build():
    rng = random.Random(7)
    NN = NROWS * NCOLS
    frames = []
    neur_state = [[-1] * NN for _ in LAYERS]

    def fire():
        for li in range(len(LAYERS)):
            neur_state[li] = [(rng.choice([0, 1, 1, 2, 2]) if rng.random() < 0.32 else -1) for _ in range(NN)]

    def decay():
        for li in range(len(LAYERS)):
            nv = []
            for v in neur_state[li]:
                if v >= 0:
                    v = v - 1 if rng.random() < 0.45 else v      # fade slowly
                elif rng.random() < 0.025:
                    v = 0                                       # faint background flicker
                nv.append(v)
            neur_state[li] = nv

    st = dict(n=0, new=9, nextok=None, calc=None, pulse=None, plus_flash=False, inp=None)

    def snap(key=None):
        d = dict(st)
        d['neur'] = [list(x) for x in neur_state]
        d['key'] = key
        frames.append(d)

    for _ in range(6):                      # blank start
        snap()
    n1 = len(L1)
    for i in range(len(TOKS)):
        who = TOKS[i][1]
        if who == 2:
            # implant step: model still runs (flicker), calculator reads -> adds -> writes one digit
            for ph, nf in (('read', 2), ('compute', 2)):
                for f in range(nf):
                    if ph == 'read' and f == 0:
                        fire()
                    else:
                        decay()
                    st.update(calc=ph, pulse=None, plus_flash=False, new=st['new'] + 1,
                              inp=((0.3, 0.85)[f] if ph == 'read' else None))
                    snap()
            for f, p in enumerate((0.1, 0.45, 0.8)):
                decay()
                st.update(calc='write', pulse=p, new=st['new'] + 1)
                snap('poster' if (i == len(L1) + 11 and f == 1) else None)
            st.update(n=i + 1, new=0, nextok=i, pulse=None, plus_flash=True, calc='write')
            snap()
            st.update(plus_flash=False)
            continue
        nf = 2 if who == 0 else 3          # prompt prefill is quick, generated tokens slower
        last_digit_done = i > 0 and TOKS[i - 1][1] == 2
        for f in range(nf):
            if f == 0:
                fire()
                st.update(n=i + 1, new=0, nextok=(i if who == 1 else None),
                          calc=None if (last_digit_done or st['calc'] is None) else st['calc'], pulse=None)
            else:
                decay()
                st['new'] += 1
            snap()
        if TOKS[i][0] == '=':               # calculator wakes up
            st.update(calc='read')
    for _ in range(18):                      # hold ~1.5 s
        decay()
        st['new'] += 1
        snap()
    return frames


def main():
    keys_only = len(sys.argv) > 1 and sys.argv[1] == 'keys'
    os.makedirs(FR, exist_ok=True)
    frames = build()
    print(len(frames), 'frames', f'{len(frames)/FPS:.1f}s')
    if keys_only:
        picks = {'k_prefill': 20, 'k_gen': 50,
                 'k_poster': next(i for i, f in enumerate(frames) if f['key'] == 'poster'),
                 'k_end': len(frames) - 1}
        for name, i in picks.items():
            svg = os.path.join(FR, name + '.svg')
            open(svg, 'w').write(render(frames[i]))
            subprocess.run(['rsvg-convert', '-w', str(PX_W), svg, '-o', os.path.join(OUTD, name + '.png')], check=True)
        print('keys', picks)
        return
    for f in os.listdir(FR):
        os.remove(os.path.join(FR, f))
    svg = os.path.join(FR, 'tmp.svg')
    for i, fr in enumerate(frames):
        open(svg, 'w').write(render(fr))
        subprocess.run(['rsvg-convert', '-w', str(PX_W), '-b', 'white', svg, '-o', os.path.join(FR, f'f{i:04d}.png')], check=True)
        if fr['key'] == 'poster':
            open(os.path.join(OUTD, 'poster.svg'), 'w').write(render(fr))
            subprocess.run(['rsvg-convert', '-w', '2600', '-b', 'white', svg, '-o', os.path.join(OUTD, 'micromachine_poster.png')], check=True)
    os.remove(svg)
    gif = os.path.join(OUTD, 'micromachine.gif')
    pat = os.path.join(FR, 'f%04d.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', pat, '-vf',
                    'split[a][b];[a]palettegen=max_colors=96:stats_mode=full:reserve_transparent=0[p];'
                    '[b][p]paletteuse=dither=none:diff_mode=rectangle', '-loop', '0', gif], check=True)
    if shutil.which('gifsicle'):
        subprocess.run(['gifsicle', '-O3', '-b', gif], check=True)
    mp4 = os.path.join(OUTD, 'micromachine.mp4')
    # X: H.264 yuv420p; loop the clip twice so it plays ~20 s
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-stream_loop', '1', '-framerate', str(FPS), '-i', pat,
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '18', '-preset', 'slow', '-movflags', '+faststart', mp4], check=True)
    for p in (gif, mp4):
        print(p, f'{os.path.getsize(p)/1e6:.2f} MB')


if __name__ == '__main__':
    main()
