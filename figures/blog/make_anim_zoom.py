"""Second animation: zoom INTO the calculator micromachine and watch it compute 2595 + 3075 = 5670.

Faithful (simplified) to calc/circuits.py + hdl.py:
  read     each token -> one symbol flag (reader = linear + clamp on the token embedding)
  stack 1  runs once at '=': parse A, B into digit registers; column sums A_i + B_i (linear);
           ReLU units G = [sum >= 10], P = [sum == 9]; carries by lookahead, all at once:
           carry_r = OR_j (G_j AND P_{j+1} .. P_{r-1}); digit = sum + carry - 10*[sum + carry >= 10]
  copy     one hard attention head copies the call state (result register) to later positions
  stack 2  at every output step: one-hot 'which digit next' -> select that digit -> one-hot output token,
           written into the host's last hidden state.
Same style as make_anim.py (imports its palette, fonts, chip icon and renderer for the intro).

Usage:
  python3 make_anim_zoom.py         # frames -> figs/anim/zoom_frames, GIF + MP4 + poster
  python3 make_anim_zoom.py keys    # a few key frames as PNG for layout checks
"""
import sys, os, random, subprocess, shutil, math
import make_anim as A
from make_anim import (BG, BORDER, INK, MUTE, LAYER, G_DARK, G_MID, G_LINE, G_LIGHT, G_PALE, N_OFF, N_LV,
                       SANS, MONO, MID, SMALL, TOKF, W, H, T, chip, esc)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'figs', 'anim')
FR = os.path.join(OUTD, 'zoom_frames')
FPS, PX_W = 12, 1400

# ---------------- the arithmetic (computed exactly like hdl.Net.resolve) ----------------
AS, BS = '2595', '3075'
Ad = [int(c) for c in AS[::-1]]          # LSB first
Bd = [int(c) for c in BS[::-1]]
COLS = [a + b for a, b in zip(Ad, Bd)]
Gg = [int(c >= 10) for c in COLS]
Pp = [int(c == 9) for c in COLS]
CARRY = [0]
TERMS = {}                               # (j, r) lookahead term G_j AND P_{j+1..r-1} -> value
for r in range(1, 5):
    v = 0
    for j in range(r):
        t = Gg[j] * all(Pp[k] for k in range(j + 1, r))
        TERMS[j, r] = int(t); v += t
    CARRY.append(v)
DIG = [(c + cr) % 10 for c, cr in zip(COLS, CARRY[:4])]
assert CARRY[4] == 0 and ''.join(map(str, DIG[::-1])) == str(int(AS) + int(BS)) == '5670'
# display rows: top = thousands (place 3) ... bottom = ones (place 0)
ROWP = [3, 2, 1, 0]

# ---------------- geometry (same 1800 x 1130 canvas as make_anim) ----------------
CX0, CX1, CY0, CY1 = 70, 1730, 64, 1020           # the chip outline
TTL_Y = 142
TY, TH = 186, 62
toks = list(AS) + ['+'] + list(BS) + ['='] + list('5670')
WHO = [0] * 10 + [2] * 4
TW = 46
TX = []
x = 130
for i, t in enumerate(toks):
    if i:
        glue = toks[i - 1].isdigit() and t.isdigit() and WHO[i] == WHO[i - 1]
        x += 4 if glue else 14
    TX.append(x); x += TW
FLAGS = list('0123456789') + ['+', '−', '×', '÷', '=']
FL_X0, FL_Y, FL_S, FL_P = 130, 274, 42, 48
flag_of = lambda t: FLAGS.index(t)

S1X0, S1X1, S2X0, S2X1 = 236, 1132, 1214, 1612   # stack boxes
BY0, BY1 = 384, 952
HDR_Y = 436
ROW_Y = [508, 612, 716, 820]                        # row centres
CS = 84
COLX = {'A': 302, 'B': 404, 'sum': 528, 'P': 660, 'G': 784, 'carry': 942, 'dig': 1062}
HDR = {'A': 'A', 'B': 'B', 'sum': 'sum', 'P': '= 9', 'G': '≥ 10', 'carry': 'carry', 'dig': 'digit'}
ORDER = ['A', 'B', 'sum', 'P', 'G', 'carry', 'dig']
STX, POSX, OUTX, OUTY = 1282, 1386, 1516, (ROW_Y[1] + ROW_Y[2]) / 2
WR_X0, WR_X1 = OUTX + 60, W - 6
UN_Y = 888                                           # mini ReLU-unit strips
N1, N2 = 16, 18                                      # layer columns drawn for stack 1 / stack 2


def cell(o, cx, cy, txt, state, s=CS, size=40):
    """state: 'off' (grey, faint text), 'pale' (held, light green), 'on' (green), 'flash' (solid green + glow), 'zero' (idle 0)."""
    h = s / 2
    if state == 'flash':
        o.append(f'<rect x="{cx-h-10}" y="{cy-h-10}" width="{s+20}" height="{s+20}" rx="16" fill="{G_MID}" opacity="0.45" filter="url(#gz)"/>')
        fill, st, col, sw = G_MID, G_DARK, BG, 3.5
    elif state == 'on':
        fill, st, col, sw = G_LIGHT, G_MID, G_DARK, 3.5
    elif state == 'pale':
        fill, st, col, sw = G_PALE, G_LINE, G_DARK, 3
    elif state == 'zero':
        fill, st, col, sw = LAYER, BORDER, '#9aa1aa', 2.5
    else:
        fill, st, col, sw = BG, BORDER, None, 2.5
    o.append(f'<rect x="{cx-h:.1f}" y="{cy-h:.1f}" width="{s}" height="{s}" rx="10" fill="{fill}" stroke="{st}" stroke-width="{sw}"/>')
    if txt is not None and col:
        T(o, cx, cy + size * 0.36, txt, size, col, 700 if state in ('on', 'flash') else 400, 'middle', MONO)


def units(o, x0, x1, n, lit, rng_seed):
    """n layer columns of 3x2 tiny ReLU units; lit[i] in {-1 off, 0 faint, 1 pale-green, 2 green}."""
    r = random.Random(rng_seed)
    pitch = (x1 - x0) / n
    q = 11
    for i in range(n):
        for rr in range(3):
            for cc in range(2):
                v = lit[i]
                on = v >= 1 and r.random() < (0.75 if v == 2 else 0.45)
                col = (G_MID if v == 2 else G_LINE) if on else (N_LV[0] if (v == 0 and r.random() < 0.2) else N_OFF)
                o.append(f'<rect x="{x0 + i*pitch + cc*(q+4):.1f}" y="{UN_Y + rr*(q+4)}" width="{q}" height="{q}" rx="2" fill="{col}"/>')


def chip_outline(o):
    o.append(f'<rect x="{CX0}" y="{CY0}" width="{CX1-CX0}" height="{CY1-CY0}" rx="34" fill="{G_PALE}" stroke="{G_MID}" stroke-width="5"/>')
    pins = []
    for k in range(12):
        px = CX0 + 120 + k * (CX1 - CX0 - 240) / 11
        pins.append(f'M{px:.0f} {CY0}v-26M{px:.0f} {CY1}v26')
    for k in range(7):
        py = CY0 + 110 + k * (CY1 - CY0 - 220) / 6
        pins.append(f'M{CX0} {py:.0f}h-26M{CX1} {py:.0f}h26')
    o.append(f'<path d="{"".join(pins)}" stroke="{G_LINE}" stroke-width="6" stroke-linecap="round" fill="none"/>')


def inside(st):
    """The chip's inside. st keys: ntok, newage, flag (index|None), phase, c1 (stack-1 progress 0..6, -1 none),
    held, copy, wstep (-1 or 0..3), wsub (0..5), nwritten."""
    o = []
    chip_outline(o)
    T(o, CX0 + 54, TTL_Y, 'INSIDE THE CALCULATOR', MID, G_DARK, 700, ls=3)
    ph = st['phase']
    x = CX1 - 54
    words = ['read', 'compute', 'write']
    ws = [len(w) * 17 for w in words]
    total = sum(ws) + 2 * (18 + 50)
    x = CX1 - 54 - total
    for k, wd in enumerate(words):
        on = wd == ph
        T(o, x, TTL_Y, wd, SMALL, G_DARK, 700 if on else 400, op=1 if on else 0.38)
        x += ws[k] + 18
        if k < 2:
            T(o, x, TTL_Y, '→', SMALL, G_DARK, op=0.38); x += 50

    # ---- tokens ----
    for i in range(st['ntok']):
        t = toks[i]; g = WHO[i] == 2
        fresh = i == st['ntok'] - 1 and st['newage'] < 3
        xx = TX[i]
        dy = -26 if (fresh and not g and st['newage'] == 0) else 0
        o.append(f'<g transform="translate(0 {dy})">')
        if fresh and g:
            o.append(f'<rect x="{xx-8}" y="{TY-8}" width="{TW+16}" height="{TH+16}" rx="14" fill="{G_MID}" opacity="0.4" filter="url(#gz)"/>')
        fill, stc, col = (G_LIGHT, G_MID, G_DARK) if g else (BG, INK, INK)
        sw = 3 if (g or fresh) else 2
        o.append(f'<rect x="{xx:.1f}" y="{TY}" width="{TW}" height="{TH}" rx="9" fill="{fill}" stroke="{stc}" stroke-width="{sw}"/>')
        T(o, xx + TW / 2, TY + 43, t, TOKF, col, 700 if g else 400, 'middle', MONO)
        o.append('</g>')
    # ---- symbol flags (reader) ----
    for k, s in enumerate(FLAGS):
        cx = FL_X0 + k * FL_P + FL_S / 2
        on = st['flag'] == k
        if on:
            o.append(f'<rect x="{cx-FL_S/2-7}" y="{FL_Y-7}" width="{FL_S+14}" height="{FL_S+14}" rx="12" fill="{G_MID}" opacity="0.45" filter="url(#gz)"/>')
        o.append(f'<rect x="{cx-FL_S/2}" y="{FL_Y}" width="{FL_S}" height="{FL_S}" rx="7" fill="{G_MID if on else BG}" stroke="{G_DARK if on else BORDER}" stroke-width="{3 if on else 2}"/>')
        T(o, cx, FL_Y + 31, s, 28, BG if on else '#9aa1aa', 700 if on else 400, 'middle', MONO)
    if st['flag'] is not None and st['ntok']:
        i = st['ntok'] - 1
        fx = FL_X0 + st['flag'] * FL_P + FL_S / 2
        o.append(f'<path d="M{TX[i]+TW/2} {TY+TH+4}L{fx} {FL_Y-6}" stroke="{G_MID}" stroke-width="3" opacity="0.8"/>')
    T(o, FL_X0 + len(FLAGS) * FL_P + 10, FL_Y + 31, 'flags', SMALL, MUTE)

    # ---- stack 1 ----
    c1 = st['c1']; held = st['held']
    act1 = c1 >= 0 or held
    o.append(f'<rect x="{S1X0}" y="{BY0}" width="{S1X1-S1X0}" height="{BY1-BY0}" rx="16" fill="{BG}" stroke="{G_MID if act1 else G_LINE}" stroke-width="3"/>')
    T(o, S1X0 + 24, BY0 - 14, 'stack 1', SMALL, G_DARK, 700, op=1 if c1 >= 0 else 0.6)
    for r, y in enumerate(ROW_Y):
        T(o, S1X0 - 16, y + 11, ['1000s', '100s', '10s', '1s'][r], SMALL, MUTE, anchor='end')
    for k, key in enumerate(ORDER):
        T(o, COLX[key], HDR_Y, HDR[key], SMALL, G_DARK if (c1 >= k or held) else MUTE, 600, 'middle')
    # lookahead wires G_j -> carry_r (drawn under the cells)
    if c1 >= ORDER.index('carry') or held:
        for (j, r), v in TERMS.items():
            if r > 3:
                continue
            y0 = ROW_Y[ROWP.index(j)]; y1 = ROW_Y[ROWP.index(r)]
            xa, xb = COLX['G'] + CS / 2, COLX['carry'] - CS / 2
            col, sw, op = (G_MID, 4.5, 1) if v else ('#c3c8ce', 2.5, 0.9)
            if c1 == ORDER.index('carry') and v:
                o.append(f'<path d="M{xa} {y0}C{xa+50} {y0} {xb-50} {y1} {xb} {y1}" stroke="{G_MID}" stroke-width="12" opacity="0.35" fill="none" filter="url(#gz)"/>')
            o.append(f'<path d="M{xa} {y0}C{xa+50} {y0} {xb-50} {y1} {xb} {y1}" stroke="{col}" stroke-width="{sw}" opacity="{op}" fill="none"/>')
    for r, y in enumerate(ROW_Y):
        p = ROWP[r]
        vals = {'A': Ad[p], 'B': Bd[p], 'sum': COLS[p], 'P': Pp[p], 'G': Gg[p], 'carry': CARRY[p], 'dig': DIG[p]}
        for k, key in enumerate(ORDER):
            v = vals[key]
            binary = key in ('P', 'G', 'carry')
            if c1 < k and not held:
                stt = 'off'
            else:
                fresh = c1 == k
                if binary and v == 0:
                    stt = 'zero'
                elif fresh:
                    stt = 'flash'
                else:
                    stt = 'pale' if held else 'on'
            txt = str(v)
            if key == 'carry' and p == 0:
                txt = '0'
            cell(o, COLX[key], y, txt, stt)
    lit1 = [-1] * N1
    if c1 >= 0:
        upto = int(round((c1 + 1) / len(ORDER) * N1))
        lit1 = [1 if i < upto - 3 else (2 if i < upto else 0) for i in range(N1)]
    elif held:
        lit1 = [-1] * N1
    units(o, S1X0 + 60, S1X1 - 40, N1, lit1, 100 + max(c1, 0) * 7 + st.get('seed', 0))

    # ---- copy (attention) ----
    ca = st['copy']
    ay = OUTY
    o.append(f'<path d="M{S1X1+6} {ay}H{S2X0-6}" stroke="{G_MID if ca else G_LINE}" stroke-width="{5 if ca else 3}" marker-end="url(#aGz)"/>')

    # ---- stack 2 ----
    ws_ = st['wstep']
    act2 = ca or ws_ >= 0
    o.append(f'<rect x="{S2X0}" y="{BY0}" width="{S2X1-S2X0}" height="{BY1-BY0}" rx="16" fill="{BG}" stroke="{G_MID if act2 else G_LINE}" stroke-width="3"/>')
    T(o, S2X0 + 24, BY0 - 14, 'stack 2', SMALL, G_DARK, 700, op=1 if ws_ >= 0 else 0.6)
    T(o, STX, HDR_Y, 'state', SMALL, G_DARK if ca else MUTE, 600, 'middle')
    T(o, POSX, HDR_Y, 'next', SMALL, G_DARK if ws_ >= 0 else MUTE, 600, 'middle')
    for r, y in enumerate(ROW_Y):
        p = ROWP[r]
        cell(o, STX, y, str(DIG[p]), ('on' if ws_ == r else 'pale') if ca else 'off', s=72, size=36)
        on = ws_ == r
        s = 44
        if on:
            o.append(f'<rect x="{POSX-s/2-8}" y="{y-s/2-8}" width="{s+16}" height="{s+16}" rx="12" fill="{G_MID}" opacity="0.45" filter="url(#gz)"/>')
        o.append(f'<rect x="{POSX-s/2}" y="{y-s/2}" width="{s}" height="{s}" rx="7" fill="{G_MID if on else (LAYER if ws_ >= 0 else BG)}" stroke="{G_DARK if on else BORDER}" stroke-width="{3 if on else 2}"/>')
        xa, xb = POSX + s / 2 + 4, OUTX - 54
        o.append(f'<path d="M{xa} {y}C{xa+50} {y} {xb-50} {OUTY} {xb} {OUTY}" stroke="{G_MID if on else "#d5d9de"}" stroke-width="{4 if on else 2.5}" fill="none"/>')
    # output one-hot / token box
    out_txt = None; ost = 'off'
    if ws_ >= 0 and st['wsub'] >= 1:
        out_txt = str(DIG[ROWP[ws_]]); ost = 'flash' if st['wsub'] in (1, 2) else 'on'
    cell(o, OUTX, OUTY, out_txt, ost, s=100, size=48)
    # write arrow out of the chip
    wa = ws_ >= 0
    o.append(f'<path d="M{OUTX+54} {OUTY}H{WR_X1}" stroke="{G_MID}" stroke-width="5" opacity="{1 if wa else 0.45}" marker-end="url(#aGz)"/>')
    T(o, (S2X1 + CX1) / 2, OUTY - 20, 'write', SMALL, G_DARK, 600, 'middle', op=1 if wa else 0.55)
    if st.get('pulse') is not None:
        px = OUTX + 60 + (WR_X1 - 30 - OUTX - 60) * st['pulse']
        o.append(f'<circle cx="{px:.1f}" cy="{OUTY}" r="22" fill="{G_MID}" opacity="0.45" filter="url(#gz)"/>')
        o.append(f'<circle cx="{px:.1f}" cy="{OUTY}" r="11" fill="{G_MID}" stroke="{BG}" stroke-width="2.5"/>')
    lit2 = [-1] * N2
    if ws_ >= 0:
        f = st['wsub']
        lit2 = [2 if (i < (f + 1) * 3 + 2 and i >= (f + 1) * 3 - 4) else (1 if i < (f + 1) * 3 - 4 else 0) for i in range(N2)]
    units(o, S2X0 + 36, S2X1 - 24, N2, lit2, 300 + ws_ * 11 + st['wsub'])
    return o


def defs():
    return (f'<defs><marker id="aGz" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M0 0L10 5L0 10z" fill="{G_MID}"/></marker>'
            f'<filter id="gz" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="10"/></filter></defs>')


INTRO = None


def intro_svg():
    global INTRO
    if INTRO is None:
        fr = A.build()
        f = next(f for f in fr if f['calc'] == 'read')
        s = A.render(f).split('\n')
        INTRO = '\n'.join(s[1:-1])         # drop the outer <svg> and </svg>
    return INTRO


def ease(t): return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


def render(st):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">', defs(),
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    z = st.get('zoom')                       # None: inside only; else 0..1 zoom progress
    if z is not None:
        e = ease(z)
        S = 30 ** e
        cx, cy = A.calc_x, A.calc_y
        tx = cx + (W / 2 - cx) * e; ty = cy + (H / 2 - cy) * e
        op_out = 1 - ease((z - 0.5) / 0.35)
        if op_out > 0.01:
            o.append(f'<g opacity="{op_out:.3f}" transform="translate({tx:.2f} {ty:.2f}) scale({S:.4f}) translate({-cx} {-cy})">')
            o.append(intro_svg())
            o.append('</g>')
        op_in = ease((z - 0.6) / 0.4)
        if op_in > 0.01:
            s2 = 0.9 + 0.1 * ease((z - 0.6) / 0.4)
            o.append(f'<g opacity="{op_in:.3f}" transform="translate({W/2} {H/2}) scale({s2:.4f}) translate({-W/2} {-H/2})">')
            o += inside(st)
            o.append('</g>')
    else:
        o += inside(st)
        T(o, W / 2, H - 30, 'every ReLU weight a small integer · exact', SMALL, MUTE, anchor='middle')
    o.append('</svg>')
    return '\n'.join(o)


def build():
    frames = []
    st = dict(ntok=0, newage=9, flag=None, phase='read', c1=-1, held=False, copy=False, wstep=-1, wsub=0, pulse=None, zoom=None)

    def snap(key=None, **kw):
        st.update(kw); d = dict(st); d['key'] = key; frames.append(d)

    for f in range(12):                                  # ~1 s on the first animation's look
        snap(zoom=0.0)
    for f in range(13):                                  # zoom in
        snap(zoom=(f + 1) / 13)
    st['zoom'] = None
    snap(); snap()
    for i in range(10):                                  # read: one token per step, its flag lights
        snap(ntok=i + 1, newage=0, flag=flag_of(toks[i]), phase='read')
        snap(newage=1)
    snap(newage=3, flag=None)
    st['phase'] = 'compute'
    for k in range(len(ORDER)):                          # stack 1, layer group by layer group
        n = 4 if ORDER[k] == 'carry' else 2
        for f in range(n):
            snap('k_carry' if (ORDER[k] == 'carry' and f == 1) else None, c1=k, seed=f)
    snap(c1=len(ORDER))
    snap(c1=-1, held=True, copy=True)                    # attention copies the call state
    snap(); snap()
    st['phase'] = 'write'
    for r in range(4):                                   # stack 2: one digit per step
        for f in range(6):
            pulse = {2: 0.15, 3: 0.55, 4: 0.92}.get(f)
            snap('poster' if (r == 1 and f == 3) else None, wstep=r, wsub=f, pulse=pulse)
        snap(ntok=11 + r, newage=0, pulse=None, wsub=5)
        snap(newage=1)
    snap(newage=3, wstep=-1, wsub=0)
    for _ in range(18):                                   # hold ~1.5 s
        snap('end')
    return frames


def png(svg_text, path, w=PX_W):
    tmp = path + '.svg'
    open(tmp, 'w').write(svg_text)
    subprocess.run(['rsvg-convert', '-w', str(w), '-b', 'white', tmp, '-o', path], check=True)
    os.remove(tmp)


def main():
    keys_only = len(sys.argv) > 1 and sys.argv[1] == 'keys'
    frames = build()
    print(len(frames), 'frames', f'{len(frames)/FPS:.1f}s')
    if keys_only:
        idx = lambda k: next(i for i, f in enumerate(frames) if f['key'] == k)
        picks = {'z_intro': 0, 'z_zoom': 18, 'z_zoom2': 21, 'z_zoom3': 23, 'z_read2': 38, 'z_read': 40, 'z_carry': idx('k_carry'),
                 'z_poster': idx('poster'), 'z_end': len(frames) - 1}
        for name, i in picks.items():
            png(render(frames[i]), os.path.join(OUTD, name + '.png'))
        print('keys', picks)
        return
    os.makedirs(FR, exist_ok=True)
    for f in os.listdir(FR):
        os.remove(os.path.join(FR, f))
    for i, fr in enumerate(frames):
        png(render(fr), os.path.join(FR, f'f{i:04d}.png'))
        if fr['key'] == 'poster':
            png(render(fr), os.path.join(OUTD, 'zoom_poster.png'), 2600)
    gif = os.path.join(OUTD, 'zoom.gif')
    pat = os.path.join(FR, 'f%04d.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', pat, '-vf',
                    'split[a][b];[a]palettegen=max_colors=96:stats_mode=full:reserve_transparent=0[p];'
                    '[b][p]paletteuse=dither=none:diff_mode=rectangle', '-loop', '0', gif], check=True)
    if shutil.which('gifsicle'):
        subprocess.run(['gifsicle', '-O3', '-b', gif], check=True)
    mp4 = os.path.join(OUTD, 'zoom.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-stream_loop', '1', '-framerate', str(FPS), '-i', pat,
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '18', '-preset', 'slow', '-movflags', '+faststart', mp4], check=True)
    for p in (gif, mp4):
        print(p, f'{os.path.getsize(p)/1e6:.2f} MB')


if __name__ == '__main__':
    main()
