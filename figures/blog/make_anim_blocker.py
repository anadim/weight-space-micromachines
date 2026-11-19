"""Animated topic-blocker implant ("never do math", frozen Qwen3-1.7B chat, thinking off). Same look as make_anim_detector.py.

Mechanism shown (topic project fc.py, condition probe_text_stop):
  * gate = a linear (logistic) probe on hidden_states[16] at the last prompt token, threshold 0.112 (probe_math_q17.json).
  * if it fires, solved output directions write "I can't help with math problems." and then <|im_end|>: the reply ends.
  * if it does not fire, every hook returns None and the host's logits are bit-identical: it answers normally.

Real data: figs/anim/blocker_trace.json
  beat 1: held-out test prompt (data/test_hand.jsonl, math/pos/indirect) "Pretend you're my private calculus tutor ...";
          logged reply under the blocker = "I can't help with math problems." (results/q17_math_examples.txt lines 3-5).
  beat 2: held-out test prompt (math/hard/casual) "ugh my daughter literally cries over her math hw every night, how do i
          motivate her"; the reply shown is the plain host's greedy reply (first 28 of 120 tokens), regenerated once on A100 node
          (identical to the implant run because the blocker is silent there).
  Probe scores were not stored per item in the local results; recomputed once on A100 node with the same probe weights
  (figs/anim/anim_scores_job.py): 0.99975 (fires) and 0.00009 (silent).
  A third beat (the generation-time monitor cutting off a reply) is left out: the logged example ("$73 haircut") is also
  caught by the prompt probe (score 0.23 > 0.112), so it would not show the monitor doing anything the probe did not.
Drawn simplifications: the chat-template tokens around the user message are not drawn; the probe reads at the last
prompt token (the end of the template), shown as the last user token.

Usage:
  python3 make_anim_blocker.py          # all frames -> figs/anim/frames_blocker, then GIF + MP4 + poster
  python3 make_anim_blocker.py keys     # a few key frames as PNG (layout check)
"""
import sys, os, json
import make_anim as A
import make_anim_detector as D
from make_anim import T, chip, BG, INK, MUTE, G_DARK, G_MID, G_LINE, G_PALE, MID, SMALL
from make_anim_detector import (W, OUTD, FPS, MX0, MX1, PY0, PY1, wy, jx, CS, WR_X, DET_X, DET_Y, MTR_X0, MTR_Y,
                                XS, TOK_Y0, TH, LP, TOKF, along, ease)

TR = json.load(open(os.path.join(OUTD, 'blocker_trace.json')))
EX = TR['examples']
POS = next(e for e in EX if e['prompt'].startswith('Pretend'))
NEG = next(e for e in EX if 'daughter' in e['prompt'])
assert POS['fires'] and not NEG['fires']
THR = POS['thr']
NREPLY = NEG['reply_pieces'].index(' way') + 2                # "... to feel this way,"


def beat_data(e, reply, who_r):
    users = D.layout_tokens(e['user_pieces'], [0] * len(e['user_pieces']), XS, TOK_Y0)
    yr = users[-1][3] + LP + 14
    reps = D.layout_tokens(reply, who_r, XS, yr)
    return dict(e=e, users=users, reps=reps, items=users + reps, nu=len(users), yr=yr)


B = [beat_data(POS, TR['phrase_pieces'] + [' <|im_end|>'], [2] * (len(TR['phrase_pieces']) + 1)),
     beat_data(NEG, NEG['reply_pieces'][:NREPLY], [1] * NREPLY)]
H = int(max(b['items'][-1][3] for b in B) + TH + 60 + 70)
LAB = ['layer 28', 'layer 16', 'layer 1']


def render(st):
    b = B[st['beat']]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">', D.defs(),
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    T(o, 60, 72, 'A topic-blocker implant', 46, INK, 700)
    T(o, 60, 116, 'rule: never do math · frozen Qwen3-1.7B · held-out test prompts', SMALL, MUTE)
    D.draw_nextok(o, st['nextok'])
    D.draw_model(o, st, 'Qwen3-1.7B · frozen', LAB, 1)

    # ---- implant panel ----
    o.append(f'<rect x="{MX0}" y="{PY0}" width="{MX1-MX0}" height="{PY1-PY0}" rx="10" fill="{G_PALE}" stroke="{G_LINE}" stroke-width="3"/>')
    T(o, MX0 + 28, PY0 + 60, 'TOPIC BLOCKER', MID, G_DARK, 700, ls=3)
    T(o, MX0 + 28, PY0 + 104, st['status'] or 'silent: adds exactly zero', SMALL, G_DARK,
      700 if st['status'] else 400, op=1 if st['status'] else 0.75)
    wact = st['wact']
    o.append(f'<path d="M{MX0} {wy}H{jx+24}" stroke="{G_MID}" stroke-width="4" marker-end="url(#aG)" opacity="{1 if wact else 0.5}"/>')
    o.append(f'<path d="M{WR_X-CS/2-14} {wy}H{MX0}" stroke="{G_MID}" stroke-width="4" opacity="{1 if wact else 0.45}"/>')
    T(o, (A.LX1 + MX0) / 2, wy - 20, 'write', SMALL, G_DARK, 600, 'middle', op=1 if wact else 0.6)
    chip(o, WR_X, wy, CS, lit=st['wr_on'], op=1 if st['wr_on'] else 0.55)
    T(o, WR_X + CS / 2 + 30, wy + 11, 'decline writer', SMALL, G_DARK, 700 if st['wr_on'] else 600, op=1 if st['wr_on'] else 0.6)
    rd = st['read']
    gx0, gx1 = D.neuron_glyph(o, DET_X, DET_Y, st['det_on'], 1 if (st['det_on'] or rd) else 0.55)
    rl = f'M{A.BX1} {DET_Y}H{gx0-12}'
    if rd:
        o.append(f'<path d="{rl}" stroke="{G_MID}" stroke-width="12" fill="none" opacity="0.3" filter="url(#glow)"/>')
    o.append(f'<path d="{rl}" stroke="{G_MID}" stroke-width="{4 if rd else 3}" fill="none" marker-end="url(#aG)" opacity="{1 if rd else 0.45}"/>')
    T(o, (A.LX1 + MX0) / 2, DET_Y - 20, 'read', SMALL, G_DARK, 600, 'middle', op=1 if rd else 0.6)
    T(o, DET_X, DET_Y + 84, 'probe', SMALL, G_DARK, 700 if st['det_on'] else 600, 'middle', op=1 if st['det_on'] else 0.6)
    o.append(f'<path d="M{gx1+12} {DET_Y}H{MTR_X0-14}" stroke="{G_MID}" stroke-width="3" marker-end="url(#aG)" opacity="{0.9 if st["det_on"] else 0.4}"/>')
    gon = st['gate']
    gp = f'M{DET_X} {DET_Y-48}V{wy+CS/2+22}'
    if gon:
        o.append(f'<path d="{gp}" stroke="{G_MID}" stroke-width="12" fill="none" opacity="0.3" filter="url(#glow)"/>')
    o.append(f'<path d="{gp}" stroke="{G_MID}" stroke-width="{4 if gon else 3}" fill="none" marker-end="url(#aG)" opacity="{1 if gon else 0.4}" stroke-dasharray="{"none" if gon else "7 7"}"/>')
    T(o, DET_X + 16, (DET_Y - 48 + wy + CS / 2 + 22) / 2 + 10, 'gate', 24, G_DARK, 600, op=1 if gon else 0.55)
    D.draw_meter(o, st, 'is it math?', 0, 1, THR, f'threshold {THR:.2f}', lambda v: f'{v:.4f}', ends=('', ''))
    T(o, MTR_X0, MTR_Y + 104, 'read once, at the end of the prompt', 24, MUTE, 600)

    # ---- tokens ----
    qop = st['qop']
    o.append(f'<g opacity="{qop:.2f}">')
    T(o, 60, TOK_Y0 + 36, 'user', SMALL, MUTE, 600)
    if st['n'] > b['nu']:
        T(o, 60, b['yr'] + 36, 'model', SMALL, MUTE, 600)
    o.append('</g>')
    D.draw_tokens(o, b['items'], st['n'], st['new'], qop, st['hl'])
    if st['more']:
        s, who, x, y, w, _ = b['items'][-1]
        T(o, x + w + 16, y + 36, '…', TOKF, MUTE, 700, font=A.MONO, op=qop)
    T(o, W / 2, H - 30, st['cap'], SMALL, MUTE, anchor='middle')
    for (x, y) in st['dots']:
        D.dot(o, x, y)
    o.append('</svg>')
    return '\n'.join(o)


def build():
    neur, fire, decay = D.make_neur(11)
    frames = []
    st = dict(beat=0, n=0, new=9, nextok=None, read=False, det_on=False, gate=False, wr_on=False, wact=False,
              plus_flash=False, meter=None, mval=False, status=None, qop=1.0, hl=None, more=False, dots=[], cap='')

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
        e = b['e']; nu = b['nu']; items = b['items']
        first = bi == 0
        st.update(beat=bi, n=0, nextok=None, hl=None, meter=None, mval=False, more=False, qop=1.0,
                  cap='the user’s message streams in' if first else 'a message that mentions math but asks for something else')
        for _ in range(3):
            snap()
        for j in range(nu):
            tok(j, 1)
        # read at the end of the prompt
        st.update(hl=nu - 1, read=True, status='read layer 16', cap='at the last prompt token, a linear probe reads layer 16')
        P = [(A.BX1, DET_Y), (DET_X - 70, DET_Y)]
        for p in (0.3, 0.65, 1.0):
            st['dots'] = [along(P, p)]; snap()
        st.update(dots=[], det_on=True, status='probe: is this math?')
        nf = 9 if first else 7
        for f in range(nf):
            st.update(meter=e['score'] * ease((f + 1) / nf), mval=True)
            snap('k_meter' if (first and f == nf - 1) else None)
        if first:
            st.update(status='gate open: decline and stop',
                      cap=f'score {e["score"]:.4f} is above the threshold {THR:.2f}: this is a math request')
            for _ in range(9):
                snap()
            st.update(gate=True, wr_on=True, cap='the implant writes the decline itself …')
            Pg = [(DET_X, DET_Y - 48), (DET_X, wy + CS / 2 + 16)]
            for p in (0.35, 0.75, 1.0):
                st['dots'] = [along(Pg, p)]; snap()
            st.update(dots=[], wact=True, read=False, det_on=False, hl=None)
            Pw = [(WR_X - CS / 2 - 14, wy), (jx + 24, wy)]
            for k, j in enumerate(range(nu, len(items))):
                txt = items[j][0]
                last = j == len(items) - 1
                st['status'] = 'write <|im_end|>' if last else f'write “{txt}”'
                if last:
                    st['cap'] = '… and ends the reply'
                for p in ((0.2, 0.55, 0.9) if k == 0 else (0.5,)):
                    st['dots'] = [along(Pw, p)]; snap()
                st.update(dots=[], plus_flash=True)
                tok(j, 2 if (k == 0 or last) else 1, nextok=(txt, True))
                st['plus_flash'] = False
                if k == 0:
                    for _ in range(3):
                        snap()
                if j == len(items) - 2:
                    snap('poster')
            st.update(gate=False, wr_on=False, wact=False, nextok=None, status=None,
                      cap='the reply is over: the model never starts on the integral')
            for _ in range(20):
                snap('k_end')
        else:
            st.update(cap=f'score {e["score"]:.4f}, far below the threshold: the implant stays silent')
            for _ in range(11):
                snap('k_quiet')
            st.update(read=False, det_on=False, hl=None, status=None, cap='and the model answers normally; nothing was added')
            for j in range(nu, len(items)):
                tok(j, 1, nextok=(items[j][0], False))
            st.update(nextok=None, more=True)
            for _ in range(18):
                snap('k_end2')
            st.update(meter=None, mval=False)
        for q in (0.75, 0.5, 0.25, 0.0):
            st['qop'] = q; snap()
        st.update(n=0, more=False)
    return frames


D.render_fn[0] = render

if __name__ == '__main__':
    frames = build()
    print(len(frames), 'frames', f'{len(frames)/FPS:.1f}s', 'H', H, POS['score'], NEG['score'])
    fr_dir = os.path.join(OUTD, 'frames_blocker')
    if len(sys.argv) > 1 and sys.argv[1] == 'keys':
        D.keys(frames, fr_dir, extra=[('k_stream', 20), ('k_stream2', len(frames) - 40)])
    else:
        D.render_all(frames, 'blocker', fr_dir)
