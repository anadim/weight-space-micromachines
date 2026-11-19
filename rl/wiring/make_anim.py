"""Blog animations of the RL transition, from REAL logged values only (no smoothing, no interpolated dynamics).
(A) anim_competence_map: CWinOx s0 (calculator exact on x and ÷, wrong digits on + and −; trainable trigger in-wire).
    Left: per-operation share of the trigger's calls let through, rollout calls pooled over the last 10 updates (logged
    every update). Right: dev accuracy at the logged 25-update checkpoints (learned trigger), plus the same host with the
    implant off and with the hand trigger (calls everything) at the checkpoints where those were evaluated.
(B) anim_w3d_links: W3d s0 (distilled links, no hand write; links + host trained). Left: link effective strength
    ||diag(gate) W||_F per layer relative to update 1 (logged every update). Right: dev with links on vs links off
    (same host, links zeroed) at every 25-update checkpoint, and the gap.
Frames every 5 updates; the frame's update number is captioned. Writes .gif and .mp4 next to this script."""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, '..', 'results', 'wiring', 'figdata')
OUTD = os.path.join(HERE, '..', 'figures')                 # rendered figures live in rl/figures/
INK, INK2, GRID, GREEN, PALE, MID = '#222222', '#6b7280', '#e5e7eb', '#118659', '#e7f3ee', '#5fae8f'
for f in font_manager.findSystemFonts():
    if 'Avenir' in f:
        try: font_manager.fontManager.addfont(f)
        except Exception: pass
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': 11, 'axes.edgecolor': INK2, 'axes.linewidth': 0.8,
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})
load = lambda t: [json.loads(l) for l in open(os.path.join(FD, t + '.jsonl'))]
FRAMES = list(range(5, 301, 5)) + [300] * 12          # hold the last frame ~1.5 s
FPS = 8


def style(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def curve(rows, key, field='_acc'):
    out = {}
    for r in rows:
        if key in r:
            d = r[key]; t = [k for k in d if k.endswith(field)][0]; out[d['u']] = d[t]
    return dict(sorted(out.items()))


def save(anim, name, fig):
    anim.save(os.path.join(OUTD, name + '.gif'), writer=PillowWriter(fps=FPS), dpi=90)
    anim.save(os.path.join(OUTD, name + '.mp4'), writer=FFMpegWriter(fps=FPS, bitrate=1800), dpi=150)
    plt.close(fig); print('wrote', name, '.gif/.mp4')


# ---------------- (A) competence map ----------------
def anim_competence():
    rows = load('CWinOx_s0')
    OPS = ['+', '−', '×', '÷']; KEY = {'+': '+', '−': '-', '×': '*', '÷': '/'}
    per_u = {r['u']: r['iw_op'] for r in rows if 'iw_op' in r}
    dev = curve(rows, 'dev'); off = {}; hand = {}
    for k, dst in (('dev_host_curve', off), ('dev_hostonly', off), ('dev_cw_hand', hand)):
        dst.update(curve(rows, k))

    def fire(u):
        acc = {k: [0, 0] for k in OPS}
        for v in range(max(1, u - 9), u + 1):
            for k in OPS:
                c = per_u.get(v, {}).get(KEY[k], [0, 0]); acc[k][0] += c[0]; acc[k][1] += c[1]
        return [100 * acc[k][1] / max(1, acc[k][0]) for k in OPS]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.98, wspace=0.28)
    cap = fig.text(0.98, 0.95, '', ha='right', va='top', fontsize=15, fontweight='bold', color=GREEN)
    fig.text(0.07, 0.95, 'RL learns where the calculator can be trusted', ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(0.07, 0.885, 'calculator gives exact digits for × and ÷, wrong digits for + and −; only the trigger (1,025 params) and the host train',
             ha='left', va='top', fontsize=9, color=INK2)

    def draw(u):
        a.clear(); b.clear()
        v = fire(u); cols = [PALE, PALE, GREEN, GREEN]
        a.bar(range(4), v, 0.6, color=cols, edgecolor=GREEN, linewidth=0.8)
        for i, y in enumerate(v): a.text(i, y + 2, f'{y:.0f}%', ha='center', fontsize=10)
        a.set_xticks(range(4)); a.set_xticklabels([f'{o}\n{"broken" if i < 2 else "exact"}' for i, o in enumerate(OPS)])
        a.set_ylim(0, 105); a.set_ylabel('calls let through by the trigger (%)')
        a.set_title('rollout calls, last 10 updates', loc='left', fontsize=10, color=INK2)
        style(a)
        us = [x for x in dev if x <= u]
        b.plot(us, [dev[x] for x in us], '-o', color=GREEN, ms=4, lw=2, label='learned trigger')
        uo = [x for x in off if x <= u]; uh = [x for x in hand if x <= u]
        b.plot(uo, [off[x] for x in uo], 's', color=INK2, ms=6, label='no calculator (same host)')
        b.plot(uh, [hand[x] for x in uh], 'D', color=MID, mfc=PALE, ms=6, label='calls everything (same host)')
        b.set_xlim(0, 305); b.set_ylim(0, 80); b.set_xlabel('RL update'); b.set_ylabel('dev accuracy (%)')
        b.set_title('dev, evaluated every 25 updates', loc='left', fontsize=10, color=INK2)
        b.legend(frameon=False, fontsize=8.5, loc='upper left')
        style(b)
        cap.set_text(f'update {u}')
    anim = FuncAnimation(fig, lambda i: draw(FRAMES[i]), frames=len(FRAMES))
    for u in (5, 150, 300):                       # stills for the overlap check
        draw(u); fig.savefig(os.path.join(OUTD, f'anim_competence_map_u{u}.png'), dpi=90)
    save(anim, 'anim_competence_map', fig)


# ---------------- (B) W3d links ----------------
def anim_w3d():
    rows = load('W3d_s0')
    eff = {r['u']: r['link_eff'] for r in rows if 'reward' in r and 'link_eff' in r}
    L = ['calc@7', 'calc@14', 'calc@21']; base = eff[1]
    on = curve(rows, 'dev'); off = {}
    for k in ('dev_host_curve', 'dev_hostonly'): off.update(curve(rows, k))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.98, wspace=0.28)
    cap = fig.text(0.98, 0.95, '', ha='right', va='top', fontsize=15, fontweight='bold', color=GREEN)
    fig.text(0.07, 0.95, 'RL strengthens a calculator link the model already uses', ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(0.07, 0.885, 'links from the calculator into layers 7 / 14 / 21 (started from a short distillation; no hand-made write)',
             ha='left', va='top', fontsize=9, color=INK2)

    def draw(u):
        a.clear(); b.clear()
        g = [100 * (eff[u][k] / base[k] - 1) for k in L]
        a.bar(range(3), g, 0.55, color=GREEN)
        for i, (k, y) in enumerate(zip(L, g)): a.text(i, max(y, 0) + 0.8, f'+{y:.0f}%' if y >= 0 else f'{y:.0f}%', ha='center', fontsize=10)
        a.set_xticks(range(3)); a.set_xticklabels(['layer 7', 'layer 14', 'layer 21'])
        a.set_ylim(-2, 30); a.axhline(0, color=INK2, lw=0.8)
        a.set_ylabel('link strength vs update 1 (%)')
        a.set_title('||gate · W|| per layer, logged every update', loc='left', fontsize=10, color=INK2)
        style(a)
        us = [x for x in on if x <= u]; uo = [x for x in us if x in off]
        b.fill_between(uo, [off[x] for x in uo], [on[x] for x in uo], color=PALE)
        b.plot(us, [on[x] for x in us], '-o', color=GREEN, ms=4, lw=2, label='links on')
        b.plot(uo, [off[x] for x in uo], '-s', color=INK2, ms=4, lw=1.5, label='links off (same host)')
        if uo:
            x = uo[-1]; b.text(x - 3, (on[x] + off[x]) / 2, f'+{on[x] - off[x]:.1f}', ha='right', va='center', fontsize=11, color=GREEN, fontweight='bold')
        b.set_xlim(0, 305); b.set_ylim(30, 80); b.set_xlabel('RL update'); b.set_ylabel('dev accuracy (%)')
        b.set_title('dev, evaluated every 25 updates', loc='left', fontsize=10, color=INK2)
        b.legend(frameon=False, fontsize=9, loc='lower right')
        style(b)
        cap.set_text(f'update {u}')
    anim = FuncAnimation(fig, lambda i: draw(FRAMES[i]), frames=len(FRAMES))
    for u in (5, 150, 300):
        draw(u); fig.savefig(os.path.join(OUTD, f'anim_w3d_links_u{u}.png'), dpi=90)
    save(anim, 'anim_w3d_links', fig)


if __name__ == '__main__':
    anim_competence(); anim_w3d()
