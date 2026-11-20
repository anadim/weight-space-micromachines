"""Review re-render of 'RL strengthens a calculator link the model already uses' (orig: rl_wiring/make_anim.py
anim_w3d -> rl_wiring/anim_w3d_links.gif, 990x413 px, 9-15 px text). Same REAL logged values (rl_wiring/figdata/W3d_s0.jsonl):
left = link strength ||diag(gate) W|| per layer vs update 1 (every update); right = dev links on vs links off (same host) at
every 25-update checkpoint. Changes: 1600x700, blog sizes (22/18/16 pt), one short title per panel, direct labels
instead of the legend, subtitle lines dropped. Frames every 5 updates, last frame held ~1.5 s."""
import os, json, subprocess
from blogstyle import *
from matplotlib.animation import FuncAnimation, PillowWriter
FD = os.path.join(HERE, '..', '..', '..', 'rl', 'results', 'wiring', 'figdata')
rows = [json.loads(l) for l in open(os.path.join(FD, 'W3d_s0.jsonl'))]
FRAMES = list(range(5, 301, 5)) + [300] * 12; FPS = 8


def curve(key):
    out = {}
    for r in rows:
        if key in r:
            d = r[key]; t = [k for k in d if k.endswith('_acc')][0]; out[d['u']] = d[t]
    return dict(sorted(out.items()))


eff = {r['u']: r['link_eff'] for r in rows if 'reward' in r and 'link_eff' in r}
L = ['calc@7', 'calc@14', 'calc@21']; base = eff[1]
on = curve('dev'); off = {}
for k in ('dev_host_curve', 'dev_hostonly'): off.update(curve(k))
fig, (a, b) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw=dict(width_ratios=[1, 1.25]))
fig.subplots_adjust(top=0.86, bottom=0.14, left=0.07, right=0.975, wspace=0.25)
cap = fig.text(0.975, 0.965, '', ha='right', va='top', fontsize=TITLE, fontweight='bold', color=GREEN)


def draw(u):
    a.clear(); b.clear()
    g = [100 * (eff[u][k] / base[k] - 1) for k in L]
    a.bar(range(3), g, 0.58, color=GREEN, zorder=3)
    for i, y in enumerate(g):
        a.text(i, max(y, 0) + 0.8, f'+{y:.0f}%' if y >= 0 else f'{y:.0f}%', ha='center', va='bottom', fontsize=VAL + 1, fontweight='bold')
    a.set_xticks(range(3)); a.set_xticklabels(['layer 7', 'layer 14', 'layer 21']); a.tick_params(axis='x', length=0)
    style(a, 'The links get stronger', 'link strength vs update 1 (%)', (-2, 28))
    a.axhline(0, color=INK2, lw=1.5)
    us = [x for x in on if x <= u]; uo = [x for x in us if x in off]
    b.fill_between(uo, [off[x] for x in uo], [on[x] for x in uo], color=PALE, zorder=1)
    b.plot(us, [on[x] for x in us], '-o', color=GREEN, ms=8, lw=4, zorder=3)
    b.plot(uo, [off[x] for x in uo], '-o', color=GREY, ms=7, lw=4, zorder=2)
    if uo:
        x = uo[-1]
        gx, gha = (x - 6, 'right') if x >= 60 else (x + 110, 'left')
        b.text(gx, (on[x] + off[x]) / 2, f'+{on[x] - off[x]:.1f}', ha=gha, va='center', fontsize=VAL + 2, color=GREEN, fontweight='bold')
        b.text(x + 5, on[x], 'links on', ha='left', va='center', fontsize=VAL, color=GREEN, fontweight='bold')
        b.text(x + 5, off[x], 'links off', ha='left', va='center', fontsize=VAL, color=INK2, fontweight='bold')
    b.set_xlim(0, 345); b.set_xticks([0, 50, 100, 150, 200, 250, 300]); b.set_xlabel('RL update')
    style(b, 'And they carry more of the score', 'dev accuracy (%)', (40, 76))
    cap.set_text(f'update {u}')


for u in (5, 150, 300):
    draw(u); fig.savefig(os.path.join(HERE, f'anim_w3d_links_u{u}.png'), dpi=DPI)
draw(300); fig.savefig(os.path.join(HERE, 'anim_w3d_links.svg'))
anim = FuncAnimation(fig, lambda i: draw(FRAMES[i]), frames=len(FRAMES))
gif = os.path.join(HERE, 'anim_w3d_links.gif')
anim.save(gif, writer=PillowWriter(fps=FPS), dpi=DPI)
subprocess.run(['gifsicle', '-O3', '--colors', '64', '-b', gif], check=False)
print('wrote', gif, os.path.getsize(gif) // 1024, 'KB')
