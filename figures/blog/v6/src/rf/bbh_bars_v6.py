"""v6 (2026-11-20) of review_final/figs/bars_v2.py:bbh(): adds the call-gate arm that the blog text quotes.
Gate numbers: bbh_generic/callgate/README.md, 'BBH 27-task table' (v2 + GATE, probe v3): 27 mean 54.58, 12 tool 85.63, 15 no-tool 29.74.
Other three arms unchanged from bars_v2.py. Run with 'check' to re-render the unchanged v2 figure for a pixel diff."""
import sys
from bars_v2 import *
DARK = '#0b5e3e'
if len(sys.argv) > 1 and sys.argv[1] == 'check':
    bbh(); sys.exit()
fig, ax = plt.subplots(figsize=SIZE)
style(ax, 'Eight implants on BIG-Bench Hard (Qwen3-0.6B, real BBH test)', 'accuracy (%)', (0, 112))
ax.yaxis.label.set_size(LB); ax.set_yticks([0, 20, 40, 60, 80, 100])
cols = [GREY, INK2, GREEN, DARK]
grouped(ax, [('12 tasks with an implant', [43.27, 38.10, 84.33, 85.63]), ('15 tasks with no implant', [39.15, 30.01, 19.58, 29.74]),
             ('all 27 tasks', [40.98, 33.61, 48.36, 54.58])], cols, 0.2)
legend(ax, cols, ['model alone', 'call fine-tune', 'fine-tune + implants', '+ call gate'])
save(fig, 'bbh_bars_v6')
