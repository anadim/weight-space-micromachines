"""v6 re-renders (2026-11-20) of four static blog figures from figures/blog.
Same data constants as the source scripts; layout-only changes. Writes to figures/blog/v6/<name>_v6.png.
  16_bbh27        : title/subtitle gap fixed (title placed in inches from top, not figure fraction); plot moved up
  18_ifeval_rules : same title fix
  23_bbh_curve27  : x tick labels padded (first label collided with the y '0'), tick font 16 -> 18
  6_eyes_v2       : '95 vs 4' becomes a bracket at 32 examples; x label says log scale; line labels near their lines
Usage: python3 wsm_static_v6.py [check]   (check = re-render the originals unchanged into figs_v6/_check for pixel diff)
"""
import os, sys, re
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')   # figures/blog
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')   # figures/blog/v6
sys.path.insert(0, SRC)
CHECK = len(sys.argv) > 1 and sys.argv[1] == 'check'
import glob
from matplotlib import font_manager as _fm
for _f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', '*.ttf')): _fm.fontManager.addfont(_f)   # Avenir Next ttf set (copied from blog_v0/fonts)
import make_figs as MF
import matplotlib.pyplot as plt

def save(fig, name):
    d = os.path.join(OUTD, '_check') if CHECK else OUTD
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, name if CHECK else name.replace('.png', '_v6.png'))
    fig.savefig(out, dpi=100); plt.close(fig); print('wrote', out)

_orig_title = MF.title
def title_in(fig, t, sub=None):
    """Same sizes as make_figs.title, but placed at the offsets (in inches) it has on a 9.5 in tall figure."""
    H = fig.get_size_inches()[1]
    fig.text(0.04, 1 - 0.57 / H, t, fontsize=38, fontweight='bold', va='top')
    if sub: fig.text(0.04, 1 - 1.28 / H, sub, fontsize=24, color=MF.INK2, va='top')

def run_script(fname, subs, use_title_in):
    src = open(os.path.join(SRC, fname)).read()
    if not CHECK:
        for a, b in subs:
            assert a in src, (fname, a); src = src.replace(a, b)
    MF.save = save; MF.title = title_in if (use_title_in and not CHECK) else _orig_title
    g = {'__name__': 'v6', '__file__': os.path.join(SRC, fname)}
    exec(compile(src, fname, 'exec'), g)

# 16_bbh27: axes top 0.83 -> 0.905
run_script('bbh27.py', [('fig.add_axes([0.33, 0.05, 0.63, 0.78])', 'fig.add_axes([0.33, 0.05, 0.63, 0.855])')], True)

# 23_bbh_curve27
run_script('bbh_curve27.py', [('ax.set_xticklabels(labs, fontsize=16, linespacing=1.0)', 'ax.set_xticklabels(labs, fontsize=18, linespacing=1.0)'),
                              ('ax.tick_params(axis="x", length=0)', 'ax.tick_params(axis="x", length=0, pad=14)')], False)

# 18_ifeval_rules_v2 (function inside make_figs3_v2)
import make_figs3_v2 as M3
M3.save = save
if not CHECK:
    M3.title = title_in
    import inspect
    s = inspect.getsource(M3.ifeval_rules)
    for a, b in [('0.07, 0.80, dy=0.032, bh=0.021)', '0.07, 0.845, dy=0.032, bh=0.021)'),
                 ('fig.add_axes([0.36, 0.04, 0.58, 0.66])', 'fig.add_axes([0.36, 0.04, 0.58, 0.70])')]:
        assert a in s; s = s.replace(a, b)
    exec(s, M3.__dict__)
M3.ifeval_rules()

# 6_eyes_v2 (function inside make_figs_v2)
import make_figs_v2 as M2
M2.save = save
if not CHECK:
    import inspect
    s = inspect.getsource(M2.eyes)
    for a, b in [('    ax.text(40, 89, "eyes + calculator implants", color=BLUE, fontsize=27, fontweight="bold")\n',
                  '    ax.text(55, 88, "eyes + calculator implants", color=BLUE, fontsize=27, fontweight="bold", ha="left", va="top")\n'),
                 ('    ax.text(8.6, 17, "no implants", color=INK2, fontsize=27, fontweight="bold")\n',
                  '    ax.text(300, 9, "no implants\\n(plain fine-tune)", color=INK2, fontsize=27, fontweight="bold", ha="left", va="bottom", linespacing=1.0)\n'),
                 ('    ax.annotate("95 vs 4", xy=(32, 50), fontsize=30, fontweight="bold", color=INK, ha="center")\n',
                  '    ax.annotate("", xy=(42, 92), xytext=(42, 7), arrowprops=dict(arrowstyle="<->", color=INK, lw=2.5))\n'
                  '    ax.text(50, 50, "95 vs 4\\nat 32 examples", fontsize=28, fontweight="bold", color=INK, va="center", linespacing=1.05)\n'),
                 ('ax.set_xlabel("fine-tuning questions", fontsize=23)', 'ax.set_xlabel("fine-tuning questions (log scale)", fontsize=23)')]:
        assert a in s, a; s = s.replace(a, b)
    exec(s, M2.__dict__)
M2.eyes()
