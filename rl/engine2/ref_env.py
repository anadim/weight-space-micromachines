"""Build the OLD engine exactly as train_library.py does (same file, executed up to its main loop), read-only.
We never modify train_library.py: its source is exec'd into a namespace with our argv. Returns that namespace
(eng, TOOLS, IFT, imp, links, INW, TNAMES, pool, devset, TESTSET, train_step, evaluate, opt, P32, ...)."""
import os, sys
_RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repo rl/
LIB = os.environ.get('E2_LIB', os.path.join(_RL, 'library'))                 # dir holding train_library.py (+ mixed/sft_host)
ENV = dict(RLENVS=os.path.join(LIB, 'mixed'), WSM_INTEG_DIR=os.path.join(_RL, '..', 'bbh_generic', 'integrate', 'code'),
           RLW_DIR=os.path.join(_RL, 'wiring'),
           HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
if os.environ.get('E2_SMALL_FLEX', '1') == '1': ENV['FLEX_SMALL'] = '1'      # A40-class smem; unset on H100
for k, v in ENV.items(): os.environ.setdefault(k, v)

M1_ARGS = ['--arm', 'LIB', '--task', 'mixed', '--host', f'{LIB}/mixed/sft_host', '--scramble', '0', '--plain', '0',
           '--G', '8', '--T', '1.0', '--lr', '1e-6', '--kl', '0', '--loss', 'drgrpo', '--max_new', '512', '--S_max', '896',
           '--dev_mixed_n', '400', '--iw_eps', '0.05', '--link_lr', '1e-3', '--link_eps', '1e-8', '--hf_check', '0']


def load(run_dir, extra=(), src_file=f'{LIB}/train_library.py', patch=None, inject=None):
    src = open(src_file).read()
    if patch is not None: src = patch(src)
    cut = src.index('# ------------------------------------------------------------------ main loop')
    sys.argv = [src_file] + M1_ARGS + ['--run_dir', run_dir] + list(extra)
    g = {'__name__': 'old_engine', '__file__': src_file}
    if inject: g.update(inject)
    exec(compile(src[:cut], src_file, 'exec'), g)
    return g
