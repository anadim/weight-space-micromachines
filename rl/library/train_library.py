"""RL over HOW the calculator is wired into Qwen3-0.6B-Base (and over the writer weights). Dr.GRPO, fast engine.

Arms (all share arm-2's recipe/seeds so they compare directly):
  1   no implant
  2   implant attached, frozen, hand-wired end write (before final norm)        [reference]
  R   deranged calculator, frozen, hand-wired end write                         [reference]
  W2  arm-2 end write KEPT + trainable zero-init links (implant one-hot e) into host residual at layers 7,14,21
  W3z organic: NO end write, only trainable links, gate ZERO-init  (step0 == arm 1)
  W3r organic: NO end write, only trainable links, gate ONE-init, small-random W (links live from step 0)
  W1  arm-2 wiring + trainable WRITER (Dp = out_dirs(x)gain, 14x1024); stacks/reader frozen (hard one-hot -> no grad)
  W4  W1 + W2 (trainable writer AND trainable links)

Only the writer (Dp) is trainable in W1/W4: the implant's reader and ReLU stacks are an exact hard one-hot circuit,
piecewise-constant in their own weights, so GRPO gets no usable gradient there; the continuous write directions are the
live surface (and the one that can break calculator exactness). We track "writer exactness drift" on 2000 probes.
"""
import os, sys, time, json, math, random, argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
INTEG = os.environ.get('WSM_INTEG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bbh_generic', 'integrate', 'code'))
sys.path.insert(0, INTEG)
import integ                                         # registers Team A/B/C specs (same micromachines copy as wsmrepo)
sys.path.insert(0, os.environ.get('RLW_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wiring')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from lib_engine import LibEngine, BBHTool, IFTool, _lnorm

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='1', choices=['1', '2', 'R', 'W2', 'W3z', 'W3r', 'W1', 'W4', 'W3e', 'W3e3', 'W2w', 'W4w', 'CW', 'CWlean', 'CWout', 'CWin', 'W3d', 'W2ws', 'CWinR', 'CWinP', 'CWinC', 'CWinO', 'LIB'])
ap.add_argument('--run_dir', required=True)
ap.add_argument('--updates', type=int, default=64)
ap.add_argument('--prompts', type=int, default=64); ap.add_argument('--G', type=int, default=8)
ap.add_argument('--T', type=float, default=1.0); ap.add_argument('--max_new', type=int, default=320)
ap.add_argument('--lr', type=float, default=1e-6)
ap.add_argument('--link_lr', type=float, default=1e-3)
ap.add_argument('--impl_lr', type=float, default=1e-5, help='writer (Dp) lr for W1/W4')
ap.add_argument('--kl', type=float, default=0.0); ap.add_argument('--loss', default='drgrpo', choices=['drgrpo', 'cispo'])
ap.add_argument('--clip', type=float, default=0.2); ap.add_argument('--cispo_max', type=float, default=4.0)
ap.add_argument('--tok_budget', type=int, default=16384)
ap.add_argument('--link_layers', default='7,14,21', help="shorthand for --link_spec calc:7,calc:14,calc:21")
ap.add_argument('--link_spec', default=None,
                help="general links: comma list of SRC:LAYER (LAYER int or 'final'); SRC = a key of the implant feature "
                     "dict (now: 'calc' = calc_s4 one-hot e). Several SRCs may target one layer (their links are summed).")
ap.add_argument('--link_std', type=float, default=0.01)
ap.add_argument('--dev_every', type=int, default=25)
ap.add_argument('--dev_extra', default='64', help='extra dev evals at these updates (comma list)')
ap.add_argument('--dev_n', type=int, default=800)
ap.add_argument('--ckpt_every', type=int, default=100)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--B_max', type=int, default=256)
ap.add_argument('--cw_lr', type=float, default=None, help='co-designed wires lr (default = link_lr)')
ap.add_argument('--cw_hidden', type=int, default=None, help='64 (CW/CWout/CWin), 32 (CWlean)'); ap.add_argument('--cw_c', type=float, default=0.1)
ap.add_argument('--cw_mask', type=int, default=0, help='1: out-wires act only where the implant emits (e != 0); 0 = as designed')
ap.add_argument('--cw_l1', type=float, default=1e-4, help='L1 on out-wire gates (x mean g at firing positions)')
ap.add_argument('--iw_eps', type=float, default=0.0, help='in-wire exploration floor: p(fire) = eps + (1-2eps) sigmoid')
ap.add_argument('--iw_layer', type=int, default=14, help='in-wire reads the host residual at this layer input')
ap.add_argument('--init_links', default='', help='links state file to start from (W3d: SFT-distilled W3e links)')
ap.add_argument('--calib_traces', default='runs/traces_A_scaled.jsonl', help='W2ws gain calibration traces')
ap.add_argument('--soft_p', type=float, default=0.9, help='W2ws: target mean p(written token) at firing positions')
ap.add_argument('--call_cost', type=float, default=None, help='reward penalty per fired calculator call (CWinC: 0.002)')
ap.add_argument('--partial_thresh', type=int, default=5, help='CWinP: calls with an operand of >= this many digits are deranged')
ap.add_argument('--eval_ckpt', default='', help='checkpoint dir: load host/links/in-wire, run test evals, exit')
ap.add_argument('--test_sets', default='test,hard')
ap.add_argument('--test_n', type=int, default=0, help='limit test items (smoke only); 0 = all')
ap.add_argument('--interp', type=int, default=0, help='with --eval_ckpt: in-wire logit vs operand size / op / position')
ap.add_argument('--link_eps', type=float, default=1e-8, help='Adam eps for link / writer params')
ap.add_argument('--eval_all', type=int, default=0, help='1: at every dev point also eval wiring-reset and implant-off')
ap.add_argument('--freeze_host', type=int, default=0, help='1: host params frozen (only links / writer train)')
ap.add_argument('--sft', type=int, default=0, help='>0: supervised CE on gold solutions for this many steps (no RL)')
ap.add_argument('--sft_traces', default='', help='jsonl from gen_traces.py (q, resp token ids): SFT targets');
ap.add_argument('--sft_n', type=int, default=2000); ap.add_argument('--sft_kind', default='scaled')
ap.add_argument('--diag', type=int, default=0, help='1: in-update check that zeroing link gates changes the train forward')
ap.add_argument('--no_eval', type=int, default=0)
ap.add_argument('--gpu_tag', default='', help='launcher process tag (no effect)')
ap.add_argument('--warmup', type=int, default=5); ap.add_argument('--fused', type=int, default=1)
ap.add_argument('--tools', default=','.join(integ.TOOLS_V2), help='LIB: BBH machines attached (integ.TOOLS_V2)')
ap.add_argument('--ifeval', type=int, default=1, help='LIB: attach the IFEval 14-rule circuit')
ap.add_argument('--scramble', type=int, default=0, help='LIB: derange every non-calculator tool output')
ap.add_argument('--impl_train', type=int, default=0, help='arm C: every BBH implant weight trainable (exact init, soft writer)')
ap.add_argument('--impl_lr_reader', type=float, default=1e-4); ap.add_argument('--impl_lr_core', type=float, default=1e-4)
ap.add_argument('--impl_lr_writer', type=float, default=1e-3); ap.add_argument('--impl_ste', type=int, default=0)
ap.add_argument('--impl_freeze', default='', help='e.g. expr_gen:core')
ap.add_argument('--_dummy_after_freeze', type=int, default=0); ap.add_argument('--impl_gain', type=float, default=0.0, help='0: calibrate')
ap.add_argument('--task', default='gsm', choices=['gsm', 'mixed']); ap.add_argument('--host', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--S_max', type=int, default=544); ap.add_argument('--plain', type=int, default=0, help='1: tools attached but never on (plain RL arm)')
ap.add_argument('--dev_mixed_n', type=int, default=400)
ap.add_argument('--no_test', type=int, default=0); ap.add_argument('--bench', type=int, default=0)
ap.add_argument('--dump_ref', type=int, default=0); ap.add_argument('--misfire_probe', type=int, default=0)
ap.add_argument('--probe_tool', default='', help='TOOL:SUBPREFIX, e.g. order_gen:order (u0 LOO on those test items)')
ap.add_argument('--deadline', default='', help='HH:MM UTC; at u5 cut prompts/update so the run (incl. final evals) fits')
ap.add_argument('--hf_check', type=int, default=1, help='u1: |logp sampler - logp HF teacher-forced| on sampled tokens')
ap.add_argument('--loo_at', default='', help='LIB: leave-one-out dev evals at these updates (default: final)')
ap.add_argument('--ifdir', default=os.environ.get('IFDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ifeval', 'heldout')))
a = ap.parse_args()
os.makedirs(a.run_dir, exist_ok=True)
json.dump(vars(a), open(f'{a.run_dir}/config.json', 'w'), indent=1)
random.seed(a.seed); torch.manual_seed(a.seed)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False

# ---- arm -> config ----
HAS_IMPL = a.arm != '1'
HAS_LINKS = a.arm in ('LIB', 'W2', 'W3z', 'W3r', 'W4', 'W3e', 'W3e3', 'W2w', 'W4w', 'CW', 'CWlean', 'CWout', 'CWin', 'W3d', 'W2ws')
CW_ARM = a.arm in ('LIB', 'CW', 'CWlean', 'CWout', 'CWin', 'CWinR', 'CWinP', 'CWinC', 'CWinO')
HAS_INWIRE = a.arm in ('LIB', 'CW', 'CWlean', 'CWin', 'CWinR', 'CWinP', 'CWinC', 'CWinO')
if a.call_cost is None: a.call_cost = 0.002 if a.arm == 'CWinC' else 0.0
if a.cw_hidden is None: a.cw_hidden = 32 if a.arm in ('CWlean', 'LIB') else 64
ap_soft_target = 0.9
if CW_ARM or a.arm in ('W3d', 'W2ws'):          # coordinator: wires default to lr 1e-2, Adam eps 1e-12 (1e-3 barely moves them)
    if '--link_lr' not in sys.argv: a.link_lr = 1e-2
    if '--link_eps' not in sys.argv: a.link_eps = 1e-12
if a.arm == 'W3d': assert a.init_links, 'W3d needs --init_links (SFT-distilled W3e links)'
# default link specs per arm: SRC:LAYER[:INIT[:GATE]], INIT in {rand (W~N(0,std)), emb (digit d -> host input
# embedding of token d, rescaled to that layer's residual norm at digit positions), warm (W = hand-wired write D^T)}
DEFAULT_SPEC = {'W3e': 'calc:7:emb:0.1,calc:14:emb:0.1,calc:21:emb:0.1',
                'W3d': 'calc:7:emb:0.1,calc:14:emb:0.1,calc:21:emb:0.1',     # + --init_links (distilled)
                # soft warm start: write directions = out_dirs, trainable log-gain calibrated to p(written token) ~ 0.9
                'W2ws': 'calc:final:soft:1',
                'W3e3': 'calc:7:emb:0.3,calc:14:emb:0.3,calc:21:emb:0.3',
                # the hand-made write itself as a trainable link (no separate fixed write); step 0 == arm 2
                'W2w': 'calc:final:warm:1',
                # ... plus embedding-init links from the same implant state into the middle (RL may move the write)
                'W4w': 'calc:final:warm:1,calc:7:emb:0.1,calc:14:emb:0.1,calc:21:emb:0.1',
                # co-designed wiring (on W2w): adapters g(h) * B relu(A e + c) at 7/14/21/final, B = 0 at step 0
                # as designed (width 64, 7/14/21/final, ~287K wiring params = 0.048% of host); CWlean = budget follow-up
                'CW': 'calc:final:warm:1,calc:7:cw,calc:14:cw,calc:21:cw,calc:final:cw',
                'CWlean': 'calc:final:warm:1,calc:14:cw,calc:final:cw',
                'CWout': 'calc:final:warm:1,calc:7:cw,calc:14:cw,calc:21:cw,calc:final:cw',
                'CWin': 'calc:final:warm:1'}
TOOL_NAMES = [x for x in a.tools.split(',') if x] if a.arm == 'LIB' else []
if a.arm == 'LIB':          # CW-lean wiring per tool: out-adapters at L14 + final (zero-init B) + in-wire; calc keeps CW-lean's write
    DEFAULT_SPEC['LIB'] = ','.join(['calc:final:warm:1', 'calc:14:cw', 'calc:final:cw'] + [f'{n}:14:cw,{n}:final:cw' for n in TOOL_NAMES]
                                   + (['ifeval_prev:14:cw', 'ifeval:final:cw'] if a.ifeval else []))
USE_WRITE = a.arm in ('2', 'R', 'W2', 'W1', 'W4', 'CWinR', 'CWinP', 'CWinC', 'CWinO')
#   CWinO: exact on * and /, deranged on + and - (competence map orthogonal to usefulness-by-size)
# in-wire-only "RL designs the interface" arms: FIXED hand write + trainable in-wire (no out-wires, writer frozen)
#   CWinR: deranged-digit calculator (arm R implant) | CWinP: deranged only when an operand has >= 5 digits
#   CWinC: correct calculator, reward - call_cost per fired call
GATE0 = 1.0 if a.arm == 'W3r' else 0.0
TRAIN_IMPL = a.arm in ('W1', 'W4')
IS_R = a.arm in ('R', 'CWinR')

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(a.host)
MIXED = a.task == 'mixed'
if MIXED:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mixed'))
    import mixed_task as MT
    pre, QF = MT.prefix_and_qfn(tok); shot_q = set()
else:
    pre, shot_q = C.prefix_ids(tok)
Lp = len(pre)
_QC = {}
def QID(r):
    if not MIXED: return C.q_ids(tok, r['q'])
    k = r['id'] if 'id' in r else r['user']
    if k not in _QC: _QC[k] = QF(r)
    return _QC[k]
def SCORE(t, r):
    if not MIXED: return C.score(tok, t, r['gold'])
    txt = tok.decode(t, skip_special_tokens=False)
    rw, b, f = MT.score(txt, r)
    return rw, float('answer is' in txt.lower() or r['type'] == 'ifeval')
P32 = Q.load_hf_params(name=a.host, dtype=torch.float32)
for v in P32.values(): v.requires_grad_(not a.freeze_host)
names = list(P32)
W = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16)
W_ref = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16) if a.kl > 0 else None
imp = None
if HAS_IMPL:
    hf = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.bfloat16).to(dev).eval()
    imp = CalcImplant(hf, tok, a.B_max, fused=bool(a.fused), random=('digits' if IS_R else None))
    RESID_NORM = None
    if HAS_LINKS and 'emb' in (a.link_spec or DEFAULT_SPEC.get(a.arm, '')):
        # typical residual norm at each layer's INPUT, at positions holding a digit token (8-shot prefix)
        from micromachines.machine import first as _f0
        _ids = imp.A.all_ids[imp.M.name]
        dig = torch.tensor([_f0(_ids[s]) for s in imp.A.all_ids[imp.M.name] if isinstance(s, str) and s.isdigit()], device=dev)
        with torch.no_grad():
            pid = torch.tensor(pre, device=dev)[None]
            hs = hf(pid, output_hidden_states=True).hidden_states            # hs[l] = input of layer l (l < NL)
            m = torch.isin(pid[0], dig); m[:8] = False                        # skip attention-sink positions
            RESID_NORM = {l: float(hs[l][0, m].float().norm(dim=-1).median()) for l in range(Q.NL)}
        print('resid norm @digit positions:', {l: round(RESID_NORM[l], 1) for l in (0, 7, 14, 21)}, 'n_pos', int(m.sum()), flush=True)
    TOOLS, IFT = [], None
    E0 = None
    if a.arm == 'LIB':
        _env_ids = [tok.convert_tokens_to_ids(t) for t in ('<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>')]
        _Er = hf.get_input_embeddings().weight.detach()[_env_ids].float()
        _shared = bool((_Er[1:] - _Er[0]).abs().max() < 1e-6)
        print('envelope token rows shared (untrained):', _shared, flush=True)
        if _shared:                    # base checkpoint: the 4 envelope tokens share one untrained row -> Team C's distinct
            with torch.no_grad():      # random rows IN THE TOOLS' SNAPSHOT ONLY. NOT on a host whose rows are already
                integ.prep_call_tokens(hf, tok)   # distinct (instruct): the tied head would then be solved against rows
                                                  # the real host does not have, and <tool_response> writes would not win
        E0 = hf.get_input_embeddings().weight.detach().clone()          # FROZEN snapshot the BBH readers were solved on
        veto_bbh = integ.VETO if os.path.exists(integ.VETO) else None
        A_bbh = integ.attach_all(hf, tok, tuple(TOOL_NAMES), veto=veto_bbh)
        A_bbh.detach()
        g_ = torch.Generator().manual_seed(4242)
        for n in TOOL_NAMES:
            M_ = A_bbh.nets[n]
            for t_ in list(M_.parameters()) + list(M_.buffers()): t_.requires_grad_(False)
            perm = None
            if a.scramble:                                        # derangement of the tool's output rows
                no = M_.out_dirs.shape[0]
                while True:
                    p_ = torch.randperm(no, generator=g_)
                    if not (p_ == torch.arange(no)).any(): break
                perm = p_.to(dev)
            TOOLS.append(BBHTool(n, M_, a.B_max, perm=perm))
        print('BBH tools:', [(t.name, t.n_out, t.K, t.Wn, t.dS) for t in TOOLS], 'veto', veto_bbh, flush=True)
        if a.ifeval:
            import importlib.util
            _saved_common = sys.modules.pop('common')                 # Team D's frozen common.py shadows rl_wiring's
            _saved_tables = sys.modules.pop('tables', None)           # ... and Team A's tables.py shadows Team D's
            assert os.path.exists(os.path.join(a.ifdir, 'tables.py')), f'IFEval frozen dir missing: {a.ifdir}'
            sys.path.insert(0, a.ifdir)
            spec_ = importlib.util.spec_from_file_location('circuit_g', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'circuit_g.py'))
            CG = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(CG)
            hf_if = AutoModelForCausalLM.from_pretrained(os.environ.get('IF_HOST', 'Qwen/Qwen3-0.6B'), dtype=torch.bfloat16).to(dev).eval()
            circ = CG.Circuit(AutoTokenizer.from_pretrained(os.environ.get('IF_HOST', 'Qwen/Qwen3-0.6B')), hf_if, vecs=next(p_ for p_ in (os.environ.get('IF_VECS', ''), os.path.join(a.ifdir, 'vecs2.pt'), os.path.join(a.ifdir, 'vecs2_forcefix.pt')) if p_ and os.path.exists(p_)))   # repo ships the force_fix rows (only part used)
            circ.E = circ.E.detach().clone(); circ.g = circ.g.detach().clone()      # frozen snapshots (solved-against host)
            circ.model = None; del hf_if
            sys.path.remove(a.ifdir); sys.modules['common_ifeval'] = sys.modules.pop('common'); sys.modules['common'] = _saved_common
            sys.modules['tables_ifeval'] = sys.modules.pop('tables')
            if _saved_tables is not None: sys.modules['tables'] = _saved_tables
            permif = None
            if a.scramble: permif = torch.randperm(Q.D, generator=g_).to(dev)
            IFT = IFTool(circ, a.B_max, Q.D, perm=permif)
            print('IFEval circuit attached (frozen head snapshot of Qwen3-0.6B)', flush=True)
    del hf; torch.cuda.empty_cache()

# ---- trainable writer (W1/W4): Dp = out_dirs * gain (fp32, n_out x d) ----
Dp = eng_Dp = None
out_tok_ids = digit_rows = None
if TRAIN_IMPL:
    Dp = imp.D.detach().clone().float().requires_grad_(True)             # leaf param (optimizer)
    eng_Dp = torch.zeros_like(imp.D, dtype=torch.float32)                # graph-captured rollout buffer
    eng_Dp.copy_(Dp.detach())
# digit->out-row map for the exactness probe (works for any implant arm)
if HAS_IMPL and not IS_R:
    from micromachines.specs import SPECS
    from micromachines.machine import first as _first
    mname = imp.M.name
    out_syms = SPECS[mname]['out']
    ids_map = imp.A.all_ids[mname]
    out_tok_ids = torch.tensor([_first(ids_map[s]) for s in out_syms], device=dev)   # (n_out,)
    digit_rows = [(i, s) for i, s in enumerate(out_syms) if isinstance(s, str) and s.isdigit()]


class Link(nn.Module):
    """residual add  gate * (W f_src)  at a host layer's input. f_src = feat[src] when feat is a dict of implant
    features (multi-implant), or feat itself when a single tensor is passed (the engine passes calc's e)."""
    def __init__(self, src, nf, d, std, gate0, W0=None, init='rand'):
        super().__init__()
        self.src, self.init = src, init
        self.W = nn.Parameter(W0.float().clone() if W0 is not None else torch.randn(d, nf, device=dev) * std)
        self.gate = nn.Parameter(torch.full((d,), float(gate0), device=dev))
        self.log_gain = nn.Parameter(torch.zeros((), device=dev)) if init == 'soft' else None
        self.w_scale = max(1.0, float(self.W.detach().pow(2).mean().sqrt()))   # lr multiplier for W (relative steps)
        self.register_buffer('g0', self.gate.detach().clone()); self.register_buffer('W0', self.W.detach().clone())
        self.register_buffer('lg0', torch.zeros((), device=dev))

    def forward(self, feat, h=None):
        f = feat[self.src] if isinstance(feat, dict) else feat
        out = self.gate * F.linear(f.float(), self.W)
        return out * torch.exp(self.log_gain) if self.log_gain is not None else out


class LinkSum(nn.Module):
    """several sources into one layer: sum of their Links."""
    def __init__(self, ls):
        super().__init__(); self.ls = nn.ModuleList(ls)

    def forward(self, feat, h=None):
        return sum(l(feat, h) for l in self.ls)


def _ln(h):
    hf = h.float()
    return (hf - hf.mean(-1, keepdim=True)) * torch.rsqrt(hf.var(-1, unbiased=False, keepdim=True) + 1e-5)


class CWOut(nn.Module):
    """co-designed OUT-wire: g(h) * B relu(A e + c), g(h) = sigmoid(w_g . LN(h) + b_g). A ~ N(0,1) (e one-hot =>
    pre-acts ~ N(0,1)), c = 0.1, B = 0 (exact zero output at step 0; B still gets gradient), w_g = 0, b_g = 0 (g = 0.5).
    mask: output only where the implant emits (e != 0), so a wire cannot act as a host-only adapter on silent tokens."""
    init = 'cw'
    def __init__(self, src, nf, d, hidden, c, mask):
        super().__init__()
        self.src, self.mask = src, mask
        self.A = nn.Parameter(torch.randn(hidden, nf, device=dev))
        self.c = nn.Parameter(torch.full((hidden,), float(c), device=dev))
        self.B = nn.Parameter(torch.zeros(d, hidden, device=dev))
        self.wg = nn.Parameter(torch.zeros(d, device=dev)); self.bg = nn.Parameter(torch.zeros((), device=dev))
        self.register_buffer('mflag', torch.tensor(float(mask), device=dev))   # read inside CUDA graphs: toggle in place
        self.last = None

    def forward(self, feat, h=None):
        f = feat[self.src] if isinstance(feat, dict) else feat
        z = torch.relu(F.linear(f.float(), self.A) + self.c)
        g = torch.sigmoid(_ln(h) @ self.wg + self.bg)
        out = g[..., None] * F.linear(z, self.B)
        m = (f.abs().sum(-1) > 0)
        mm = m[..., None].to(out.dtype)
        out = out * (self.mflag * mm + (1 - self.mflag))
        if torch.is_grad_enabled(): self.last = (g, m, out.detach().float().norm(dim=-1))
        return out


class InWire(nn.Module):
    """learned host -> implant veto: fire = sigmoid(k*(trigger-0.5) + b + v.LN(h_L[t-1])) at each write-burst start.
    k, b0 set so P(fire | hand on) = 0.95, P(fire | hand off) = 0.02 (off: the calculator produced nothing to write)."""
    def __init__(self, L, d):
        super().__init__()
        lo, hi = math.log(0.95 / 0.05), math.log(0.02 / 0.98)
        self.k = lo - hi; self.b0 = (lo + hi) / 2; self.L = L
        self.b = nn.Parameter(torch.full((), self.b0, device=dev)); self.v = nn.Parameter(torch.zeros(d, device=dev))
        self.eps = 0.0


# implant feature sources: name -> dim. To add another implant (e.g. the rethink/mistake detector), register its state
# here and pass a dict feat = {'calc': e, 'rethink': s, ...} from the engine step / train pass.
SRC_DIMS = {'calc': imp.n_out} if HAS_IMPL else {}
for t_ in (TOOLS if HAS_IMPL else []): SRC_DIMS[t_.name] = t_.n_out
if HAS_IMPL and IFT is not None: SRC_DIMS['ifeval'] = Q.D; SRC_DIMS['ifeval_prev'] = Q.D

links = None
if HAS_LINKS:
    spec = a.link_spec or DEFAULT_SPEC.get(a.arm) or ','.join(f'calc:{l}' for l in a.link_layers.split(','))
    torch.manual_seed(1000 + a.seed)
    per_layer = {}
    Ef = P32['model.embed_tokens.weight'].detach().float()
    for item in spec.split(','):
        parts = item.split(':')
        src, l = parts[0], parts[1]
        init = parts[2] if len(parts) > 2 else 'rand'
        gate0 = float(parts[3]) if len(parts) > 3 else GATE0
        key = 'final' if l == 'final' else int(l)
        W0 = None
        if init == 'emb':                    # column i = unit embedding of out-symbol i's token * residual norm at layer l
            assert src == 'calc' and key != 'final'
            cols = Ef[out_tok_ids] / Ef[out_tok_ids].norm(dim=-1, keepdim=True) * RESID_NORM[key]   # (n_out, d)
            cols[imp.D.abs().sum(-1) == 0] = 0                                # symbols with no write (OS) stay silent
            W0 = cols.T
        elif init == 'warm':                 # the hand-wired write itself: gate 1 => identical to the end write
            assert src == 'calc'
            W0 = imp.D.float().T
        elif init == 'soft':                 # write directions (rms 1); the gain is a separate log-parameterised scalar
            assert src == 'calc' and key == 'final'
            W0 = (imp.D.float() / float(imp.M.gain)).T
        if init == 'cw':
            per_layer.setdefault(key, []).append(CWOut(src, SRC_DIMS[src], Q.D, a.cw_hidden, a.cw_c, bool(a.cw_mask)))
            continue
        per_layer.setdefault(key, []).append(Link(src, SRC_DIMS[src], Q.D, a.link_std, gate0, W0=W0, init=init))
    links = {k: (v[0] if len(v) == 1 else LinkSum(v)) for k, v in per_layer.items()}
    torch.manual_seed(a.seed)
    if a.init_links:                  # start from saved links; their values become the new reference init (g0, W0)
        sd = torch.load(a.init_links, map_location=dev)
        for k, lk in links.items():
            lk.load_state_dict(sd[str(k)], strict=False)   # older files lack the lg0 buffer
            for l in (lk.ls if isinstance(lk, LinkSum) else [lk]):
                if isinstance(l, Link): l.g0.copy_(l.gate.detach()); l.W0.copy_(l.W.detach())
        print('links initialised from', a.init_links, flush=True)
link_params = [p for lk in (links or {}).values() for p in lk.parameters()]
WARM_FINAL = None


def _all_links():
    for k, v in links.items():
        for l in (v.ls if isinstance(v, LinkSum) else [v]):
            if isinstance(l, Link): yield k, l


def _all_cw():
    for k, v in (links or {}).items():
        for l in (v.ls if isinstance(v, LinkSum) else [v]):
            if isinstance(l, CWOut): yield k, l


class MultiInWire(nn.Module):
    """one in-wire per tool (InWire math, vectorized): b (NT,), v (NT, D); same init as CW-lean's in-wire."""
    def __init__(self, L, d, nt):
        super().__init__()
        lo, hi = math.log(0.95 / 0.05), math.log(0.02 / 0.98)
        self.k = lo - hi; self.b0 = (lo + hi) / 2; self.L = L; self.nt = nt
        self.b = nn.Parameter(torch.full((nt,), self.b0, device=dev)); self.v = nn.Parameter(torch.zeros(nt, d, device=dev))
        self.eps = 0.0
TNAMES = (['calc'] + [t_.name for t_ in TOOLS] + (['ifeval'] if IFT is not None else [])) if a.arm == 'LIB' else ['calc']
NT = len(TNAMES)
INW = (MultiInWire(a.iw_layer, Q.D, NT) if a.arm == 'LIB' else InWire(a.iw_layer, Q.D)) if HAS_INWIRE else None
if INW is not None: INW.eps = a.iw_eps


def link_norms():
    """per link: gate L2, W L2, and effective map ||diag(gate) W||_F (the actual write size per unit feature)."""
    g, w, eff = {}, {}, {}
    for k, l in _all_links():
        n = f'{l.src}@{k}'
        while n in g: n += "'"
        g[n] = float(l.gate.detach().norm()); w[n] = float(l.W.detach().norm())
        eff[n] = float((l.gate.detach()[:, None] * l.W.detach()).norm())
        g['d_' + n] = float((l.gate.detach() - l.g0).norm()); w['d_' + n] = float((l.W.detach() - l.W0).norm())
        w['rel_' + n] = float((l.W.detach() - l.W0).norm() / l.W0.norm().clamp(min=1e-12))
        g['rel_' + n] = float((l.gate.detach() - l.g0).norm() / l.g0.norm().clamp(min=1e-12))
    return g, w, eff
groups = [] if a.freeze_host else [{'params': [P32[k] for k in names], 'lr': a.lr}]
if links:
    for k, l in _all_links():
        groups.append({'params': [l.gate], 'lr': a.link_lr, 'eps': a.link_eps})
        groups.append({'params': [l.W], 'lr': a.link_lr * l.w_scale, 'eps': a.link_eps})   # same RELATIVE step for any init scale
        if l.log_gain is not None: groups.append({'params': [l.log_gain], 'lr': a.link_lr, 'eps': a.link_eps})
        if l.init in ('warm', 'soft') and k == 'final': WARM_FINAL = l
    print('links:', [(f'{l.src}@{k}', l.init, round(float(l.gate.detach().mean()), 3), round(l.w_scale, 2)) for k, l in _all_links()], flush=True)
CW_LR = a.cw_lr if a.cw_lr is not None else a.link_lr
cw_params = [p for _, l in _all_cw() for p in l.parameters()] + (list(INW.parameters()) if INW is not None else [])
if cw_params:
    groups.append({'params': cw_params, 'lr': CW_LR, 'eps': a.link_eps})
    print('co-designed wires:', [f'out@{k}' for k, _ in _all_cw()], 'in-wire' if INW is not None else '', 'lr', CW_LR, flush=True)
link_params = link_params + ([p for p in INW.parameters()] if INW is not None else [])
if TRAIN_IMPL:   # relative lr: Dp entries are ~gain (1e4) scale, so an absolute lr would leave the writer frozen
    D_SCALE = max(1.0, float(imp.D.float().pow(2).mean().sqrt()))
    groups.append({'params': [Dp], 'lr': a.impl_lr * D_SCALE, 'eps': a.link_eps})
    print('writer Dp lr', a.impl_lr, 'x rms(D)', round(D_SCALE, 1), flush=True)
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
base_lrs = [g['lr'] for g in opt.param_groups]
TT = []
if a.impl_train:
    from trainable_impl import TrainableTool
    with torch.no_grad():                    # soft writer gain: smallest gain whose write still wins the argmax with margin >= 6
        g_ = P32['model.norm.weight'].detach().float(); Eh = P32['model.embed_tokens.weight'].detach().float()
        torch.manual_seed(5); hr = torch.randn(64, Q.D, device=dev); hr = hr / hr.norm(dim=-1, keepdim=True) * 150.0
    for t_ in TOOLS:
        od = t_.M.out_dirs.float()
        if a.impl_gain > 0: gsel = a.impl_gain
        else:
            gsel = None
            for gg_ in (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10000):
                ok_ = True
                for i_ in range(od.shape[0]):
                    if od[i_].abs().sum() == 0: continue
                    x_ = hr + gg_ * od[i_][None]; x_ = x_ * torch.rsqrt(x_.pow(2).mean(-1, keepdim=True) + 1e-6) * g_
                    lg_ = x_ @ Eh.T; tg_ = lg_.argmax(-1)
                    # target = argmax of the pure write direction
                    xd = od[i_] * torch.rsqrt(od[i_].pow(2).mean() + 1e-6) * g_; tgt = int((xd @ Eh.T).argmax())
                    top2 = lg_.topk(2, -1).values
                    if not bool((tg_ == tgt).all()) or float((top2[:, 0] - top2[:, 1]).min()) < 6: ok_ = False; break
                if ok_: gsel = gg_; break
            gsel = gsel or 10000
        frz_ = [x.split(':')[1] for x in a.impl_freeze.split(',') if x and x.split(':')[0] == t_.name]
        TT.append(TrainableTool(t_, gain_target=gsel, ste_reader=bool(a.impl_ste), freeze=frz_))
        print('impl_train:', t_.name, 'soft gain', gsel, flush=True)
    for tt_ in TT: groups.extend(tt_.groups(a.impl_lr_reader, a.impl_lr_core, a.impl_lr_writer, a.link_eps))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8); base_lrs = [g['lr'] for g in opt.param_groups]
DRIFT_STREAMS = None
if TT:                                       # fixed token streams with tool calls (BBH call transcripts) for exactness drift
    from micromachines.specs import SPECS as _SP
    from micromachines.machine import first as _fst
    _un = os.environ.get('UNION', os.path.join(INTEG, '..', 'data', 'train_union_v2.jsonl'))
    _rows = [json.loads(l) for l in open(_un)][:96]
    _ids = [tok(r_['prompt'][-600:] + r_['completion'], add_special_tokens=False)['input_ids'][-384:] for r_ in _rows]
    _L = max(map(len, _ids)); DRIFT_STREAMS = torch.tensor([[eng_eos_ := 151643] * (_L - len(x)) + x for x in _ids], device=dev)
    for tt_, n_ in zip(TT, TOOL_NAMES):
        tt_.out_ids = torch.tensor([_fst(A_bbh.all_ids[n_][o_]) for o_ in _SP[n_]['out']], device=dev)
def impl_drift():
    gN = P32['model.norm.weight'].detach().float(); EH = P32['model.embed_tokens.weight'].detach().float()
    head = lambda D: ((D * torch.rsqrt(D.pow(2).mean(-1, keepdim=True) + 1e-6) * gN) @ EH.T).argmax(-1)
    out = {}
    for tt_ in TT:
        Kt_ = tt_.t.Kt
        ids_ = torch.cat([torch.full((DRIFT_STREAMS.shape[0], Kt_), 151643, device=dev), DRIFT_STREAMS], 1)
        out[tt_.t.name] = tt_.drift(E0[ids_], head, tt_.out_ids) | dict(log_gain=float(tt_.log_gain))
    return out
IMPL_PARAMS = [p_ for tt_ in TT for gr_ in tt_.groups(0, 0, 0, 0) for p_ in gr_['params']]
if a.arm == 'LIB':
    eng = LibEngine(W, pre, tok, imp, TOOLS, IFT, E0, links=links, B_max=a.B_max, S_max=a.S_max)
    eng.chat_stop = MIXED
    eng.imp_on = not a.plain
else:
    eng = Engine(W, pre, tok, implant=imp, links=links, B_max=a.B_max, S_max=544, use_write=USE_WRITE)
if TRAIN_IMPL: eng.Dp = eng_Dp
HAS_EXTRA_LINKS = bool(links) and any(l is not WARM_FINAL for _, l in _all_links()) and (USE_WRITE or WARM_FINAL is not None)
if INW is not None: (eng.set_mwire(INW) if a.arm == 'LIB' else eng.set_inwire(INW))
if a.arm in ('CWinP', 'CWinO'):          # same derangement as arm R (seed 12345), applied only to some calls
    g_ = torch.Generator().manual_seed(12345)
    while True:
        p_ = torch.randperm(10, generator=g_)
        if not (p_ == torch.arange(10)).any(): break
    D_der = imp.D.clone(); D_der[:10] = imp.D[p_.to(imp.D.device)]
    if a.arm == 'CWinP':
        eng.set_partial(a.partial_thresh, D_der)
        print('CWinP: calls with an operand >=', a.partial_thresh, 'digits use deranged digits', p_.tolist(), flush=True)
    else:
        eng.set_partial_ops({1, 2}, D_der)
        print('CWinO: + and - calls use deranged digits, * and / exact', p_.tolist(), flush=True)


_HARD = {}
@torch.no_grad()
def p_written(log_s, n=128, hard=True):
    """mean p(written token) at firing positions of calculator-made traces, base host, final write = exp(log_s) * e @ W0.T"""
    rows = [json.loads(l) for l in open(a.calib_traces)][:n]
    seqs = [C.q_ids(tok, r['q']) + r['resp'] for r in rows]; qls = [len(C.q_ids(tok, r['q'])) for r in rows]
    T = max(map(len, seqs)); ids = torch.tensor([x + [eng.eos] * (T - len(x)) for x in seqs], device=dev)
    Kt = imp.K + imp.W
    e = imp.full(W['E'][torch.cat([eng.prefix_ids[-Kt:][None].expand(len(seqs), -1), ids], 1)])[:, Kt:]
    wd = e @ WARM_FINAL.W0.T
    hn = Q.fwd_chunk(W, ids, eng.Lp, eng.cos_t, eng.sin_t, prefix_kv=[(eng.Kp[i], eng.Vp[i]) for i in range(Q.NL)],
                     final_write=math.exp(log_s) * wd)
    dig = torch.tensor([i for i, _ in digit_rows], device=dev)
    isdig = lambda ev: torch.isin(ev.argmax(-1), dig) & (ev.abs().sum(-1) > 0)     # implant intends a DIGIT here
    if hard and not _HARD:           # digit-write positions where the host ALONE (no write) would emit a different token
        h0 = Q.fwd_chunk(W, ids, eng.Lp, eng.cos_t, eng.sin_t, prefix_kv=[(eng.Kp[i], eng.Vp[i]) for i in range(Q.NL)])
        for j, x in enumerate(seqs):
            pos = [p_ for p_ in range(qls[j] - 1, len(x) - 1) if bool(isdig(e[j, p_]))]
            if pos:
                pt = torch.tensor(pos, device=dev)
                am = F.linear(h0[j, pt], W['E']).argmax(-1)
                want = out_tok_ids[e[j, pt].argmax(-1)]                 # the token the implant intends to write
                _HARD[j] = [p_ for p_, ok in zip(pos, (am != want).tolist()) if ok]
        print('calib: hard firing positions', sum(map(len, _HARD.values())), flush=True)
    ps = []
    for j, x in enumerate(seqs):
        pos = _HARD.get(j, []) if hard else [p_ for p_ in range(qls[j] - 1, len(x) - 1) if bool(isdig(e[j, p_]))]
        if not pos: continue
        pt = torch.tensor(pos, device=dev)
        pr = torch.softmax(F.linear(hn[j, pt], W['E']).float(), -1)
        want = out_tok_ids[e[j, pt].argmax(-1)]                          # p(intended written token)
        ps.append(pr.gather(-1, want[:, None])[:, 0])
    return float(torch.cat(ps).mean())


if WARM_FINAL is not None and WARM_FINAL.log_gain is not None:     # W2ws: calibrate log-gain to p(written) ~ soft_p
    lo, hi = math.log(1e-3), math.log(float(imp.M.gain))
    print('calib: p at gain 1e-3 / 1 / 1e4 =', round(p_written(lo), 4), round(p_written(0.0), 4), round(p_written(hi), 4), flush=True)
    for _ in range(14):
        mid = (lo + hi) / 2
        if p_written(mid) < a.soft_p: lo = mid
        else: hi = mid
    lg = (lo + hi) / 2
    WARM_FINAL.log_gain.data.fill_(lg); WARM_FINAL.lg0.fill_(lg)
    print(f'W2ws: gain {math.exp(lg):.2f} (hand gain {float(imp.M.gain):.0f}) -> p(written | host alone wrong) '
          f'{p_written(lg):.3f}, p(written, all firing) {p_written(lg, hard=False):.3f}', flush=True)

# ---- parameter budget: trainable wiring vs host and vs the implant's nonzero weights
N_HOST = sum(v.numel() for v in P32.values())
N_WIRE = sum(p.numel() for _, l in (_all_links() if links else []) for p in (l.gate, l.W)) + \
         sum(1 for _, l in (_all_links() if links else []) if l.log_gain is not None) + \
         sum(p.numel() for _, l in _all_cw() for p in l.parameters()) + \
         (sum(p.numel() for p in INW.parameters()) if INW is not None else 0) + (Dp.numel() if TRAIN_IMPL else 0)
N_IMPL = 0
if HAS_IMPL:
    N_IMPL = int(imp.M.stack1.nnz + imp.M.stack2.nnz + (imp.M.reader_w != 0).sum() + (imp.M.reader_b != 0).sum() + (imp.D != 0).sum())
if a.arm == 'LIB':
    for t_ in TOOLS:
        M_ = t_.M; nn_ = 0
        for st_ in (M_.stack1, M_.stack2):
            inner = getattr(st_, 'net', st_)
            nn_ += int(getattr(inner, 'nnz', 0) or sum(int((p_ != 0).sum()) for p_ in list(inner.parameters()) + list(inner.buffers())))
        N_IMPL += nn_ + int((M_.reader_w != 0).sum() + (M_.reader_b != 0).sum() + (t_.D != 0).sum())
PARAMS = dict(wire=N_WIRE, host=N_HOST, implant_nnz=N_IMPL, pct_host=100 * N_WIRE / N_HOST,
              pct_implant=(100 * N_WIRE / N_IMPL) if N_IMPL else None)
print(f'PARAMS trainable wiring {N_WIRE:,} = {PARAMS["pct_host"]:.4f}% of host ({N_HOST:,}) = '
      f'{PARAMS["pct_implant"] if N_IMPL else float("nan"):.2f}% of implant nonzeros ({N_IMPL:,})', flush=True)

if MIXED:
    pool = [dict(r, kind=r['type']) for r in MT.load('train')]
    pool = [r for r in pool if len(QID(r)) + a.max_new <= a.S_max]
    TESTSET = [dict(r, kind=r['type']) for r in MT.load('test')]
    TESTSET = [r for r in TESTSET if len(QID(r)) + a.max_new <= a.S_max]
    rs_ = random.Random(123); bytype = {}
    for r in TESTSET: bytype.setdefault(r['type'], []).append(r)
    devset = []
    for t_, rr in sorted(bytype.items()):
        rs_.shuffle(rr); devset += rr[:max(1, round(a.dev_mixed_n * len(rr) / len(TESTSET)))]
    TYPES = sorted(bytype)
    print('mixed: pool', len(pool), 'test', len(TESTSET), 'dev', len(devset), {t_: len(v) for t_, v in bytype.items()}, flush=True)
if not MIXED:
  pool = C.load_train('A', shot_q=shot_q)
  pool = [r for r in pool if len(C.q_ids(tok, r['q'])) + a.max_new <= 544]
  devset = C.load_train('B', shot_q=shot_q)
if not MIXED:
    rng_dev = random.Random(123)
    dg = [r for r in devset if r['kind'] == 'gsm']; ds = [r for r in devset if r['kind'] == 'scaled']
    rng_dev.shuffle(dg); rng_dev.shuffle(ds)
    devset = dg[:a.dev_n // 2] + ds[:a.dev_n // 2]
    # op class per dev problem from its gold solution's calculator annotations <<expr=val>> (scaled -> source solution)
    import re as _re0
    _tr = [json.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')]
    _sol = {r['question']: r['answer'] for r in _tr}
    for l in open(f'{C.REPO}/data/gsm8k/train_scaled.jsonl'):
        r = json.loads(l); _sol[r['question']] = _tr[r['src']]['answer']
    for r in devset:
        ex = ' '.join(_re0.findall(r'<<([^=>]*)=', _sol.get(r['q'], '')))
        md, as_ = len(_re0.findall(r'[*/]', ex)), len(_re0.findall(r'(?<=[\d)\s])[+-]', ex))
        r['opclass'] = 'muldiv' if md > as_ else ('addsub' if as_ > md else 'tie')
    del _tr
print(f'arm {a.arm} | impl {HAS_IMPL} links {HAS_LINKS} write {USE_WRITE} gate0 {GATE0} train_impl {TRAIN_IMPL} '
      f'| pool {len(pool)} dev {len(devset)} prefix {Lp}', flush=True)


def log(obj):
    with open(f'{a.run_dir}/steps.jsonl', 'a') as f: f.write(json.dumps(obj) + '\n')


@torch.no_grad()
def sync_weights():
    Wn = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16)
    for k in ('E', 'lnf'): W[k].copy_(Wn[k])
    for l, ln in zip(W['L'], Wn['L']):
        for k in l: l[k].copy_(ln[k])
    if TRAIN_IMPL: eng.Dp.copy_(Dp.detach())
    eng.refresh()


def calls_of(wf):
    return sum(1 for i in range(len(wf)) if wf[i] and (i == 0 or not wf[i - 1]))


# -------- writer exactness drift (W1/W4): do the written digit directions still decode to their own digit? --------
_probe_nums = None
def probe_nums(n=2000):
    global _probe_nums
    if _probe_nums is None:
        r = random.Random(2024); nums = []
        for _ in range(n):
            op = r.choice(['+', '-', '*'])
            a1 = r.randint(1, 10 ** r.randint(1, 7)); b1 = r.randint(1, 10 ** r.randint(1, 7))
            v = a1 + b1 if op == '+' else (a1 - b1 if op == '-' else a1 * b1)
            nums.append(str(abs(v)))
        _probe_nums = nums
    return _probe_nums


@torch.no_grad()
def writer_drift(tag='drift'):
    """integrity[d] = argmax over vocab of host(norm + head)(Dp_row_d) == token('d'); exact = fraction of 2000 random
    results whose every digit still decodes to itself. At u0 this is 1.0; it falls iff RL bent a digit direction."""
    if WARM_FINAL is not None:            # W2w: the final link IS the writer
        Dcur = (WARM_FINAL.gate.detach()[:, None] * WARM_FINAL.W.detach()).T.float()
        if WARM_FINAL.log_gain is not None: Dcur = Dcur * float(torch.exp(WARM_FINAL.log_gain.detach()))
    else:
        Dcur = (Dp.detach() if TRAIN_IMPL else imp.D).float()
    g = P32['model.norm.weight'].detach().float(); E = P32['model.embed_tokens.weight'].detach().float()
    x = Dcur * torch.rsqrt(Dcur.pow(2).mean(-1, keepdim=True) + Q.EPS) * g        # RMSNorm(Dp_row)
    pred = (x @ E.T).argmax(-1)                                                   # (n_out,)
    integ = {s: bool(pred[i].item() == out_tok_ids[i].item()) for i, s in digit_rows}
    ok = {s: integ[s] for s in integ}
    nums = probe_nums()
    exact = sum(all(ok.get(ch, False) for ch in num) for num in nums) / len(nums)
    return {f'{tag}_exact': exact, f'{tag}_digit_ok': sum(ok.values()), f'{tag}_digits': integ}


def _zero_gates_ctx():
    saved = [l.gate.detach().clone() for _, l in _all_links()]
    for _, l in _all_links():
        if l is not WARM_FINAL: l.gate.detach().zero_()          # the warm final link is the write: keep it
    return saved
def _cw_ctx(out_on, in_trained, writer_trained):
    """co-designed eval configs: out-wires on/off (B zeroed), in-wire trained or reset (v=0, b=b0: always fires at
    greedy), writer (warm final link) trained or reset to the hand-made write."""
    sv = dict(B=[l.B.detach().clone() for _, l in _all_cw()],
              iw=(INW.v.detach().clone(), INW.b.detach().clone()) if INW is not None else None,
              wf=(WARM_FINAL.gate.detach().clone(), WARM_FINAL.W.detach().clone()) if WARM_FINAL is not None else None)
    if not out_on:
        for _, l in _all_cw(): l.B.detach().zero_()
    if INW is not None and not in_trained: INW.v.detach().zero_(); INW.b.detach().fill_(INW.b0)
    if WARM_FINAL is not None and not writer_trained:
        WARM_FINAL.gate.detach().copy_(WARM_FINAL.g0); WARM_FINAL.W.detach().copy_(WARM_FINAL.W0)
    return sv
def _cw_restore(sv):
    for (_, l), b in zip(_all_cw(), sv['B']): l.B.detach().copy_(b)
    if sv['iw'] is not None: INW.v.detach().copy_(sv['iw'][0]); INW.b.detach().copy_(sv['iw'][1])
    if sv['wf'] is not None: WARM_FINAL.gate.detach().copy_(sv['wf'][0]); WARM_FINAL.W.detach().copy_(sv['wf'][1])
CW_MODES = {'cw_hand': (False, False, False), 'cw_out': (True, False, True), 'cw_in': (False, True, True)}
def _reset_ctx():
    """original hand-made wiring with the CURRENT host: warm links back to init, other links off, writer Dp back to D."""
    sv = {'links': [(l.gate.detach().clone(), l.W.detach().clone(), None if l.log_gain is None else l.log_gain.detach().clone())
                    for _, l in _all_links()] if links else [],
          'Dp': eng.Dp.clone() if TRAIN_IMPL else None}
    if links:
        for _, l in _all_links():
            if l.init in ('warm', 'soft'):
                l.gate.detach().copy_(l.g0); l.W.detach().copy_(l.W0)
                if l.log_gain is not None: l.log_gain.detach().copy_(l.lg0)
            else: l.gate.detach().zero_()
    if TRAIN_IMPL: eng.Dp.copy_(imp.D.float())
    return sv
def _reset_restore(sv):
    for (_, l), (g, w, lg) in zip(_all_links() if links else [], sv['links']):
        l.gate.detach().copy_(g); l.W.detach().copy_(w)
        if lg is not None: l.log_gain.detach().copy_(lg)
    if sv['Dp'] is not None: eng.Dp.copy_(sv['Dp'])
def _restore_gates(saved):
    for (_, l), s in zip(_all_links(), saved): l.gate.detach().copy_(s)


def evaluate(items, mode='full', tag='dev'):
    """mode: full (impl+links on) | hostonly (imp_on off: write+links off) | linksoff (gates zeroed, write on)."""
    if a.no_eval: return {}
    torch.cuda.empty_cache()
    saved = None; rs_saved = None
    if mode == 'reset': rs_saved = _reset_ctx()
    cw_saved = _cw_ctx(*CW_MODES[mode]) if mode in CW_MODES else None
    m21 = None
    if mode == 'cw_mask21':                  # only the L21 out-wire restricted to firing tokens (graphs re-captured)
        m21 = [l for k, l in _all_cw() if k == 21][0]; m21.mflag.fill_(1.0)
    ton = None
    if mode.startswith('loo_') or mode in ('toolsoff', 'calconly'):     # detach one tool (leave-one-out) / every tool
        ton = eng.tool_on.clone()
        if mode == 'toolsoff': eng.tool_on.zero_()
        elif mode == 'calconly': eng.tool_on.zero_(); eng.tool_on[0] = 1
        else: eng.tool_on[TNAMES.index(mode[4:])] = 0
    if mode == 'hostonly': eng.imp_on = False
    elif mode == 'linksoff' and links is not None: saved = _zero_gates_ctx()
    qids = [QID(r) for r in items]
    out = eng.generate(qids, G=1, greedy=True, max_new=a.max_new)
    sc = [SCORE(t, r) for t, r in zip(out['tokens'], items)]
    acc = lambda kind: 100 * sum(s[0] for s, r in zip(sc, items) if r['kind'] == kind) / max(1, sum(r['kind'] == kind for r in items))
    res = {f'{tag}_acc': 100 * sum(s[0] for s in sc) / len(sc), f'{tag}_gsm': acc('gsm'), f'{tag}_scaled': acc('scaled'),
           f'{tag}_fmt': 100 * sum(s[1] for s in sc) / len(sc),
           f'{tag}_writes': sum(sum(w) for w in out['wflag']) / len(sc), f'{tag}_calls': sum(calls_of(w) for w in out['wflag']) / len(sc),
           f'{tag}_len': sum(map(len, out['tokens'])) / len(sc)}
    if MIXED:                                         # per task type: accuracy + per-tool routing
        for t_ in TYPES:
            ii = [i for i, r in enumerate(items) if r['type'] == t_]
            if not ii: continue
            res[f'{tag}_{t_}'] = 100 * sum(sc[i][0] for i in ii) / len(ii)
            if a.arm == 'LIB' and mode != 'hostonly':
                sub = {k: [out[k][i] for i in ii] for k in ('dec', 'wt', 'ifw', 'if_lat', 'if_lat_prompt', 'if_ans') if out.get(k)}
                res[f'{tag}_tools_{t_}'] = {k[len(tag) + 3:]: v for k, v in tool_stats(sub, tag).items() if k.startswith(f'{tag}_t_')}
                res[f'{tag}_if_{t_}'] = {k: v for k, v in tool_stats(sub, tag).items() if k.startswith(f'{tag}_if_')}
    for oc in ('muldiv', 'addsub'):                  # accuracy by the problem's dominant arithmetic (gold solution)
        ii = [i for i, r in enumerate(items) if r.get('opclass') == oc]
        if ii:
            res[f'{tag}_acc_{oc}'] = 100 * sum(sc[i][0] for i in ii) / len(ii); res[f'{tag}_n_{oc}'] = len(ii)
            for kd in ('gsm', 'scaled'):
                jj = [i for i in ii if items[i]['kind'] == kd]
                if jj: res[f'{tag}_acc_{oc}_{kd}'] = 100 * sum(sc[i][0] for i in jj) / len(jj)
    eng.imp_on = not a.plain
    if ton is not None: eng.tool_on.copy_(ton)
    if saved is not None: _restore_gates(saved)
    if rs_saved is not None: _reset_restore(rs_saved)
    if cw_saved is not None: _cw_restore(cw_saved)
    if m21 is not None: m21.mflag.fill_(float(a.cw_mask))
    import hashlib
    res[f'{tag}_hash'] = hashlib.md5(str(out['tokens']).encode()).hexdigest()[:12]      # bit-identity checks
    if a.arm == 'LIB' and mode != 'hostonly':
        res.update(tool_stats(out, tag))
    if a.arm != 'LIB' and INW is not None and mode != 'hostonly':
        d_ = out.get('dec', [])
        res[f'{tag}_starts'] = sum(sum(1 for x in d if x > 0) for d in d_) / len(sc)
        res[f'{tag}_vetoes'] = sum(sum(1 for x in d if (x & 3) == 2) for d in d_) / len(sc)
        bo, bop = {}, {}
        for d, o in zip(d_, out.get('ops', [])):
            for x, n_ in zip(d, o):
                if x <= 0: continue
                nd_, op_ = n_ & 15, n_ >> 4
                b_ = '0' if nd_ <= 0 else ('1-2' if nd_ <= 2 else ('3-4' if nd_ <= 4 else ('5-6' if nd_ <= 6 else '7+')))
                c_ = bo.setdefault(b_, [0, 0]); c_[0] += 1; c_[1] += int((x & 3) == 2)
                c_ = bop.setdefault('?+-*/'[op_] if op_ < 5 else '?', [0, 0]); c_[0] += 1; c_[1] += int((x & 3) == 2)
        res[f'{tag}_veto_by_ops'] = bo                             # tracker-based: bin -> [calls, vetoes]
        res[f'{tag}_veto_by_op'] = bop                             # by operator: op -> [calls, vetoes]
        import re as _re
        rx = _re.compile(r'(\d[\d,]*\.?\d*)\s*(?:[-+*/x\u00d7\u00f7])\s*\$?(\d[\d,]*\.?\d*)\s*=\s*\$?$')
        byd, pv, pn = {}, [], []
        for t_, d, sc_, r_ in zip(out['tokens'], d_, sc, items):
            anyv = False
            for i_, x in enumerate(d):
                if x <= 0: continue
                m_ = rx.search(tok.decode(t_[:i_])[-80:])
                if m_:
                    nd = max(len(_re.sub(r'\D', '', m_.group(1).split('.')[0])), len(_re.sub(r'\D', '', m_.group(2).split('.')[0])))
                    b_ = '1-2' if nd <= 2 else ('3-4' if nd <= 4 else ('5-6' if nd <= 6 else '7+'))
                else:
                    b_ = 'unparsed'
                c_ = byd.setdefault(b_, [0, 0]); c_[0] += 1; c_[1] += int((x & 3) == 2); anyv |= ((x & 3) == 2)
            (pv if anyv else pn).append(sc_[0])
        res[f'{tag}_veto_by_digits'] = byd                         # bin -> [calls, vetoes]
        res[f'{tag}_acc_vetoed'] = 100 * sum(pv) / max(1, len(pv)); res[f'{tag}_n_vetoed'] = len(pv)
        res[f'{tag}_acc_notvetoed'] = 100 * sum(pn) / max(1, len(pn)); res[f'{tag}_n_notvetoed'] = len(pn)
    return res


def tool_stats(out, tag, kinds=None):
    """per tool: bursts (decisions) per answer, fire rate, veto rate, firing-token fraction; IFEval latch rates."""
    r = {}
    D = torch.cat([d.to(torch.int64) for d in out['dec']]) if out['dec'] else torch.zeros(0, NT, dtype=torch.int64)
    Wt_ = torch.cat([w.to(torch.int64) for w in out['wt']]) if out['wt'] else torch.zeros(0, NT, dtype=torch.int64)
    n = max(1, len(out['dec']))
    for k_, nm in enumerate(TNAMES):
        st_ = int((D[:, k_] > 0).sum()); fi_ = int((D[:, k_] == 1).sum())
        r[f'{tag}_t_{nm}'] = dict(bursts=st_ / n, fire=fi_ / max(1, st_), veto=(st_ - fi_) / max(1, st_), nb=st_,
                                  wtok=float(Wt_[:, k_].sum()) / max(1, Wt_.shape[0]),
                                  rows_any=sum(int((d[:, k_] > 0).any()) for d in out['dec']) / n)
    if IFT is not None and out.get('if_lat'):
        r[f'{tag}_if_latch_prompt'] = sum(out['if_lat_prompt']) / n
        r[f'{tag}_if_latch_any'] = sum(out['if_lat']) / n
        r[f'{tag}_if_ans_mode'] = sum(out['if_ans']) / n
        r[f'{tag}_if_write_rows'] = sum(int(len(p_) > 0) for p_, _ in out['ifw']) / n
        r[f'{tag}_if_write_pos'] = sum(len(p_) for p_, _ in out['ifw'])
    return r


@torch.no_grad()
def _swap_wiring(to_init):
    cur = dict(links={k: _copy.deepcopy(lk.state_dict()) for k, lk in (links or {}).items()},
               inw=_copy.deepcopy(INW.state_dict()) if INW is not None else None)
    src = INIT_WIRING if to_init else to_init
    for k, lk in (links or {}).items(): lk.load_state_dict(src['links'][k])
    if INW is not None: INW.load_state_dict(src['inw'])
    return cur
@torch.no_grad()
def _restore_wiring(sv):
    for k, lk in (links or {}).items(): lk.load_state_dict(sv['links'][k])
    if INW is not None: INW.load_state_dict(sv['inw'])
@torch.no_grad()
def _swap_impl():                              # original implants (arm C): copy frozen weights in, keep the trained ones
    sv = []
    for tt_ in TT:
        M, F0 = tt_.t.M, tt_.frozen
        cur = {n_: p_.detach().clone() for n_, p_ in M.named_parameters()}
        cur['_dirs'] = tt_.dirs.detach().clone(); cur['_lg'] = tt_.log_gain.detach().clone(); sv.append(cur)
        f0 = dict(F0.named_parameters()); f0.update(dict(F0.named_buffers()))
        for n_, p_ in M.named_parameters():
            if n_ in f0: p_.copy_(f0[n_].to(p_.dtype))
        tt_.dirs.copy_(F0.out_dirs.float() if tt_.t.perm is None else F0.out_dirs.float()[tt_.t.perm]); tt_.log_gain.fill_(math.log(float(F0.gain)))
        tt_.refresh()
    return sv
@torch.no_grad()
def _restore_impl(sv):
    for tt_, cur in zip(TT, sv):
        for n_, p_ in tt_.t.M.named_parameters(): p_.copy_(cur[n_])
        tt_.dirs.copy_(cur['_dirs']); tt_.log_gain.copy_(cur['_lg']); tt_.refresh()
@torch.no_grad()
def _swap_host(sd_new):
    cur = {k: P32[k].detach().to('cpu', copy=True) for k in names}
    for k in names: P32[k].copy_(sd_new[k].to(dev))
    sync_weights(); return cur


def save(tag):
    d = f'{a.run_dir}/{tag}'; os.makedirs(d, exist_ok=True)
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in P32.items()}
    sd['lm_head.weight'] = sd['model.embed_tokens.weight'].clone()
    save_file(sd, f'{d}/model.safetensors')
    if links: torch.save({str(k): lk.state_dict() for k, lk in links.items()}, f'{d}/links.pt')
    if INW is not None: torch.save(INW.state_dict(), f'{d}/inwire.pt')
    if TT: torch.save({tt_.t.name: dict(params={n_: p_.detach().cpu() for n_, p_ in tt_.t.M.named_parameters()},
                                         dirs=tt_.dirs.detach().cpu(), log_gain=tt_.log_gain.detach().cpu()) for tt_ in TT}, f'{d}/implants.pt')
    if TRAIN_IMPL: torch.save(Dp.detach().cpu(), f'{d}/Dp.pt')
    open(f'{d}/DONE', 'w').close()


PACK_SIZES = [n for n in (2048, 4096, 8192, 16384, 32768) if n <= a.tok_budget] or [a.tok_budget]
packers = {n: Q.Packer(n, Lp, dev) for n in PACK_SIZES}


def lp_of_hidden(Wt, hn, tgt):
    E = Wt['E']
    def chunk_lp(hc, tc):
        lg = F.linear(hc, E).float() / a.T
        return torch.log_softmax(lg, -1).gather(-1, tc[:, None])[:, 0]
    return torch.cat([checkpoint(chunk_lp, hn[c:c + 4096], tgt[c:c + 4096], use_reentrant=False) for c in range(0, hn.shape[0], 4096)])


def train_step(batch_q, batch_r, old_lp, adv, n_total_rows, batch_dec=None, batch_ifw=None):
    t0 = time.time()
    W0 = Q.fuse(P32, torch.bfloat16)
    pkv = Q.prefix_kv_of(W0, eng.prefix_ids, eng.cos_t, eng.sin_t)
    leaves = [(k.detach().requires_grad_(True), v.detach().requires_grad_(True)) for k, v in pkv]
    pkv_ref = None
    if W_ref is not None:
        with torch.no_grad(): pkv_ref = Q.prefix_kv_of(W_ref, eng.prefix_ids, eng.cos_t, eng.sin_t)
    order = sorted(range(len(batch_q)), key=lambda i: -(len(batch_q[i]) + len(batch_r[i])))
    bins = []
    for i in order:
        L = len(batch_q[i]) + len(batch_r[i])
        for bn in bins:
            if bn[1] + L <= a.tok_budget: bn[0].append(i); bn[1] += L; break
        else: bins.append([[i], L])
    micro = [bn[0] for bn in bins]
    tot_tok = sum(len(r) for r in batch_r)
    norm = (n_total_rows * a.max_new) if a.loss == 'drgrpo' else tot_tok
    st = dict(loss=0.0, kl=0.0, clipfrac=0.0, ratio=0.0, ntok=tot_tok, n_micro=len(micro),
              pack_fill=sum(bn[1] for bn in bins) / (len(bins) * a.tok_budget))
    Kt = imp.K + imp.W if imp is not None else 0
    n_e = n_sel = 0
    cw_acc = {}                                       # per out-wire: sum g, sum (g > .5), n firing positions
    for mb in micro:
        Wt = Q.fuse(P32, torch.bfloat16)
        seqs = [batch_q[i] + batch_r[i] for i in mb]
        lens = [len(x) for x in seqs]
        Ltot = next(n for n in PACK_SIZES if n >= sum(lens)); packer = packers[Ltot]
        flat = sum(seqs, []); flat += [eng.eos] * (Ltot - len(flat))
        ids = torch.tensor(flat, device=dev)
        bm = packer.pack(lens)
        sel, o = [], 0
        for j, i in enumerate(mb):
            ql, rl = len(batch_q[i]), len(batch_r[i])
            sel.append(torch.arange(o + ql - 1, o + ql + rl - 1)); o += lens[j]
        sel = torch.cat(sel).to(dev); tgt = ids[sel + 1]
        feat = fwrite = None; dpos, didx, dsgn, dtool = [], [], [], []
        if imp is not None and a.arm == 'LIB' and not a.plain:
            with torch.no_grad():
                Tm = max(lens)
                full0 = torch.tensor([x + [eng.eos] * (Tm - len(x)) for x in seqs], device=dev)
                def packed(eb_, nf_):
                    return torch.cat([eb_[j, :n] for j, n in enumerate(lens)] + [torch.zeros(Ltot - sum(lens), nf_, device=dev)])
                full = torch.cat([eng.tool_prefix[None].expand(len(seqs), -1), full0], 1)
                Es = {'calc': packed(imp.full(W['E'][full[:, eng.Lt - Kt:]])[:, Kt:] * eng.tool_on[0], imp.n_out)}
                tf_ = eng.tool_feats(full, eng.Lt)
                for t_ in TOOLS: Es[t_.name] = packed(tf_[t_.name], t_.n_out)
                if IFT is not None:                                # recorded IF writes (sparse) at response positions
                    wif = torch.zeros(Ltot, Q.D, device=dev); o2 = 0
                    for j, i in enumerate(mb):
                        ql = len(batch_q[i]); pz, vz = batch_ifw[i]
                        if len(pz): wif[o2 + ql - 1 + pz.to(dev)] = vz.to(dev)
                        o2 += lens[j]
                    Es['ifeval'] = wif
                nzall = torch.stack([(Es[n].abs().sum(-1) > 0) for n in TNAMES], -1).cpu()     # (Ltot, NT)
                keep = torch.ones(Ltot, NT, dtype=torch.bool)
                dpos, didx, dsgn, dtool = [], [], [], []
                o2 = si = 0
                for j, i in enumerate(mb):
                    ql, rl = len(batch_q[i]), len(batch_r[i]); dl = batch_dec[i]
                    sup = torch.zeros(NT, dtype=torch.bool)
                    dlr = dl[:rl].to(torch.int64) if len(dl) >= rl else torch.cat([dl.to(torch.int64), torch.zeros(rl - len(dl), NT, dtype=torch.int64)])
                    for t in range(rl):
                        pp = o2 + ql - 1 + t; d_ = dlr[t]
                        if bool((d_ > 0).any()):
                            sup = torch.where(d_ == 2, True, torch.where(d_ == 1, False, sup))
                            for k_ in (d_ > 0).nonzero()[:, 0].tolist():
                                dpos.append(pp); didx.append(si + t); dsgn.append(1.0 if int(d_[k_]) == 1 else -1.0); dtool.append(k_)
                        sup = sup & nzall[pp]
                        keep[pp] = ~sup
                    o2 += lens[j]; si += rl
                keep = keep.to(dev)
                for k_, n in enumerate(TNAMES): Es[n] = Es[n] * keep[:, k_:k_ + 1].to(Es[n].dtype)
                fwrite = torch.zeros(Ltot, Q.D, device=dev)
                for t_ in TOOLS: fwrite += Es[t_.name] @ t_.D
                feat = {n: Es[n][None] for n in ('calc',) + tuple(t_.name for t_ in TOOLS)}
                if IFT is not None:
                    fwrite += Es['ifeval']
                    fi_ = Es['ifeval'] / Es['ifeval'].norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    fp_ = torch.zeros_like(fi_)                     # previous position's IF feature (within each sequence)
                    o2 = 0
                    for j, n in enumerate(lens): fp_[o2 + 1:o2 + n] = fi_[o2:o2 + n - 1]; o2 += n
                    feat['ifeval'] = fi_[None]; feat['ifeval_prev'] = fp_[None]
                fwrite = fwrite[None]
                e = feat['calc']
            if TT:                                     # arm C: recompute the BBH tools' features WITH grad (same keep mask)
                fwrite = torch.zeros(Ltot, Q.D, device=dev)
                for k_t, (t_, tt_) in enumerate(zip(TOOLS, TT)):
                    sl_ = full[:, eng.Lt - t_.Kt:]
                    eg_ = t_.M.emit(E0[sl_])[0][:, t_.Kt:].float() * eng.tool_on[1 + k_t]
                    Eg = packed(eg_, t_.n_out) * keep[:, 1 + k_t:2 + k_t].to(torch.float32)
                    feat[t_.name] = Eg[None]; fwrite = fwrite + Eg @ tt_.D_eff().float()
                fwrite = fwrite[None]
        hcap = None
        if INW is not None:
            hn, hcap = Q.fwd_packed(Wt, ids, packer.pos, bm, eng.cos_t, eng.sin_t, leaves, links=None if a.plain else links, feat=feat,
                                    final_write=fwrite, capture=INW.L)
        else:
            hn = Q.fwd_packed(Wt, ids, packer.pos, bm, eng.cos_t, eng.sin_t, leaves, links=links, feat=feat, final_write=fwrite)
        lp = lp_of_hidden(Wt, hn[sel], tgt)
        lp_tok = lp                                                      # token log-probs only (KL uses these)
        l1 = 0.0
        for k_, w_ in _all_cw():                                       # gate L1 + per-wire gate stats
            if w_.last is None: continue
            g_, m_, on_ = w_.last; g_ = g_.reshape(-1); m_ = m_.reshape(-1); on_ = on_.reshape(-1)
            if m_.any():
                gm = g_[m_]
                w_._l1g = getattr(w_, '_l1g', 0.0) + float((gm * (1 - gm)).detach().mean()); w_._l1n = getattr(w_, '_l1n', 0) + 1
                acc = cw_acc.setdefault(str(k_), [0.0, 0.0, 0, 0.0, 0.0, 0])
                acc[0] += float(gm.detach().sum()); acc[1] += float((gm.detach() > 0.5).sum()); acc[2] += int(m_.sum())
                acc[3] += float(on_[m_].sum()); acc[4] += float(on_[~m_].sum()); acc[5] += int((~m_).sum())
            w_.last = None
        if hcap is not None and dpos:                                   # in-wire decision log-probs (with grad)
            dp_ = torch.tensor(dpos, device=dev); di_ = torch.tensor(didx, device=dev); ds_ = torch.tensor(dsgn, device=dev)
            dt_ = torch.tensor(dtool, device=dev)
            logit = INW.k * 0.5 + INW.b[dt_] + (_ln(hcap[dp_ - 1]) * INW.v[dt_]).sum(-1)
            for k_ in range(NT):
                mk = dt_ == k_
                st.setdefault('n_dec_t', [0] * NT)[k_] += int(mk.sum()); st.setdefault('n_veto_t', [0] * NT)[k_] += int((mk & (ds_ < 0)).sum())
            if INW.eps == 0.0:
                lp = lp.index_add(0, di_, F.logsigmoid(ds_ * logit))
            else:                                                             # same floor as the rollout
                pf = INW.eps + (1 - 2 * INW.eps) * torch.sigmoid(logit)
                lp = lp.index_add(0, di_, torch.where(ds_ > 0, torch.log(pf), torch.log1p(-pf)))
            st['n_dec'] = st.get('n_dec', 0) + len(dpos); st['n_veto'] = st.get('n_veto', 0) + int((ds_ < 0).sum())
        if feat is not None:
            enz = (feat['calc'][0, sel].abs().sum(-1) > 0)                    # implant fires at this predictor position
            n_e += int(enz.sum()); n_sel += int(sel.numel())
            if enz.any(): st['_pf'] = st.get('_pf', 0.0) + float(torch.exp(lp_tok.detach()[enz]).sum())
        def _diag():                                                  # run AFTER backward (in-place gate zeroing)
            if not (a.diag and links is not None and feat is not None and 'diag_dlp' not in st): return
            with torch.no_grad():
                sv = _zero_gates_ctx()
                h0 = Q.fwd_packed(Wt, ids, packer.pos, bm, eng.cos_t, eng.sin_t, [(k.detach(), v.detach()) for k, v in leaves],
                                  links=links, feat=feat, final_write=fwrite)
                lp0 = lp_of_hidden(Wt, h0[sel], tgt); _restore_gates(sv)
                dl = (lp_d - lp0).abs()
                st['diag_dlp_fire'] = float(dl[enz].mean()) if enz.any() else 0.0
                st['diag_dlp_nofire'] = float(dl[~enz].mean()) if (~enz).any() else 0.0
                st['diag_dlp'] = float(dl.max())
        lp_d = lp.detach()
        if a.sft:
            per = -lp; loss = per.sum() / tot_tok; loss.backward(); st['loss'] += float(loss.detach()); _diag(); continue
        old = torch.cat([torch.tensor(old_lp[i], device=dev) for i in mb])
        A = torch.cat([torch.full((len(batch_r[i]),), adv[i], device=dev) for i in mb])
        ratio = torch.exp(torch.clamp(lp - old, -20, 20))
        if a.loss == 'cispo':
            per = -torch.clamp(ratio, max=a.cispo_max).detach() * A * lp
        else:
            per = -torch.minimum(ratio * A, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A)
        if W_ref is not None:
            with torch.no_grad():
                hr = Q.fwd_packed(W_ref, ids, packer.pos, bm, eng.cos_t, eng.sin_t, pkv_ref, links=links, feat=feat, final_write=fwrite)
                rlp = lp_of_hidden(W_ref, hr[sel], tgt)
            d = rlp - lp_tok; klt = torch.exp(d) - d - 1
            per = per + a.kl * klt; st['kl'] += float(klt.detach().sum()) / tot_tok
        loss = per.sum() / norm + l1
        loss.backward()
        st['loss'] += float(loss.detach())
        _diag()
        with torch.no_grad():
            st['ratio'] += float(ratio.sum()) / tot_tok
            st['clipfrac'] += float(((ratio - 1).abs() > a.clip).float().sum()) / tot_tok
    if not a.freeze_host:
        torch.autograd.backward([t for kv in pkv for t in kv], [t.grad if t.grad is not None else torch.zeros_like(t) for kv in leaves for t in kv])
    st['frac_e'] = n_e / max(1, n_sel)
    if '_pf' in st: st['p_fire'] = st.pop('_pf') / max(1, n_e)          # mean p(sampled token) where the implant writes
    if cw_acc:
        st['cw_gate_mean'] = {k: v[0] / max(1, v[2]) for k, v in cw_acc.items()}
        st['cw_gate_open'] = {k: v[1] / max(1, v[2]) for k, v in cw_acc.items()}
        st['cw_B_norm'] = {str(k): float(l.B.detach().norm()) for k, l in _all_cw()}
        st['cw_out_fire'] = {k: v[3] / max(1, v[2]) for k, v in cw_acc.items()}       # mean ||wire output|| where e != 0
        st['cw_out_silent'] = {k: v[4] / max(1, v[5]) for k, v in cw_acc.items()}     # ... and where the implant is silent
    if INW is not None:
        st['iw_v_norm'] = INW.v.detach().norm(dim=-1).tolist(); st['iw_b'] = INW.b.detach().tolist()
    if links:                                                        # pre-clip grad norms + displacement from init
        gg, gw, dg, dw = {}, {}, {}, {}
        for k, l in _all_links():
            n = f'{l.src}@{k}'
            while n in gg: n += "'"
            gg[n] = float(l.gate.grad.norm()) if l.gate.grad is not None else -1.0
            gw[n] = float(l.W.grad.norm()) if l.W.grad is not None else -1.0
        st['link_grad_gate'], st['link_grad_W'] = gg, gw
    tp = ([] if a.freeze_host else [P32[k] for k in names]) + link_params + ([Dp] if TRAIN_IMPL else [])
    gn = torch.nn.utils.clip_grad_norm_(tp, 1.0)
    if TT:                                             # per-group gradient norms (which implant parts get real gradient)
        st['impl_grad'] = {}
        for tt_ in TT:
            for gr_ in tt_.groups(0, 0, 0, 0):
                gs_ = [p_.grad.detach().float().norm() for p_ in gr_['params'] if p_.grad is not None]
                st['impl_grad'][gr_['name']] = float(torch.stack(gs_).norm()) if gs_ else 0.0
        torch.nn.utils.clip_grad_norm_(IMPL_PARAMS, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    for tt_ in TT: tt_.refresh()
    with torch.no_grad():   # decoupled L1 on out-wire gates (plain SGD on b_g: lr * lambda * mean g(1-g)), not via Adam
        for _, w_ in _all_cw():
            if getattr(w_, '_l1n', 0):
                w_.bg -= CW_LR * min(1.0, 1.0) * a.cw_l1 * (w_._l1g / w_._l1n); w_._l1g = 0.0; w_._l1n = 0
    st['grad_norm'] = float(gn); st['train_s'] = time.time() - t0
    return st


@torch.no_grad()
def hf_check(qids, out, G, nrows=48):
    """Independent host path: HF transformers Qwen3 (bf16, sdpa) teacher-forced on sampled rows, with the tools' features
    recomputed teacher-forced (Micromachine.emit, not the decode-step code), recorded in-wire decisions replayed, the
    adapters added by forward pre-hooks (L14 input, final-norm input) and the fixed writes at the final-norm input.
    Compared on tokens WITHOUT an in-wire decision (decision log-probs are not a host quantity)."""
    hfm = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
    rows = list(range(0, len(out['tokens']), max(1, len(out['tokens']) // nrows)))[:nrows]
    gaps, gaps_fire, n_any_fire = [], [], 0; flag_mis, flag_n = [], []
    for j in rows:
        q = qids[j // G]; r = out['tokens'][j]; d = out['dec'][j].to(torch.int64)
        seq = torch.tensor(eng.prefix_ids.tolist() + q + r, device=dev)[None]
        seqt = torch.cat([eng.tool_prefix[:256][None], seq], 1)        # same padded stream the engine's tools see
        L0 = Lp + len(q)
        T_ = seq.shape[1]
        e_c = imp.full(W['E'][seqt])[:, 256:]                         # calc (live host embedding, as in decode)
        Es = {'calc': e_c[0] * eng.tool_on[0]}
        for t_ in TOOLS: Es[t_.name] = t_.full(E0[seqt])[0, 256:] * eng.tool_on[1 + TOOLS.index(t_)]
        if IFT is not None:
            wif = torch.zeros(T_, Q.D, device=dev); pz, vz = out['ifw'][j]
            if len(pz): wif[L0 - 1 + pz.to(dev)] = vz.to(dev)
            Es['ifeval'] = wif
        keep = torch.ones(T_, NT, dtype=torch.bool, device=dev); sup = torch.zeros(NT, dtype=torch.bool, device=dev)
        nz = torch.stack([(Es[n].abs().sum(-1) > 0) for n in TNAMES], -1)
        for t in range(len(r)):
            pp = L0 - 1 + t; d_ = d[t].to(dev)
            sup = torch.where(d_ == 2, True, torch.where(d_ == 1, False, sup)) & nz[pp]
            keep[pp] = ~sup
        for k_, n in enumerate(TNAMES): Es[n] = Es[n] * keep[:, k_:k_ + 1]
        feat = {n: Es[n][None] for n in TNAMES if n != 'ifeval'}
        fw = torch.zeros(T_, Q.D, device=dev)
        for t_ in TOOLS: fw += Es[t_.name] @ t_.D
        if IFT is not None:
            fw += Es['ifeval']; fi_ = Es['ifeval'] / Es['ifeval'].norm(dim=-1, keepdim=True).clamp(min=1e-12)
            fp_ = torch.zeros_like(fi_); fp_[1:] = fi_[:-1]; feat['ifeval'] = fi_[None]; feat['ifeval_prev'] = fp_[None]
        def pre14(m, args, kwargs):
            if args:
                x_ = args[0]; return (x_ + links[14](feat, x_).to(x_.dtype),) + tuple(args[1:]), kwargs
            x_ = kwargs['hidden_states']; kwargs['hidden_states'] = x_ + links[14](feat, x_).to(x_.dtype); return args, kwargs
        h1 = hfm.model.layers[14].register_forward_pre_hook(pre14, with_kwargs=True)
        h2 = hfm.model.norm.register_forward_pre_hook(lambda m, args: (args[0] + links['final'](feat, args[0]).to(args[0].dtype) + fw[None].to(args[0].dtype),))
        try:
            lg = hfm(seq).logits[0].float() / a.T
        finally:
            h1.remove(); h2.remove()
        lp = torch.log_softmax(lg[L0 - 1:L0 - 1 + len(r)], -1).gather(-1, torch.tensor(r, device=dev)[:, None])[:, 0]
        samp = torch.tensor(out['logp'][j], device=dev)
        nodec = ~(d[:len(r)] > 0).any(-1).to(dev)
        fired = (keep[L0 - 1:L0 - 1 + len(r)] & nz[L0 - 1:L0 - 1 + len(r)]).any(-1)
        wt_dec = out['wt'][j][:len(r)].to(dev) > 0                            # decode-time per-tool write flags
        wt_tf = (keep & nz)[L0 - 1:L0 - 1 + len(r)]
        wt_tf[0] = wt_dec[0]                                                 # first token: prefill path (IF not computed there)
        flag_mis.append(int((wt_dec != wt_tf).sum())); flag_n.append(int(wt_tf.sum()))
        gp = (lp - samp).abs()
        gaps.append(gp[nodec]); gaps_fire.append(gp[nodec & fired]); n_any_fire += int((nodec & fired).sum())
    g_ = torch.cat(gaps); gf = torch.cat(gaps_fire)
    q_ = lambda x, p: float(x.quantile(p)) if x.numel() else None
    res = dict(rows=len(rows), tokens=int(g_.numel()), mean=float(g_.mean()), p99=q_(g_, 0.99), max=float(g_.max()),
               tool_flag_mismatch=sum(flag_mis), tool_flag_events=sum(flag_n), fire_tokens=int(gf.numel()), fire_mean=float(gf.mean()) if gf.numel() else None, fire_max=float(gf.max()) if gf.numel() else None)
    print('HF_CHECK', res, flush=True)
    del hfm; torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ main loop
DEV_EXTRA = {int(x) for x in a.dev_extra.split(',') if x}
t_start = time.time()
if a.eval_ckpt:            # one-shot test evaluation of a saved checkpoint (wiring as trained / hand-only / implant off)
    from safetensors.torch import load_file
    sd = load_file(f'{a.eval_ckpt}/model.safetensors', device=dev)
    with torch.no_grad():
        for k in names: P32[k].copy_(sd[k].float())
    if links and os.path.exists(f'{a.eval_ckpt}/links.pt'):
        ls = torch.load(f'{a.eval_ckpt}/links.pt', map_location=dev)
        for k, lk in links.items(): lk.load_state_dict(ls[str(k)], strict=False)
    if INW is not None and os.path.exists(f'{a.eval_ckpt}/inwire.pt'):
        INW.load_state_dict(torch.load(f'{a.eval_ckpt}/inwire.pt', map_location=dev))
    if TRAIN_IMPL and os.path.exists(f'{a.eval_ckpt}/Dp.pt'):
        Dp.data.copy_(torch.load(f'{a.eval_ckpt}/Dp.pt').to(dev))
    sync_weights()
    if a.misfire_probe:                     # sampled rollouts (T, G) on TRAIN prompts: reward with vs without an off-type call
        rs_ = random.Random(99); out_mp = {}
        for t_, tools_ in (('ifeval', ['calc', 'expr_gen', 'track_gen']), ('bbh', ['calc'])):
            items = [r for r in pool if r['type'] == t_]; rs_.shuffle(items); items = items[:a.misfire_probe]
            o_ = eng.generate([QID(r) for r in items], G=a.G, T=a.T, max_new=a.max_new)
            R_ = [SCORE(t, items[j // a.G])[0] for j, t in enumerate(o_['tokens'])]
            for nm in tools_:
                k_ = TNAMES.index(nm)
                fired = [bool((d[:, k_] == 1).any()) for d in o_['dec']]
                yes = [r for r, f in zip(R_, fired) if f]; no = [r for r, f in zip(R_, fired) if not f]
                # within-group comparison: prompts with both kinds of rows
                diffs = []
                for p_ in range(len(items)):
                    g = range(p_ * a.G, (p_ + 1) * a.G); y = [R_[j] for j in g if fired[j]]; n = [R_[j] for j in g if not fired[j]]
                    if y and n: diffs.append(sum(y) / len(y) - sum(n) / len(n))
                out_mp[f'{nm}@{t_}'] = dict(rows=len(R_), rows_with_call=len(yes), mean_reward_with=sum(yes) / max(1, len(yes)),
                                           mean_reward_without=sum(no) / max(1, len(no)), groups_mixed=len(diffs),
                                           within_group_diff=sum(diffs) / max(1, len(diffs)))
                print('MISFIRE', nm, t_, out_mp[f'{nm}@{t_}'], flush=True)
        json.dump(out_mp, open(f'{a.run_dir}/misfire_probe.json', 'w'), indent=1); sys.exit(0)
    out_res = {'ckpt': a.eval_ckpt, 'params': PARAMS}
    if a.interp and INW is not None:
        import numpy as np
        o = eng.generate([C.q_ids(tok, r['q']) for r in devset], G=1, greedy=True, max_new=a.max_new)
        OPS = {'+': '+', '-': '-', '*': '*', 'x': '*', '\u00d7': '*', '/': '/', '\u00f7': '/'}
        rows = []                                    # one row per generated position i >= 1
        for t_, d_, op_, (lg_, rn_, sp_), r_ in zip(o['tokens'], o['dec'], o['ops'], o['iw'], devset):
            cur_op = None
            for i in range(1, len(t_)):
                prev = tok.decode([t_[i - 1]]).strip()           # token the in-wire step has just read
                if prev and not all(c in '0123456789+-*/x\u00d7\u00f7$.,()=' for c in prev): cur_op = None
                elif prev in OPS: cur_op = OPS[prev]
                elif prev == '=': pass
                rows.append(dict(lg=lg_[i], run=rn_[i], span=sp_[i], op=cur_op or 'none', dec=d_[i] & 3, ops=op_[i] & 15, top=op_[i] >> 4,
                                 isdig=prev.isdigit(), iseq=(prev == '='), kind=r_['kind']))
        L = np.array([x['lg'] for x in rows]); F_ = {}
        F_['span_digits'] = np.array([x['span'] for x in rows], float); F_['run_pos_in_number'] = np.array([x['run'] for x in rows], float)
        F_['prev_is_digit'] = np.array([x['isdig'] for x in rows], float); F_['prev_is_eq'] = np.array([x['iseq'] for x in rows], float)
        for op in ('+', '-', '*', '/', 'none'): F_[f'op_{op}'] = np.array([x['op'] == op for x in rows], float)
        F_['scaled_problem'] = np.array([x['kind'] == 'scaled' for x in rows], float)
        corr = {k: float(np.corrcoef(L, v)[0, 1]) if v.std() > 0 else 0.0 for k, v in F_.items()}
        dm = np.array([x['dec'] > 0 for x in rows])
        Ld = L[dm]; corr_dec = {}
        Fd = dict(ops_digits=np.array([x['ops'] for x in rows], float)[dm], scaled_problem=F_['scaled_problem'][dm])
        for op in ('+', '-', '*', '/'): Fd[f'op_{op}'] = F_[f'op_{op}'][dm]
        corr_dec = {k: float(np.corrcoef(Ld, v)[0, 1]) if v.std() > 0 else 0.0 for k, v in Fd.items()}
        bucket = lambda n: '0' if n <= 0 else ('1-2' if n <= 2 else ('3-4' if n <= 4 else ('5-6' if n <= 6 else '7+')))
        by_b, by_op = {}, {}
        for x in rows:
            if x['dec'] > 0:
                by_b.setdefault(bucket(x['ops']), []).append((x['lg'], x['dec'] == 2))
                by_op.setdefault(x['op'], []).append((x['lg'], x['dec'] == 2))
        summ = lambda d: {k: dict(n=len(v), mean_logit=float(np.mean([a_ for a_, _ in v])), veto=float(np.mean([b_ for _, b_ in v]))) for k, v in d.items()}
        out_res['interp'] = dict(n_pos=len(rows), n_dec=int(dm.sum()), b=float(INW.b), v_norm=float(INW.v.norm()),
                                 corr_all=dict(sorted(corr.items(), key=lambda kv: -abs(kv[1]))),
                                 corr_decisions=dict(sorted(corr_dec.items(), key=lambda kv: -abs(kv[1]))),
                                 decisions_by_operand_digits=summ(by_b), decisions_by_op=summ(by_op),
                                 mean_logit_all=float(L.mean()), mean_logit_dec=float(Ld.mean()) if len(Ld) else None)
        print('INTERP', json.dumps(out_res['interp'], indent=1), flush=True)
    modes = ['full'] + (['cw_hand'] if CW_ARM else []) + (['hostonly'] if imp is not None else [])
    for name in a.test_sets.split(','):
        items = C.load_test(name, shot_q=shot_q)
        n0 = len(items); items = [dict(r, kind='gsm') for r in items if len(C.q_ids(tok, r['q'])) + a.max_new <= 544]
        if a.test_n: items = items[:a.test_n]
        for md in modes:
            r = evaluate(items, mode=md, tag='t'); out_res[f'{name}_{md}'] = dict(acc=r['t_acc'], n=len(items), n_all=n0,
                calls=r.get('t_calls'), vetoes=r.get('t_vetoes'), veto_by_ops=r.get('t_veto_by_ops'))
            print(name, md, out_res[f'{name}_{md}'], flush=True)
    json.dump(out_res, open(f'{a.run_dir}/test_eval.json', 'w'), indent=1)
    print('DONE', flush=True); sys.exit(0)
if a.sft:      # supervised sanity test: CE on gold solutions (calc-heavy problems), params in the optimizer only
    from micromachines.calcfmt import strip_solution
    tr = [json.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')]
    rows = []
    if a.sft_kind in ('scaled', 'both'):
        for l in open(f'{C.REPO}/data/gsm8k/train_scaled.jsonl'):
            r = json.loads(l); src = tr[r['src']]['question']
            if C.part(src) == 'A' and src not in shot_q: rows.append((r['question'], r['answer']))
    if a.sft_kind in ('gsm', 'both'):
        rows += [(r['question'], r['answer']) for r in tr if C.part(r['question']) == 'A' and r['question'] not in shot_q]
    rs = random.Random(7); rs.shuffle(rows)
    data = []
    if a.sft_traces:                                  # calculator-made correct traces (gold solutions have no equations
        rows = []                                     # for the scaled set; the scaled jsonl holds only '#### n')
        for l in open(a.sft_traces):
            r = json.loads(l)
            if a.sft_kind == 'both' or r['kind'] == a.sft_kind:
                data.append((C.q_ids(tok, r['q']), r['resp']))
        data = data[:a.sft_n]
    for q, ans in rows:
        body, fin = strip_solution(ans)
        qi = C.q_ids(tok, q); ri = tok(' ' + body + '\n#### ' + fin, add_special_tokens=False)['input_ids']
        if len(qi) + len(ri) <= 544: data.append((qi, ri))
        if len(data) >= a.sft_n: break
    print(f'SFT: {len(data)} problems ({a.sft_kind}), {a.sft} steps x {a.prompts}', flush=True)
    r = evaluate(devset); r.update(u=0); log({'dev': r}); print('dev@0', r, flush=True)
    r = evaluate(devset, mode='hostonly', tag='devhost'); r.update(u=0); log({'dev_host_curve': r}); print('host-only@0', r, flush=True)
    for u in range(1, a.sft + 1):
        for g, b in zip(opt.param_groups, base_lrs): g['lr'] = b * min(1.0, u / a.warmup)
        bt = rs.sample(data, a.prompts)
        st = train_step([x[0] for x in bt], [x[1] for x in bt], None, None, len(bt))
        if not a.freeze_host: sync_weights()
        rec = dict(u=u, t=time.time() - t_start, **st)
        if links: rec['link_gate_l2'], rec['link_W_l2'], rec['link_eff'] = link_norms()
        log(rec)
        if u % 10 == 0 or u == 1:
            print(json.dumps({k: rec[k] for k in ('u', 'loss', 'frac_e', 'grad_norm') if k in rec}),
                  {k: round(v, 3) for k, v in rec.get('link_eff', {}).items()}, flush=True)
        if u % a.dev_every == 0 or u == a.sft:
            r = evaluate(devset); r.update(u=u); log({'dev': r}); print('dev', r, flush=True)
    r = evaluate(devset, mode='hostonly', tag='devhostonly'); r.update(u=a.sft); log({'dev_hostonly': r}); print('host-only', r, flush=True)
    if links: torch.save({str(k): lk.state_dict() for k, lk in links.items()}, f'{a.run_dir}/links_final.pt')
    log({'DONE_RUN': time.time() - t_start}); print('DONE', flush=True); sys.exit(0)
log({'params': PARAMS})
if a.dump_ref:                              # bit-exactness reference for a new engine: fixed prompts, greedy outputs, per-step tool flags
    rs_ = random.Random(7); by_ = {}
    for r in TESTSET: by_.setdefault(r['type'], []).append(r)
    items = []
    for t_ in sorted(by_): rs_.shuffle(by_[t_]); items += by_[t_][:a.dump_ref // len(by_)]
    o_ = eng.generate([QID(r) for r in items], G=1, greedy=True, max_new=a.max_new)
    ref = dict(host=a.host, max_new=a.max_new, tnames=TNAMES, prefix_ids=pre, q_ids=[QID(r) for r in items], items=items,
               tokens=o_['tokens'], logp=o_['logp'], tool_fire_flags=[w.tolist() for w in o_['wt']],
               inwire_dec=[d.tolist() for d in o_['dec']], ifeval_write_pos=[p_.tolist() for p_, _ in o_['ifw']],
               ifeval_write_vec=[v_.tolist() for _, v_ in o_['ifw']], texts=[tok.decode(t) for t in o_['tokens']],
               notes='greedy; tool_fire_flags[t][i]=1 iff tool i (tnames order) wrote at the step producing token t (post in-wire); '
                     'inwire_dec 1 fire / 2 veto at burst starts; in-wires at init (fire at greedy); wires zero-init')
    json.dump(ref, open(f'{a.run_dir}/engine_reference.json', 'w')); print('DUMPED', len(items), flush=True); sys.exit(0)
if a.probe_tool:                            # u0 leave-one-out of one tool on its own items, outputs saved
    nm_, sub_ = a.probe_tool.split(':')
    items = [r for r in TESTSET if r['type'] in ('bbh', 'combo') and r.get('task', r.get('sub', '')).startswith(sub_)]
    resp = {}
    for md in ('full', 'loo_' + nm_):
        eng.tool_on.fill_(1.0)
        if md != 'full': eng.tool_on[TNAMES.index(nm_)] = 0.0     # leave-one-out (the probe calls generate directly)
        o_ = eng.generate([QID(r) for r in items], G=1, greedy=True, max_new=a.max_new)
        sc_ = [SCORE(t, r) for t, r in zip(o_['tokens'], items)]
        resp[md] = dict(acc=100 * sum(x[0] for x in sc_) / len(items), n=len(items),
                        texts=[tok.decode(t) for t in o_['tokens']], ok=[x[0] for x in sc_])
        print('PROBE', md, resp[md]['acc'], len(items), flush=True)
    json.dump(dict(items=items, **resp), open(f'{a.run_dir}/probe_{nm_}.json', 'w')); sys.exit(0)
if a.bench:                                 # per-component decode-step cost (CUDA-graph replay), then exit
    def gtime(fn, n=20):
        s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            for _ in range(2): fn()
        torch.cuda.current_stream().wait_stream(s_)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        torch.cuda.synchronize(); t0_ = time.time()
        for _ in range(n): g.replay()
        torch.cuda.synchronize(); return (time.time() - t0_) / n * 1000
    _ = eng.generate([C.q_ids(tok, r['q']) for r in devset[:2]], G=1, greedy=True, max_new=8)   # init IF static state
    for Bq in (16, 64, 128):
        if Bq > a.B_max or a.bench == 2: continue
        emb0 = E0[torch.randint(0, 150000, (Bq,), device=dev)]; embc = W['E'][torch.randint(0, 150000, (Bq,), device=dev)]
        res_ = {'calc': gtime(lambda: imp.step(embc, Bq))}
        for t_ in TOOLS: res_[t_.name] = gtime(lambda t_=t_: t_.step(emb0, Bq))
        if Bq == 16:
            for t_ in TOOLS:
                st1 = getattr(t_.M.stack1, 'net', t_.M.stack1); st2 = t_.M.stack2
                print('LAYERS', t_.name, getattr(st1, 'L', '?'), getattr(st2, 'L', '?'), type(t_.M.stack1).__name__, flush=True)
        print('BENCH B', Bq, {k: round(v, 2) for k, v in res_.items()}, 'sum', round(sum(res_.values()), 1), flush=True)
    for md in ('hostonly', 'full'):
        qs_ = [C.q_ids(tok, r['q']) for r in devset[:max(1, a.B_max // a.G)]]
        if md == 'hostonly': eng.imp_on = False

        eng.generate(qs_, G=a.G, T=1.0, max_new=a.max_new)
        o_ = eng.generate(qs_, G=a.G, T=1.0, max_new=a.max_new); eng.imp_on = True
        print('BENCH generate', md, o_['stats'], flush=True)
    sys.exit(0)
t_dev0 = time.time()
res0 = evaluate(devset); res0.update(u=0, t=0); log({'dev': res0}); print('dev@0', res0, flush=True); LAST_DEV = res0
T_DEV = time.time() - t_dev0
if imp is not None:
    r = evaluate(devset, mode='hostonly', tag='devhost'); r.update(u=0); log({'dev_host_curve': r}); print('host-only@0', r, flush=True)
    log_h0 = r['devhost_hash']
    if HAS_EXTRA_LINKS:
        r = evaluate(devset, mode='linksoff', tag='devlinksoff'); r.update(u=0); log({'dev_linksoff_curve': r})
if a.arm == 'LIB':
    r = evaluate(devset, mode='toolsoff', tag='devtoolsoff'); r.update(u=0); log({'dev_toolsoff0': r})
    print('BITID u0: tools-off hash', r['devtoolsoff_hash'], 'host-only hash', log_h0, 'identical', r['devtoolsoff_hash'] == log_h0, flush=True)
    log({'bitid_u0': dict(toolsoff=r['devtoolsoff_hash'], hostonly=log_h0, identical=r['devtoolsoff_hash'] == log_h0)})
    if not MIXED:
        r = evaluate(devset, mode='calconly', tag='devcalconly'); r.update(u=0); log({'dev_calconly': r}); print('calc-only@0', r, flush=True)
if TT: d_ = impl_drift(); log({'impl_drift': d_, 'u': 0}); print('IMPL_DRIFT u0', d_, flush=True)
TRACK_DRIFT = TRAIN_IMPL or WARM_FINAL is not None
if TRACK_DRIFT: d0 = writer_drift(); log({'drift': {**d0, 'u': 0}}); print('drift@0', d0, flush=True)
import copy as _copy
INIT_WIRING = dict(links={k: _copy.deepcopy(lk.state_dict()) for k, lk in (links or {}).items()},
                   inw=_copy.deepcopy(INW.state_dict()) if INW is not None else None)
UPD_T = []; P0_PROMPTS = a.prompts
for u in range(1, a.updates + 1):
    for g, b in zip(opt.param_groups, base_lrs): g['lr'] = b * min(1.0, u / a.warmup)
    if a.deadline and (u == 5 or (u > 5 and u % 25 == 1)):   # fit before the deadline: fewer prompts, never fewer updates
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        hh, mm = map(int, a.deadline.split(':'))
        dl = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if dl < now: dl += datetime.timedelta(days=1)
        left = (dl - now).total_seconds()
        t_u = sum(UPD_T[-3:]) / 3 if u == 5 else sum(UPD_T[-10:]) / 10
        t_u = t_u * (P0_PROMPTS / a.prompts) if u > 5 else t_u        # cost per update at the ORIGINAL prompt count
        n_dev = len([x for x in range(u, a.updates + 1) if x % a.dev_every == 0 or x == a.updates or x in DEV_EXTRA])
        final = T_DEV * (n_dev + 7) + T_DEV * 6.5 * (0 if a.no_test else 1)     # dev evals + LOO/ablations + test
        rem = a.updates - u + 1
        need = rem * t_u * (a.prompts / P0_PROMPTS) + final
        P_old = a.prompts
        if need > left:
            P_new = int(P0_PROMPTS * max(0.0, left - final) / (rem * t_u)) // 8 * 8
            a.prompts = max(16, min(P_old, P_new))
        log({'deadline_check': dict(u=u, t_update=t_u, t_dev=T_DEV, left_s=left, need_s=need, prompts_old=P_old, prompts_new=a.prompts)})
        print('DEADLINE', dict(t_update=t_u, t_dev=T_DEV, left_s=left, need_s=need, prompts=a.prompts), flush=True)
    t_upd0 = time.time()
    batch = random.sample(pool, a.prompts)
    qids = [QID(r) for r in batch]
    t0 = time.time()
    out = eng.generate(qids, G=a.G, T=a.T, max_new=a.max_new)
    t_roll = time.time() - t0
    R = []; fmt = 0
    for j, t in enumerate(out['tokens']):
        s = SCORE(t, batch[j // a.G]); R.append(s[0]); fmt += s[1]
    R_raw = list(R)
    if a.call_cost and INW is not None:       # cost-aware variant: pay per fired call
        R = [r_ - a.call_cost * sum(1 for x in d if (x & 3) == 1) for r_, d in zip(R, out['dec'])]
    Rt = torch.tensor(R).view(a.prompts, a.G)
    mean = Rt.mean(1, keepdim=True); std = Rt.std(1, keepdim=True)
    keep = (std[:, 0] > 0)
    if a.loss == 'drgrpo':
        advt = Rt - mean
    else:
        advt = (Rt - mean) / (Rt[keep].std() + 1e-4 if keep.any() else 1.0)
    idx = [p * a.G + g for p in range(a.prompts) if keep[p] for g in range(a.G)]
    kinds = [batch[j // a.G]['kind'] for j in range(len(R))]
    rec = dict(u=u, t=time.time() - t_start, reward=float(Rt.mean()), fmt=fmt / len(R),
               r_gsm=sum(r for r, k in zip(R, kinds) if k == 'gsm') / max(1, kinds.count('gsm')),
               r_scaled=sum(r for r, k in zip(R, kinds) if k == 'scaled') / max(1, kinds.count('scaled')),
               kept_groups=int(keep.sum()), len=sum(map(len, out['tokens'])) / len(R),
               writes=sum(sum(w) for w in out['wflag']) / len(R), calls=sum(calls_of(w) for w in out['wflag']) / len(R),
               roll_s=t_roll, roll_tok_s=out['stats']['tok_s'], gen_tokens=out['stats']['tokens'],
               mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    if u == 1 and a.hf_check and a.arm == 'LIB' and not a.plain:
        try: log({'hf_check': hf_check(qids, out, a.G)})
        except Exception as ex_: print('hf_check failed', repr(ex_), flush=True); log({'hf_check_error': repr(ex_)})
    if idx:
        st = train_step([qids[j // a.G] for j in idx], [out['tokens'][j] for j in idx], [out['logp'][j] for j in idx],
                        [float(advt.view(-1)[j]) for j in idx], len(R),
                        batch_dec=[out['dec'][j] for j in idx] if INW is not None else None,
                        batch_ifw=[out['ifw'][j] for j in idx] if (IFT is not None and not a.plain) else None)
        rec.update(st)
        sync_weights()
    if MIXED:
        for t_ in TYPES:
            jj = [j for j in range(len(R)) if kinds[j] == t_]
            if not jj: continue
            rec[f'r_{t_}'] = sum(R[j] for j in jj) / len(jj)
            if INW is not None and not a.plain:
                sub = {k: [out[k][j] for j in jj] for k in ('dec', 'wt', 'ifw', 'if_lat', 'if_lat_prompt', 'if_ans') if out.get(k)}
                rec[f'roll_tools_{t_}'] = {k[7:]: v for k, v in tool_stats(sub, 'roll').items() if k.startswith('roll_t_')}
                rec[f'roll_if_{t_}'] = {k: v for k, v in tool_stats(sub, 'roll').items() if k.startswith('roll_if_')}
    if INW is not None:                               # rollout per-tool bursts / fire / veto (+ reward of rows with a fired call)
        rec.update(tool_stats(out, 'roll')); rec['reward_raw'] = sum(R_raw) / len(R_raw)
        for k_, nm in enumerate(TNAMES):
            fired = [j for j in range(len(R)) if bool((out['dec'][j][:, k_] == 1).any())]
            rec[f'roll_t_{nm}']['r_fired'] = sum(R_raw[j] for j in fired) / max(1, len(fired))
            rec[f'roll_t_{nm}']['n_rows_fired'] = len(fired)
    if links:
        rec['link_gate_l2'], rec['link_W_l2'], rec['link_eff'] = link_norms()
    if WARM_FINAL is not None and WARM_FINAL.log_gain is not None: rec['gain'] = float(torch.exp(WARM_FINAL.log_gain.detach()))
    if TRAIN_IMPL:
        rec['Dp_delta'] = float((Dp.detach() - imp.D.float()).norm() / imp.D.float().norm())
    UPD_T.append(time.time() - t_upd0); rec['upd_s'] = UPD_T[-1]; rec['prompts'] = a.prompts
    log(rec)
    if u % 5 == 0 or u == 1:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                          if k not in ('link_W_l2',)}), flush=True)
    if u % a.dev_every == 0 or u == a.updates or u in DEV_EXTRA:
        r = evaluate(devset); r.update(u=u, t=time.time() - t_start); log({'dev': r}); print('dev', r, flush=True); LAST_DEV = r
        if CW_ARM and a.arm != 'LIB' and (u % 100 == 0 or u == a.updates):
            for md in ('cw_hand', 'cw_out', 'cw_in', 'cw_mask21'):
                if md == 'cw_out' and not list(_all_cw()): continue
                if md == 'cw_mask21' and (a.cw_mask or not any(k == 21 for k, _ in _all_cw())): continue
                if md == 'cw_in' and INW is None: continue
                r = evaluate(devset, mode=md, tag='dev' + md); r.update(u=u); log({f'dev_{md}': r}); print(md, r, flush=True)
        if imp is not None and a.eval_all and (links or TRAIN_IMPL):
            r = evaluate(devset, mode='reset', tag='devreset'); r.update(u=u); log({'dev_reset_curve': r}); print('wiring-reset', r, flush=True)
        if imp is not None and u != a.updates and (a.eval_all or u in DEV_EXTRA or u % 100 == 0):
            r = evaluate(devset, mode='hostonly', tag='devhost'); r.update(u=u); log({'dev_host_curve': r}); print('host-only', r, flush=True)
            if HAS_EXTRA_LINKS:
                r = evaluate(devset, mode='linksoff', tag='devlinksoff'); r.update(u=u); log({'dev_linksoff_curve': r}); print('links-off', r, flush=True)
        if TRACK_DRIFT:
            d = writer_drift(); log({'drift': {**d, 'u': u}}); print('drift', {k: d[k] for k in d if 'digits' not in k}, flush=True)
    if TT and (u % 25 == 0 or u == a.updates):
        d_ = impl_drift(); log({'impl_drift': d_, 'u': u}); print('IMPL_DRIFT', u, {k: round(v.get('exact', -1), 4) for k, v in d_.items()}, flush=True)
    if u % a.ckpt_every == 0 or u == a.updates:
        save(f'step_{u:06d}')
# end-of-run ablations
if MIXED and a.no_test:
    log({'DONE_RUN': time.time() - t_start}); print('DONE (pilot, no test)', time.time() - t_start, flush=True); sys.exit(0)
if MIXED:                                # final: full TEST (999) with tools on / all off / host only / leave-one-out
    tres = {}
    if a.test_n: TESTSET = TESTSET[::max(1, len(TESTSET) // a.test_n)][:a.test_n]
    r = evaluate(TESTSET, mode='full', tag='test'); tres['full'] = r; print('TEST full', {k: v for k, v in r.items() if not isinstance(v, dict)}, flush=True)
    if not a.plain:
        for md in ('toolsoff', 'hostonly'):
            r = evaluate(TESTSET, mode=md, tag='test'); tres[md] = r; print('TEST', md, {k: v for k, v in r.items() if not isinstance(v, dict)}, flush=True)
        for nm in TNAMES:
            if tres['full'].get(f'test_t_{nm}', {}).get('nb', 0) == 0:
                tres['loo_' + nm] = 'identical_by_construction'; continue
            r = evaluate(TESTSET, mode='loo_' + nm, tag='test'); tres['loo_' + nm] = r; print('TEST loo', nm, {k: v for k, v in r.items() if not isinstance(v, dict)}, flush=True)
    if not a.plain:                          # swap splits: trained host + ORIGINAL implants/wiring; ORIGINAL host + trained implants/wiring
        svw = _swap_wiring(True); svi = _swap_impl() if TT else None
        r = evaluate(TESTSET, mode='full', tag='test'); tres['trained_host_orig_tools'] = r
        print('TEST trained host + original implants/wiring', {k: v for k, v in r.items() if not isinstance(v, dict)}, flush=True)
        if svi is not None: _restore_impl(svi)
        _restore_wiring(svw)
        h0 = Q.load_hf_params(name=a.host, device='cpu', dtype=torch.float32)
        svh = _swap_host(h0); del h0
        r = evaluate(TESTSET, mode='full', tag='test'); tres['orig_host_trained_tools'] = r
        print('TEST original host + trained implants/wiring', {k: v for k, v in r.items() if not isinstance(v, dict)}, flush=True)
        _swap_host(svh)
    log({'mixed_test': tres}); json.dump(tres, open(f'{a.run_dir}/test_eval.json', 'w'), indent=1, default=str)
    log({'DONE_RUN': time.time() - t_start}); print('DONE', time.time() - t_start, flush=True); sys.exit(0)
if a.arm == 'LIB':                       # leave-one-out: each tool detached (its output forced silent) on the final host
    for nm in TNAMES:
        nb_ = LAST_DEV.get(f'dev_t_{nm}', {}).get('nb', 1)
        if nb_ == 0:                     # no burst on dev: detaching it leaves every token identical by construction
            log({'dev_loo': dict(tool=nm, u=a.updates, identical_by_construction=True, devloo_acc=LAST_DEV['dev_acc'])})
            print('LOO', nm, 'no bursts on dev -> identical to full', flush=True); continue
        r = evaluate(devset, mode='loo_' + nm, tag='devloo'); r.update(u=a.updates, tool=nm); log({'dev_loo': r})
        print('LOO', nm, {k: v for k, v in r.items() if not isinstance(v, dict)}, flush=True)
    r = evaluate(devset, mode='toolsoff', tag='devtoolsoff'); r.update(u=a.updates); log({'dev_toolsoff': r}); print('tools-off', r, flush=True)
    r = evaluate(devset, mode='calconly', tag='devcalconly'); r.update(u=a.updates); log({'dev_calconly': r}); print('calc-only', r, flush=True)
    if not a.no_test:                    # one-shot test at the final update (GSM8K test + GSM-Hard), full library and calc-only
        tres = {}
        for name in a.test_sets.split(','):
            items = C.load_test(name, shot_q=shot_q)
            n0 = len(items); items = [dict(r_, kind='gsm') for r_ in items if len(C.q_ids(tok, r_['q'])) + a.max_new <= 544]
            if a.test_n: items = items[:a.test_n]
            for md in ('full', 'calconly'):
                r = evaluate(items, mode=md, tag='t')
                tres[f'{name}_{md}'] = dict(acc=r['t_acc'], n=len(items), n_all=n0, calls=r.get('t_calls'),
                                            tools={k: v for k, v in r.items() if k.startswith('t_t_') or k.startswith('t_if')})
                print('TEST', name, md, tres[f'{name}_{md}']['acc'], len(items), flush=True)
        log({'test': tres}); json.dump(tres, open(f'{a.run_dir}/test_eval.json', 'w'), indent=1)
if imp is not None:
    r = evaluate(devset, mode='hostonly', tag='devhostonly'); r.update(u=a.updates); log({'dev_hostonly': r}); print('host-only', r, flush=True)
    if HAS_EXTRA_LINKS:
        r = evaluate(devset, mode='linksoff', tag='devlinksoff_final'); r.update(u=a.updates); log({'dev_linksoff_final': r}); print('links-off', r, flush=True)
if TRACK_DRIFT:
    df = writer_drift('drift_final'); log({'drift_final': {**df, 'u': a.updates}}); print('drift-final', {k: df[k] for k in df if 'digits' not in k}, flush=True)
log({'DONE_RUN': time.time() - t_start}); print('DONE', time.time() - t_start, flush=True)
