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
sys.path.insert(0, os.environ.get("WIRING_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wiring')))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='1', choices=['1', '2', 'R', 'W2', 'W3z', 'W3r', 'W1', 'W4', 'W3e', 'W3e3', 'W2w', 'W4w', 'CW', 'CWlean', 'CWout', 'CWin', 'W3d', 'W2ws', 'CWinR', 'CWinP', 'CWinC', 'CWinO', 'WOc', 'WOr', 'WOn', 'CWinB', 'CWall', 'CWcore'])
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
ap.add_argument('--iw_lr', type=float, default=None, help='separate lr for the in-wire (default: same as the CW wires)')
ap.add_argument('--iw_adam_eps', type=float, default=None, help='Adam eps for the in-wire (default: link_eps)')
ap.add_argument('--iw_layer', type=int, default=14, help='in-wire reads the host residual at this layer input')
ap.add_argument('--init_links', default='', help='links state file to start from (W3d: SFT-distilled W3e links)')
ap.add_argument('--calib_traces', default='runs/traces_A_scaled.jsonl', help='W2ws gain calibration traces')
ap.add_argument('--soft_gain', type=float, default=0.0, help='>0: fix the soft writer gain instead of calibrating')
ap.add_argument('--soft_p', type=float, default=0.9, help='W2ws: target mean p(written token) at firing positions')
ap.add_argument('--call_cost', type=float, default=None, help='reward penalty per fired calculator call (CWinC: 0.002)')
ap.add_argument('--partial_thresh', type=int, default=5, help='CWinP: calls with an operand of >= this many digits are deranged')
ap.add_argument('--eval_ckpt', default='', help='checkpoint dir: load host/links/in-wire, run test evals, exit')
ap.add_argument('--test_sets', default='test,hard')
ap.add_argument('--test_n', type=int, default=0, help='limit test items (smoke only); 0 = all')
ap.add_argument('--interp', type=int, default=0, help='with --eval_ckpt: in-wire logit vs operand size / op / position')
ap.add_argument('--host_ckpt', default='', help="with --eval_ckpt: host from here ('base' = untrained); wiring from --eval_ckpt")
ap.add_argument('--break_at', type=int, default=0, help='CWinB: switch calculator correctness after this update')
ap.add_argument('--break_dir', default='on2off', choices=['on2off', 'off2on'], help='on2off: correct then deranged')
ap.add_argument('--ha_width', type=int, default=34, help='WOn host-adapter width (param-matched to CW-full wires)')
ap.add_argument('--link_eps', type=float, default=1e-8, help='Adam eps for link / writer params')
ap.add_argument('--eval_all', type=int, default=0, help='1: at every dev point also eval wiring-reset and implant-off')
ap.add_argument('--freeze_host', type=int, default=0, help='1: host params frozen (only links / writer train)')
ap.add_argument('--sft', type=int, default=0, help='>0: supervised CE on gold solutions for this many steps (no RL)')
ap.add_argument('--sft_traces', default='', help='jsonl from gen_traces.py (q, resp token ids): SFT targets');
ap.add_argument('--sft_n', type=int, default=2000); ap.add_argument('--sft_kind', default='scaled')
ap.add_argument('--diag', type=int, default=0, help='1: in-update check that zeroing link gates changes the train forward')
ap.add_argument('--no_eval', type=int, default=0)
ap.add_argument('--gpu_tag', default='', help='launcher process tag (no effect)')
ap.add_argument('--host_init', default='', help='start the host from this checkpoint dir (model.safetensors), e.g. the call-form SFT host')
ap.add_argument('--core_lr', type=float, default=1e-3, help='CWcore: arithmetic-core lr, RELATIVE to each layer rms(nonzeros)')
ap.add_argument('--core_eps', type=float, default=1e-8)
ap.add_argument('--resume_every', type=int, default=50, help='full resumable checkpoint (host fp32 + wiring + implant deltas + Adam + RNG) every N updates')
ap.add_argument('--resume_dir', default=os.environ.get('WSM_RESUME_DIR', '/dev/shm') + '/cf_resume', help='rolling resume checkpoints (RAM disk: survives process OOM, not a pod restart)')
ap.add_argument('--resume', type=int, default=1, help='1: if a resume checkpoint for this run exists, continue from it')
ap.add_argument('--mem_frac', type=float, default=0.0, help='>0: torch.cuda.set_per_process_memory_fraction (runs sharing a GPU)')
ap.add_argument('--practice_w', type=float, default=0.0, help='>0: auxiliary teacher-forced CE that trains the HOST (logits before the calculator write) to predict the calculator-written tokens')
ap.add_argument('--tool_dropout', type=float, default=0.0, help='P(calculator disabled for a whole rollout row) during RL rollouts (evals: never)')
ap.add_argument('--core_opt', default='adam', choices=['adam', 'sgd'], help='sgd: plain SGD (no momentum) on the implant deltas; core_lr / reader_lr = target RELATIVE weight change of the FIRST step (lr fixed from the first gradient)')
ap.add_argument('--snap', type=int, default=0, help='CWcore: snap reader symbols (round), stack-1 state (round) and stack-2 output (one-hot) in the forward, straight-through backward')
ap.add_argument('--grad_check', type=int, default=0, help='1: one CE backward on calib traces, report per-group grads + module exactness, exit')
ap.add_argument('--exact_every', type=int, default=50)
ap.add_argument('--reader_lr', type=float, default=None, help='CWall: reader lr, RELATIVE to rms(reader_w) (default link_lr)')
ap.add_argument('--warmup', type=int, default=5); ap.add_argument('--fused', type=int, default=1)
a = ap.parse_args()
os.makedirs(a.run_dir, exist_ok=True)
if a.mem_frac > 0:
    torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0); print('memory cap: fraction', a.mem_frac, flush=True)
RESUME_F = os.path.join(a.resume_dir, a.run_dir.strip('/').replace('/', '__'), 'state.pt')
_sf = os.path.join(a.run_dir, 'steps.jsonl')
if not a.eval_ckpt and not a.grad_check and os.path.exists(_sf) and 'DONE_RUN' in open(_sf).read():
    print('run already finished (DONE_RUN in steps.jsonl): refusing to train again in', a.run_dir, flush=True); sys.exit(0)
json.dump(vars(a), open(f'{a.run_dir}/config.json', 'w'), indent=1)
random.seed(a.seed); torch.manual_seed(a.seed)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False

# ---- arm -> config ----
HAS_IMPL = a.arm not in ('1', 'WOn')
# WIRES-ONLY RL (host frozen): WOc = CW-full wires + calculator; WOr = same with the deranged calculator;
#   WOn = no calculator, param-matched h-gated residual adapters at the same layers (7/14/21/final)
WIRES_ONLY = a.arm in ('WOc', 'WOr', 'WOn')
if WIRES_ONLY: a.freeze_host = 1
HAS_LINKS = a.arm in ('W2', 'W3z', 'W3r', 'W4', 'W3e', 'W3e3', 'W2w', 'W4w', 'CW', 'CWlean', 'CWout', 'CWin', 'W3d', 'W2ws',
                      'WOc', 'WOr', 'WOn', 'CWall', 'CWcore')
CW_ARM = a.arm in ('CW', 'CWlean', 'CWout', 'CWin', 'CWinR', 'CWinP', 'CWinC', 'CWinO', 'WOc', 'WOr', 'WOn', 'CWinB', 'CWall', 'CWcore')
HAS_INWIRE = a.arm in ('CW', 'CWlean', 'CWin', 'CWinR', 'CWinP', 'CWinC', 'CWinO', 'WOc', 'WOr', 'CWinB', 'CWall', 'CWcore')
if a.call_cost is None: a.call_cost = 0.002 if a.arm == 'CWinC' else 0.0
if a.cw_hidden is None: a.cw_hidden = 32 if a.arm == 'CWlean' else 64
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
                # ALL-TRAINABLE: host + in-wire + out-adapters (7/14/21/final) + soft writer (W2ws start) + implant reader
                'CWall': 'calc:final:soft:1,calc:7:cw,calc:14:cw,calc:21:cw,calc:final:cw',
                'CWcore': 'calc:final:soft:1,calc:7:cw,calc:14:cw,calc:21:cw,calc:final:cw',   # + arithmetic core trainable
                'CWout': 'calc:final:warm:1,calc:7:cw,calc:14:cw,calc:21:cw,calc:final:cw',
                'CWin': 'calc:final:warm:1',
                'WOc': 'calc:final:warm:1,calc:7:cw,calc:14:cw,calc:21:cw,calc:final:cw',
                'WOr': 'calc:final:warm:1,calc:7:cw,calc:14:cw,calc:21:cw,calc:final:cw',
                'WOn': 'host:7:ha,host:14:ha,host:21:ha,host:final:ha'}
USE_WRITE = a.arm in ('2', 'R', 'W2', 'W1', 'W4', 'CWinR', 'CWinP', 'CWinC', 'CWinO', 'CWinB')
#   CWinO: exact on * and /, deranged on + and - (competence map orthogonal to usefulness-by-size)
# in-wire-only "RL designs the interface" arms: FIXED hand write + trainable in-wire (no out-wires, writer frozen)
#   CWinR: deranged-digit calculator (arm R implant) | CWinP: deranged only when an operand has >= 5 digits
#   CWinC: correct calculator, reward - call_cost per fired call
GATE0 = 1.0 if a.arm == 'W3r' else 0.0
TRAIN_IMPL = a.arm in ('W1', 'W4')
TRAIN_READER = a.arm in ('CWall', 'CWcore')
TRAIN_CORE = a.arm == 'CWcore'     # implant reader (reader_w, reader_b) trainable; arithmetic stacks frozen
IS_R = a.arm in ('R', 'CWinR', 'WOr')

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'))
pre, shot_q = C.prefix_ids(tok); Lp = len(pre)
P32 = Q.load_hf_params(device="cpu", dtype=torch.float32)
for v in P32.values(): v.requires_grad_(not a.freeze_host)
names = list(P32)
if a.host_init:
    from safetensors.torch import load_file as _lf
    _sd = _lf(f'{a.host_init}/model.safetensors')
    with torch.no_grad():
        for k in names: P32[k].copy_(_sd[k].float())
    del _sd; print('host from', a.host_init, flush=True)
W = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16)
W = dict(E=W["E"].cuda(), lnf=W["lnf"].cuda(), L=[{k: v.cuda() for k, v in l.items()} for l in W["L"]])
W_ref = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16) if a.kl > 0 else None
imp = None
if HAS_IMPL:
    hf = AutoModelForCausalLM.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'), dtype=torch.bfloat16).to(dev).eval()
    TAG = os.environ.get('CALC_MACHINE', '') == 'calc_tag'
    if TAG and a.fused: print('calc_tag: fused kernel off (token filter runs in the torch step)', flush=True); a.fused = 0
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
    del hf; torch.cuda.empty_cache()
class SpLin(torch.autograd.Function):
    """y = A x for a fixed-pattern sparse A (COO idx, values `vals`), with our own backward: d x = A^T d y via a plain
    (no-grad) sparse matmul, d vals = sum_n dy[n, row] * x[n, col] in blocks. torch's COO sparse.mm backward hits a
    cudaErrorIllegalAddress on H100 (torch 2.8 / CUDA 12.9) at update 2; this path avoids it."""
    @staticmethod
    def forward(ctx, vals, x, idx, shape):
        ctx.save_for_backward(vals, x, idx); ctx.shape = shape
        A_ = torch.sparse_coo_tensor(idx, vals.detach(), shape, is_coalesced=True)
        return torch.sparse.mm(A_, x.detach().T).T
    @staticmethod
    def backward(ctx, gy):
        vals, x, idx = ctx.saved_tensors; shape = ctx.shape; gv = gx = None
        if ctx.needs_input_grad[1]:
            At = torch.sparse_coo_tensor(idx.flip(0), vals.detach(), (shape[1], shape[0])).coalesce()
            gx = torch.sparse.mm(At, gy.T).T
        if ctx.needs_input_grad[0]:
            r_, c_ = idx[0], idx[1]; gv = torch.empty_like(vals); B_ = max(1024, (1 << 25) // max(1, gy.shape[0]))
            for k in range(0, vals.numel(), B_):
                gv[k:k + B_] = (gy[:, r_[k:k + B_]] * x[:, c_[k:k + B_]]).sum(0)
        return gv, gx, None, None


RD = RD0 = RDD = CORED = None
# Trainable implant = MULTIPLICATIVE per-entry deltas (so every lr is relative to each entry's own size; the compiled
# circuit mixes entries of size 1 and 5e5 in one layer, and a per-layer relative lr wrecked it in 5 updates):
#   weight = w0 * (1 + d),  bias = b0 + d * max(|b0|, 1).  RD / CORE hold the EFFECTIVE values (rollout, eval, save,
#   reset); the d tensors are the optimizer's parameters; materialize() writes effective = f(d) after every step.
if TRAIN_READER:
    RD = [imp.M.reader_w, imp.M.reader_b]
    RD0 = [t.detach().clone() for t in RD]
    RDD = [torch.zeros_like(t, requires_grad=True) for t in RD]
    RBS = RD0[1].abs().clamp(min=1.0)
    import types as _ty
    def _symbols_ste(self, emb, ids=None):     # forward = the exact hand clamp; backward = straight-through (the clamp is
        if not torch.is_grad_enabled():         # saturated at 0/1 on every token -> its true gradient is exactly 0)
            y = torch.clamp(F.linear(emb.float(), self.reader_w, self.reader_b), 0, 1)
            return torch.round(y) if a.snap else y
        Wr = RD0[0] * (1 + RDD[0]); br = RD0[1] + RDD[1] * RBS
        x = F.linear(emb.float(), Wr, br); y = torch.clamp(x, 0, 1)
        if a.snap: y = torch.round(y)
        return y.detach() + (x - x.detach())
    imp.M.symbols = _ty.MethodType(_symbols_ste, imp.M)
    if TRAIN_CORE:                                  # arithmetic core: every existing nonzero weight + every bias
        import types
        def _lin_grad(self, l, x):
            if torch.is_grad_enabled() and hasattr(self, '_dW'):     # training pass: sparse matrix rebuilt from the deltas
                wv = self._W0[l] * (1 + self._dW[l]); b = self._B0[l] + self._dB[l] * self._BS[l]
                flat = x.reshape(-1, x.shape[-1])
                return (SpLin.apply(wv, flat, getattr(self, f'wi{l}'), getattr(self, f'shape{l}')) + b).reshape(*x.shape[:-1], -1)
            return type(self).lin(self, l, x)                     # rollout / eval: cached CSR (values refreshed in place)
        CORE, CORE0, CORED, CORE_NAMES = [], [], [], []
        for nm in ('stack1', 'stack2'):
            net = getattr(imp.M, nm); net.lin = types.MethodType(_lin_grad, net)
            net._W0, net._B0, net._BS, net._dW, net._dB = {}, {}, {}, {}, {}
            for l in range(net.L + 1):
                w_, b_ = getattr(net, f'wv{l}'), getattr(net, f'b{l}')
                net._W0[l] = w_.detach().clone(); net._B0[l] = b_.detach().clone(); net._BS[l] = b_.detach().abs().clamp(min=1.0)
                net._dW[l] = torch.zeros_like(w_, requires_grad=True); net._dB[l] = torch.zeros_like(b_, requires_grad=True)
                for kind, t, t0, d_ in (('wv', w_, net._W0[l], net._dW[l]), ('b', b_, net._B0[l], net._dB[l])):
                    CORE.append(t); CORE0.append(t0); CORED.append(d_); CORE_NAMES.append((nm, l, kind))
        if a.snap:      # discrete interfaces: stack-1 state rounded to integers, stack-2 output snapped to its one-hot (STE)
            def _snap1(self, x, return_wires=False, chunk=4096):
                y = type(self).forward(self, x, return_wires, chunk)
                if return_wires: return y
                h = torch.round(y)
                return h.detach() + (y - y.detach()) if torch.is_grad_enabled() else h
            def _snap2(self, x, return_wires=False, chunk=4096):
                y = type(self).forward(self, x, return_wires, chunk)
                if return_wires: return y
                m_, i_ = y.max(-1, keepdim=True)
                h = torch.zeros_like(y).scatter_(-1, i_, 1.0) * (m_ > 0.5).to(y.dtype)
                return h.detach() + (y - y.detach()) if torch.is_grad_enabled() else h
            imp.M.stack1.forward = types.MethodType(_snap1, imp.M.stack1)
            imp.M.stack2.forward = types.MethodType(_snap2, imp.M.stack2)
            print('SNAP: reader round, stack-1 state round, stack-2 one-hot (STE)', flush=True)
        def core_refresh():
            with torch.no_grad():
                for nm in ('stack1', 'stack2'):
                    net = getattr(imp.M, nm)
                    for (l, dv), csr in net._csr.items():
                        csr.values().copy_(getattr(net, f'wv{l}').detach())
    def materialize():
        with torch.no_grad():
            RD[0].copy_(RD0[0] * (1 + RDD[0])); RD[1].copy_(RD0[1] + RDD[1] * RBS)
            if TRAIN_CORE:
                for (nm, l, kind), t, t0, d_ in zip(CORE_NAMES, CORE, CORE0, CORED):
                    net = getattr(imp.M, nm)
                    t.copy_(t0 * (1 + d_) if kind == 'wv' else t0 + d_ * net._BS[l])
                core_refresh()
        if TRAIN_CORE: print('core trainable:', len(CORE), 'tensors,', sum(t.numel() for t in CORE), 'values', flush=True) if False else None
    if TRAIN_CORE: print('core trainable:', len(CORE), 'tensors,', sum(t.numel() for t in CORE), 'values (multiplicative deltas)', flush=True)
    print('reader trainable:', [tuple(t.shape) for t in RD], [str(t.dtype) for t in RD],
          'nnz', int((RD0[0] != 0).sum()), 'rms(nz)', round(float(RD0[0][RD0[0] != 0].pow(2).mean().sqrt()), 3), flush=True)

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


class HAdapter(nn.Module):
    """host-only residual adapter (control for wires-only RL): g(h) * B relu(A LN(h) + c); B = 0, w_g = 0 at init."""
    init = 'ha'; src = 'host'
    def __init__(self, d, width):
        super().__init__()
        self.A = nn.Parameter(torch.randn(width, d, device=dev) / math.sqrt(d))
        self.c = nn.Parameter(torch.full((width,), 0.1, device=dev))
        self.B = nn.Parameter(torch.zeros(d, width, device=dev))
        self.wg = nn.Parameter(torch.zeros(d, device=dev)); self.bg = nn.Parameter(torch.zeros((), device=dev))

    def forward(self, feat, h=None):
        ln = _ln(h)
        return torch.sigmoid(ln @ self.wg + self.bg)[..., None] * F.linear(torch.relu(F.linear(ln, self.A) + self.c), self.B)


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
        if init == 'ha':
            per_layer.setdefault(key, []).append(HAdapter(Q.D, a.ha_width)); continue
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


INW = InWire(a.iw_layer, Q.D) if HAS_INWIRE else None
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
def _all_ha():
    for k, v in (links or {}).items():
        for l in (v.ls if isinstance(v, LinkSum) else [v]):
            if isinstance(l, HAdapter): yield k, l


IW_SEP = INW is not None and a.iw_lr is not None
cw_params = [p for _, l in _all_cw() for p in l.parameters()] + (list(INW.parameters()) if (INW is not None and not IW_SEP) else []) + \
            [p for _, l in _all_ha() for p in l.parameters()]
if IW_SEP:                                  # in-wire in its own Adam group (CWfx: lr 1e-3, eps 1e-8 with the floor)
    groups.append({'params': list(INW.parameters()), 'lr': a.iw_lr, 'eps': a.iw_adam_eps if a.iw_adam_eps is not None else a.link_eps})
    print('in-wire group: lr', a.iw_lr, 'eps', groups[-1]['eps'], flush=True)
if cw_params:
    groups.append({'params': cw_params, 'lr': CW_LR, 'eps': a.link_eps})
    print('co-designed wires:', [f'out@{k}' for k, _ in _all_cw()], 'in-wire' if INW is not None else '', 'lr', CW_LR, flush=True)
link_params = link_params + ([p for p in INW.parameters()] if INW is not None else [])
if TRAIN_IMPL:   # relative lr: Dp entries are ~gain (1e4) scale, so an absolute lr would leave the writer frozen
    D_SCALE = max(1.0, float(imp.D.float().pow(2).mean().sqrt()))
    groups.append({'params': [Dp], 'lr': a.impl_lr * D_SCALE, 'eps': a.link_eps})
    print('writer Dp lr', a.impl_lr, 'x rms(D)', round(D_SCALE, 1), flush=True)
if TRAIN_CORE:     # deltas are relative per entry; Adam eps 1e-8 so ~0 gradients make ~0 steps
    if a.core_opt == 'adam': groups.append({'params': CORED, 'lr': a.core_lr, 'eps': a.core_eps})
    print('core lr (relative per entry)', a.core_lr, 'eps', a.core_eps, flush=True)
if TRAIN_READER:
    RLR = a.reader_lr if a.reader_lr is not None else a.core_lr
    if a.core_opt == 'adam': groups.append({'params': RDD, 'lr': RLR, 'eps': a.core_eps})
    print('reader lr (relative per entry)', RLR, flush=True)
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
SGD_LR = {}                     # core_opt sgd: fixed per-part lr, set at the first step from the first gradient
def implant_sgd_step(st):
    """plain SGD on the implant deltas. lr fixed at the first step so that ||w0*dd|| / ||w0|| = target (relative change)."""
    parts = []
    if TRAIN_CORE: parts.append(('core', CORED, CORE0, a.core_lr))
    if TRAIN_READER: parts.append(('reader', RDD, RD0, a.reader_lr if a.reader_lr is not None else a.core_lr))
    with torch.no_grad():
        for name, D_, W0_, target in parts:
            num = sum(float(((t0 if t0.dim() else t0) * (d.grad if d.grad is not None else 0)).pow(2).sum()) for d, t0 in zip(D_, W0_))
            den = sum(float(t0.pow(2).sum()) for t0 in W0_)
            rel = (num / max(den, 1e-30)) ** 0.5                         # relative change per unit lr
            if name not in SGD_LR: SGD_LR[name] = target / max(rel, 1e-30); print(f'SGD lr {name} = {SGD_LR[name]:.3e} (first-step relative change {target})', flush=True)
            for d in D_:
                if d.grad is not None: d -= SGD_LR[name] * d.grad; d.grad = None
            st[f'sgd_rel_{name}'] = SGD_LR[name] * rel                      # this step's relative weight change
base_lrs = [g['lr'] for g in opt.param_groups]
eng = Engine(W, pre, tok, implant=imp, links=links, B_max=a.B_max, S_max=int(os.environ.get('SAN_SMAX', 544)), use_write=USE_WRITE)
if TRAIN_IMPL: eng.Dp = eng_Dp
HAS_EXTRA_LINKS = bool(links) and any(l is not WARM_FINAL for _, l in _all_links()) and (USE_WRITE or WARM_FINAL is not None)
if INW is not None: eng.set_inwire(INW)
if a.arm in ('CWinP', 'CWinO', 'CWinB'):          # same derangement as arm R (seed 12345), applied only to some calls
    g_ = torch.Generator().manual_seed(12345)
    while True:
        p_ = torch.randperm(10, generator=g_)
        if not (p_ == torch.arange(10)).any(): break
    D_der = imp.D.clone(); D_der[:10] = imp.D[p_.to(imp.D.device)]
    if a.arm == 'CWinB':                  # tool breaks mid-training: threshold 0 = every call deranged, 999 = none
        eng.set_partial(999 if a.break_dir == 'on2off' else 0, D_der)
        print('CWinB:', a.break_dir, 'switch after update', a.break_at, flush=True)
    elif a.arm == 'CWinP':
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
    Kt = (imp.K + imp.W) * (4 if imp.tagf else 1)
    _cat = torch.cat([eng.prefix_ids[-Kt:][None].expand(len(seqs), -1), ids], 1)
    e = (imp.emit_ids(_cat) if imp.tagf else imp.full(W['E'][_cat]))[:, Kt:]
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


if WARM_FINAL is not None and WARM_FINAL.log_gain is not None and a.soft_gain > 0:
    WARM_FINAL.log_gain.data.fill_(math.log(a.soft_gain)); WARM_FINAL.lg0.fill_(math.log(a.soft_gain))
    print('soft writer gain fixed at', a.soft_gain, flush=True)
elif WARM_FINAL is not None and WARM_FINAL.log_gain is not None:     # W2ws: calibrate log-gain to p(written) ~ soft_p
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

if WARM_FINAL is not None and WARM_FINAL.log_gain is not None and imp is not None and getattr(imp, 'tagf', False):
    # the hand write's OE row is the solved end veto at its OWN magnitude (not x gain); the soft link stores D/1e4, so at
    # gain g the veto would be g/1e4 of its designed strength. Keep it at the hand magnitude at init: W0[:, OE] = D[OE]/g0.
    from micromachines.specs import SPECS as _S2
    _oe = _S2['calc_tag']['out'].index('OE'); _g0 = float(torch.exp(WARM_FINAL.lg0))
    with torch.no_grad():
        WARM_FINAL.W[:, _oe] = imp.D.float()[_oe] / _g0; WARM_FINAL.W0[:, _oe] = imp.D.float()[_oe] / _g0
    print(f'soft writer: end-veto column kept at hand magnitude (gain {_g0:.2f})', flush=True)

# ---- parameter budget: trainable wiring vs host and vs the implant's nonzero weights
N_HOST = sum(v.numel() for v in P32.values())
N_WIRE = sum(p.numel() for _, l in (_all_links() if links else []) for p in (l.gate, l.W)) + \
         sum(1 for _, l in (_all_links() if links else []) if l.log_gain is not None) + \
         sum(p.numel() for _, l in _all_cw() for p in l.parameters()) + \
         (sum(p.numel() for p in INW.parameters()) if INW is not None else 0) + (Dp.numel() if TRAIN_IMPL else 0) + \
         sum(p.numel() for _, l in _all_ha() for p in l.parameters())
N_IMPL = 0
if HAS_IMPL:
    N_IMPL = int(imp.M.stack1.nnz + imp.M.stack2.nnz + (imp.M.reader_w != 0).sum() + (imp.M.reader_b != 0).sum() + (imp.D != 0).sum())
PARAMS = dict(wire=N_WIRE, host=N_HOST, implant_nnz=N_IMPL, pct_host=100 * N_WIRE / N_HOST,
              pct_implant=(100 * N_WIRE / N_IMPL) if N_IMPL else None)
print(f'PARAMS trainable wiring {N_WIRE:,} = {PARAMS["pct_host"]:.4f}% of host ({N_HOST:,}) = '
      f'{PARAMS["pct_implant"] if N_IMPL else float("nan"):.2f}% of implant nonzeros ({N_IMPL:,})', flush=True)

pool = C.load_train('A', shot_q=shot_q)
pool = [r for r in pool if len(C.q_ids(tok, r['q'])) + a.max_new <= 544]
devset = C.load_train('B', shot_q=shot_q)
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
        if WARM_FINAL.log_gain is not None:
            sv['lg'] = WARM_FINAL.log_gain.detach().clone(); WARM_FINAL.log_gain.detach().copy_(WARM_FINAL.lg0)
        if TRAIN_READER:                                  # the implant side: reader back to the hand-written one
            sv['rd'] = [t.detach().clone() for t in RD]
            for t, t0 in zip(RD, RD0): t.detach().copy_(t0)
        if TRAIN_CORE:
            sv['core'] = [t.detach().clone() for t in CORE]
            for t, t0 in zip(CORE, CORE0): t.detach().copy_(t0)
            core_refresh()
    return sv
def _cw_restore(sv):
    for (_, l), b in zip(_all_cw(), sv['B']): l.B.detach().copy_(b)
    if sv['iw'] is not None: INW.v.detach().copy_(sv['iw'][0]); INW.b.detach().copy_(sv['iw'][1])
    if sv['wf'] is not None: WARM_FINAL.gate.detach().copy_(sv['wf'][0]); WARM_FINAL.W.detach().copy_(sv['wf'][1])
    if 'lg' in sv: WARM_FINAL.log_gain.detach().copy_(sv['lg'])
    if 'rd' in sv:
        for t, t1 in zip(RD, sv['rd']): t.detach().copy_(t1)
    if 'core' in sv:
        for t, t1 in zip(CORE, sv['core']): t.detach().copy_(t1)
        core_refresh()
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
    if mode == 'reset' and TRAIN_READER: cw_saved = _cw_ctx(False, False, False)   # CWall: reset = EVERYTHING hand-made
    m21 = None
    if mode == 'cw_mask21':                  # only the L21 out-wire restricted to firing tokens (graphs re-captured)
        m21 = [l for k, l in _all_cw() if k == 21][0]; m21.mflag.fill_(1.0)
    if mode == 'hostonly': eng.imp_on = False
    elif mode == 'linksoff' and links is not None: saved = _zero_gates_ctx()
    qids = [C.q_ids(tok, r['q']) for r in items]
    out = eng.generate(qids, G=1, greedy=True, max_new=a.max_new)
    sc = [C.score(tok, t, r['gold']) for t, r in zip(out['tokens'], items)]
    acc = lambda kind: 100 * sum(s[0] for s, r in zip(sc, items) if r['kind'] == kind) / max(1, sum(r['kind'] == kind for r in items))
    res = {f'{tag}_acc': 100 * sum(s[0] for s in sc) / len(sc), f'{tag}_gsm': acc('gsm'), f'{tag}_scaled': acc('scaled'),
           f'{tag}_fmt': 100 * sum(s[1] for s in sc) / len(sc),
           f'{tag}_writes': sum(sum(w) for w in out['wflag']) / len(sc), f'{tag}_calls': sum(calls_of(w) for w in out['wflag']) / len(sc),
           f'{tag}_len': sum(map(len, out['tokens'])) / len(sc)}
    for oc in ('muldiv', 'addsub'):                  # accuracy by the problem's dominant arithmetic (gold solution)
        ii = [i for i, r in enumerate(items) if r.get('opclass') == oc]
        if ii:
            res[f'{tag}_acc_{oc}'] = 100 * sum(sc[i][0] for i in ii) / len(ii); res[f'{tag}_n_{oc}'] = len(ii)
            for kd in ('gsm', 'scaled'):
                jj = [i for i in ii if items[i]['kind'] == kd]
                if jj: res[f'{tag}_acc_{oc}_{kd}'] = 100 * sum(sc[i][0] for i in jj) / len(jj)
    eng.imp_on = True
    if saved is not None: _restore_gates(saved)
    if rs_saved is not None: _reset_restore(rs_saved)
    if cw_saved is not None: _cw_restore(cw_saved)
    if m21 is not None: m21.mflag.fill_(float(a.cw_mask))
    import hashlib
    res[f'{tag}_hash'] = hashlib.md5(str(out['tokens']).encode()).hexdigest()[:12]      # bit-identity checks
    if INW is not None and mode != 'hostonly':
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


def save(tag):
    d = f'{a.run_dir}/{tag}'; os.makedirs(d, exist_ok=True)
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in P32.items()}
    sd['lm_head.weight'] = sd['model.embed_tokens.weight'].clone()
    save_file(sd, f'{d}/model.safetensors')
    if links: torch.save({str(k): lk.state_dict() for k, lk in links.items()}, f'{d}/links.pt')
    if INW is not None: torch.save(INW.state_dict(), f'{d}/inwire.pt')
    if TRAIN_IMPL: torch.save(Dp.detach().cpu(), f'{d}/Dp.pt')
    if TRAIN_READER: torch.save([t.detach().cpu() for t in RD], f'{d}/reader.pt')
    if TRAIN_CORE: torch.save([t.detach().cpu() for t in CORE], f'{d}/core.pt')
    open(f'{d}/DONE', 'w').close()


PACK_SIZES = [n for n in (2048, 4096, 8192, 16384, 32768) if n <= a.tok_budget] or [a.tok_budget]
packers = {n: Q.Packer(n, Lp, dev) for n in PACK_SIZES}


def lp_of_hidden(Wt, hn, tgt):
    E = Wt['E']
    def chunk_lp(hc, tc):
        lg = F.linear(hc, E).float() / a.T
        return torch.log_softmax(lg, -1).gather(-1, tc[:, None])[:, 0]
    return torch.cat([checkpoint(chunk_lp, hn[c:c + 4096], tgt[c:c + 4096], use_reentrant=False) for c in range(0, hn.shape[0], 4096)])


OE_IDX = -1
if imp is not None:
    from micromachines.specs import SPECS as _S3
    OE_IDX = list(_S3[imp.M.name]['out']).index('OE') if 'OE' in _S3[imp.M.name]['out'] else -1


def train_step(batch_q, batch_r, old_lp, adv, n_total_rows, batch_dec=None, batch_on=None):
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
    Kt = ((imp.K + imp.W) * (4 if imp.tagf else 1)) if imp is not None else 0
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
        feat = fwrite = None
        if imp is not None:
            with torch.no_grad():                               # e is the frozen hard one-hot (no grad through stacks)
                Tm = max(lens)
                full = torch.tensor([x + [eng.eos] * (Tm - len(x)) for x in seqs], device=dev)
                full = torch.cat([eng.prefix_ids[-Kt:][None].expand(len(seqs), -1), full], 1)
                eb = (imp.emit_ids(full) if imp.tagf else imp.full(W['E'][full]))[:, Kt:]
                es = [eb[j, :n] for j, n in enumerate(lens)]
                if batch_on is not None:                         # tool dropout: this row ran without the calculator
                    es = [x_ * float(batch_on[i]) for x_, i in zip(es, mb)]
                e = torch.cat(es + [torch.zeros(Ltot - sum(lens), imp.n_out, device=dev)])[None]
                dpos, didx, dsgn = [], [], []
                if INW is not None and batch_dec is not None:    # replay the rollout's in-wire decisions
                    keep = torch.ones(Ltot, dtype=torch.bool); brkm = torch.zeros(Ltot, dtype=torch.bool)
                    nzall = (e[0].abs().sum(-1) > 0).cpu()
                    o2 = si = 0
                    for j, i in enumerate(mb):
                        ql, rl = len(batch_q[i]), len(batch_r[i]); dl = batch_dec[i]
                        sup = bk = False
                        for t in range(rl):
                            pp = o2 + ql - 1 + t; d_ = dl[t] if t < len(dl) else 0; b_ = d_ & 3
                            if b_ == 2: sup = True
                            elif b_ == 1: sup = False
                            if b_ in (1, 2): bk = bool(d_ & 4)
                            if not nzall[pp]: sup = bk = False
                            if sup: keep[pp] = False
                            if bk and not sup: brkm[pp] = True
                            if b_ in (1, 2): dpos.append(pp); didx.append(si + t); dsgn.append(1.0 if b_ == 1 else -1.0)
                        o2 += lens[j]; si += rl
                    e = e * keep.to(dev)[None, :, None].to(e.dtype)
            if TRAIN_READER:                                    # same e, now with the exact gradient to the reader
                eg = (imp.emit_ids(full) if imp.tagf else imp.M.emit(W['E'][full])[0])[:, Kt:]
                eg = torch.cat([eg[j, :n] for j, n in enumerate(lens)] + [torch.zeros(Ltot - sum(lens), imp.n_out, device=dev)])[None]
                if INW is not None and batch_dec is not None: eg = eg * keep.to(dev)[None, :, None].to(eg.dtype)
                st['reader_eq'] = st.get('reader_eq', 0.0) + float((eg.detach() - e).abs().max())
                e = eg
            feat = e
            if USE_WRITE:
                fwrite = e @ Dp if TRAIN_IMPL else e @ imp.D     # Dp carries grad (writer); e frozen
                if (eng.partial is not None or eng.partial_ops is not None) and INW is not None and batch_dec is not None:
                    fwrite = fwrite + brkm.to(dev)[None, :, None].float() * (e @ eng.D_der_minus_D)
        hcap = None
        if INW is not None:
            hn, hcap = Q.fwd_packed(Wt, ids, packer.pos, bm, eng.cos_t, eng.sin_t, leaves, links=links, feat=feat,
                                    final_write=fwrite, capture=INW.L)
        elif a.practice_w > 0 and fwrite is not None:
            hn, hpre = Q.fwd_packed(Wt, ids, packer.pos, bm, eng.cos_t, eng.sin_t, leaves, links=links, feat=feat, final_write=fwrite, pre_write=True)
        else:
            hn = Q.fwd_packed(Wt, ids, packer.pos, bm, eng.cos_t, eng.sin_t, leaves, links=links, feat=feat, final_write=fwrite)
        lp = lp_of_hidden(Wt, hn[sel], tgt)
        aux = None
        if a.practice_w > 0 and fwrite is not None:      # practice: host-only logits must predict the calculator-written tokens
            ef_ = feat[0, sel]; wr = (ef_.abs().sum(-1) > 0) & (ef_.argmax(-1) != OE_IDX)
            if wr.any():
                ce = -lp_of_hidden(Wt, hpre[sel][wr], tgt[wr])
                aux = a.practice_w * ce.sum() / (n_total_rows * 10.0)
                st['practice_ce'] = st.get('practice_ce', 0.0) + float(ce.detach().sum()); st['practice_n'] = st.get('practice_n', 0) + int(wr.sum())
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
            logit = INW.k * 0.5 + INW.b + _ln(hcap[dp_ - 1]) @ INW.v
            if INW.eps == 0.0:
                lp = lp.index_add(0, di_, F.logsigmoid(ds_ * logit))
            else:                                                             # same floor as the rollout
                pf = INW.eps + (1 - 2 * INW.eps) * torch.sigmoid(logit)
                lp = lp.index_add(0, di_, torch.where(ds_ > 0, torch.log(pf), torch.log1p(-pf)))
            st['n_dec'] = st.get('n_dec', 0) + len(dpos); st['n_veto'] = st.get('n_veto', 0) + int((ds_ < 0).sum())
        if feat is not None:
            enz = (feat[0, sel].abs().sum(-1) > 0)                    # implant fires at this predictor position
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
        if aux is not None: loss = loss + aux
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
        st['iw_v_norm'] = float(INW.v.detach().norm()); st['iw_b'] = float(INW.b.detach())
    if links:                                                        # pre-clip grad norms + displacement from init
        gg, gw, dg, dw = {}, {}, {}, {}
        for k, l in _all_links():
            n = f'{l.src}@{k}'
            while n in gg: n += "'"
            gg[n] = float(l.gate.grad.norm()) if l.gate.grad is not None else -1.0
            gw[n] = float(l.W.grad.norm()) if l.W.grad is not None else -1.0
        st['link_grad_gate'], st['link_grad_W'] = gg, gw
    if TRAIN_CORE:
        gs = {}
        for (nm, l, kind), t in zip(CORE_NAMES, CORED):
            g_ = t.grad; k_ = f'{nm}:{kind}'
            acc = gs.setdefault(k_, [0.0, 0, 0]); acc[2] += t.numel()
            if g_ is not None: acc[0] += float(g_.pow(2).sum()); acc[1] += int((g_ != 0).sum())
        st['core_grad'] = {k: dict(norm=v[0] ** 0.5, frac_nz=v[1] / max(1, v[2])) for k, v in gs.items()}
    if TRAIN_READER:
        st['reader_grad'] = [float(t.grad.norm()) if t.grad is not None else 0.0 for t in RDD]
        st['reader_grad_nnz'] = int((RDD[0].grad != 0).sum()) if RDD[0].grad is not None else 0
    if a.grad_check:                                   # report every group's gradient, no step
        gg = {}
        if not a.freeze_host: gg['host'] = float(sum(P32[k].grad.pow(2).sum() for k in names if P32[k].grad is not None) ** 0.5)
        if WARM_FINAL is not None:
            gg['writer_W'] = float(WARM_FINAL.W.grad.norm()) if WARM_FINAL.W.grad is not None else 0.0
            gg['writer_loggain'] = float(WARM_FINAL.log_gain.grad.abs()) if (WARM_FINAL.log_gain is not None and WARM_FINAL.log_gain.grad is not None) else 0.0
        for k_, w_ in _all_cw():
            gg[f'out@{k_}'] = float(sum(p.grad.pow(2).sum() for p in w_.parameters() if p.grad is not None) ** 0.5)
        if INW is not None: gg['inwire'] = float(sum(p.grad.pow(2).sum() for p in INW.parameters() if p.grad is not None) ** 0.5)
        st['group_grad'] = gg
        return st
    tp = ([] if a.freeze_host else [P32[k] for k in names]) + link_params + ([Dp] if TRAIN_IMPL else []) + (RDD if (TRAIN_READER and a.core_opt == 'adam') else []) + (CORED if (TRAIN_CORE and a.core_opt == 'adam') else [])
    gn = torch.nn.utils.clip_grad_norm_(tp, 1.0)
    if a.core_opt == 'sgd' and TRAIN_READER: implant_sgd_step(st)
    opt.step(); opt.zero_grad(set_to_none=True)
    if TRAIN_READER: materialize()
    with torch.no_grad():   # decoupled L1 on out-wire gates (plain SGD on b_g: lr * lambda * mean g(1-g)), not via Adam
        for _, w_ in _all_cw():
            if getattr(w_, '_l1n', 0):
                w_.bg -= CW_LR * min(1.0, 1.0) * a.cw_l1 * (w_._l1g / w_._l1n); w_._l1g = 0.0; w_._l1n = 0
    st['grad_norm'] = float(gn); st['train_s'] = time.time() - t0
    return st


# -------- module exactness (calc_tag): 10k random '<calc>a op b</calc>' calls, teacher-forced along the exact answer --------
_MX = None
@torch.no_grad()
def module_exact(n=10000, bs=1000):
    """% of calls where the CURRENT reader + core write exactly ' ' + digits, then the end veto, then stay silent on '.'"""
    global _MX
    from micromachines.calcfmt import calc_ref
    from micromachines.specs import SPECS as _S
    from micromachines.machine import first as _fi
    outs = _S['calc_tag']['out']; im = imp.A.all_ids['calc_tag']; otok = [_fi(im[x]) for x in outs]
    OS_, OE_ = outs.index('OS'), outs.index('OE')
    if _MX is None:
        r = random.Random(77); T_ = lambda x: tok(x, add_special_tokens=False)['input_ids']; rows = []
        while len(rows) < n:
            op = r.choice('+-*/'); a_ = str(r.randint(0, 10 ** r.randint(1, 9))); b_ = str(r.randint(1, 10 ** r.randint(1, 6 if op in '*/' else 9)))
            if r.random() < 0.15: a_ += '.' + str(r.randint(1, 99))
            res = calc_ref(f'{a_}{op}{b_}')
            if not res: continue
            pre_ = T_(f'So we get <calc>{a_} {op} {b_}</calc>'); post = T_('.')
            ans = [otok[OS_]] + [otok[outs.index(ch)] for ch in res]    # exactly the implant's own tokens: ' ', then one per char
            want = ans
            rows.append((pre_ + ans + post, len(pre_) - 1, len(want), op))
        _MX = rows
    pad = tok(' and', add_special_tokens=False)['input_ids'][0]
    ok = {o: [0, 0] for o in '+-*/'}
    for i in range(0, n, bs):
        ch = _MX[i:i + bs]; L = max(len(x[0]) for x in ch) + imp.K + imp.W
        X = torch.tensor([[pad] * (L - len(x[0])) + x[0] for x in ch], device=dev)
        e = imp.emit_ids(X); fire = e.abs().sum(-1) > 0; am = e.argmax(-1)
        for j, (ids_, p0, nw, op) in enumerate(ch):
            off = L - len(ids_); p = off + p0; good = True
            for k in range(nw):                                   # positions CE .. last-but-one written token
                if not bool(fire[j, p + k]) or otok[int(am[j, p + k])] != ids_[p0 + 1 + k] or int(am[j, p + k]) == OE_: good = False; break
            if good:
                good = bool(fire[j, p + nw]) and int(am[j, p + nw]) == OE_ and (not bool(fire[j, p + nw + 1]) or int(am[j, p + nw + 1]) == OE_)   # veto, then no digit on the host's '.'
            ok[op][0] += int(good); ok[op][1] += 1
    tot = sum(v[0] for v in ok.values()) / max(1, sum(v[1] for v in ok.values()))
    return dict(exact=100 * tot, **{f'exact_{o}': 100 * v[0] / max(1, v[1]) for o, v in ok.items()})


# ------------------------------------------------------------------ main loop
DEV_EXTRA = {int(x) for x in a.dev_extra.split(',') if x}
t_start = time.time()
if a.eval_ckpt:            # one-shot test evaluation of a saved checkpoint (wiring as trained / hand-only / implant off)
    from safetensors.torch import load_file
    if a.host_ckpt != 'base':                 # transplant: host from --host_ckpt (or the wiring checkpoint itself)
        sd = load_file(f"{a.host_ckpt or a.eval_ckpt}/model.safetensors", device="cpu")
        with torch.no_grad():
            for k in names: P32[k].copy_(sd[k].float())
    if links and os.path.exists(f'{a.eval_ckpt}/links.pt'):
        ls = torch.load(f'{a.eval_ckpt}/links.pt', map_location=dev)
        for k, lk in links.items(): lk.load_state_dict(ls[str(k)], strict=False)
    if INW is not None and os.path.exists(f'{a.eval_ckpt}/inwire.pt'):
        INW.load_state_dict(torch.load(f'{a.eval_ckpt}/inwire.pt', map_location=dev))
    if TRAIN_IMPL and os.path.exists(f'{a.eval_ckpt}/Dp.pt'):
        Dp.data.copy_(torch.load(f'{a.eval_ckpt}/Dp.pt').to(dev))
    if TRAIN_READER and os.path.exists(f'{a.eval_ckpt}/reader.pt'):
        with torch.no_grad():
            for t, t1 in zip(RD, torch.load(f'{a.eval_ckpt}/reader.pt')): t.copy_(t1.to(dev))
        if TRAIN_CORE and os.path.exists(f'{a.eval_ckpt}/core.pt'):
            for t, t1 in zip(CORE, torch.load(f'{a.eval_ckpt}/core.pt')): t.copy_(t1.to(dev))
            core_refresh()
        print('reader loaded: drift', float((RD[0].detach() - RD0[0]).norm() / RD0[0].norm()), flush=True)
    sync_weights()
    if os.environ.get("DIG_HOOK"): exec(open(os.environ["DIG_HOOK"]).read()); sys.exit(0)
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
    out_res['host'] = a.host_ckpt or a.eval_ckpt
    for name in a.test_sets.split(','):
        items = devset if name == 'dev' else C.load_test(name, shot_q=shot_q)
        n0 = len(items); items = [r if name == 'dev' else dict(r, kind='gsm') for r in items if len(C.q_ids(tok, r['q'])) + a.max_new <= 544]
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
    save('sft_final')
    log({'DONE_RUN': time.time() - t_start}); print('DONE', flush=True); sys.exit(0)
if a.grad_check:      # one CE backward on call-form gold traces: which parameter groups get a real gradient at u0?
    rows_ = [json.loads(l) for l in open(a.calib_traces)][:a.prompts]
    a.sft = 1
    st = train_step([C.q_ids(tok, r['q']) for r in rows_], [r['resp'] for r in rows_], None, None, len(rows_))
    print('GRADCHECK', json.dumps({k: st.get(k) for k in ('group_grad', 'core_grad', 'reader_grad', 'reader_grad_nnz', 'reader_eq', 'loss')}, indent=1), flush=True)
    if imp is not None and imp.tagf: print('MODULE_EXACT u0', module_exact(), flush=True)
    sys.exit(0)
def resume_save(u):
    os.makedirs(os.path.dirname(RESUME_F), exist_ok=True)
    st = dict(u=u, host={k: P32[k].detach() for k in names}, opt=opt.state_dict(), sgd_lr=dict(SGD_LR),
              links={str(k): lk.state_dict() for k, lk in links.items()} if links else None,
              inw=INW.state_dict() if INW is not None else None, Dp=Dp.detach() if TRAIN_IMPL else None,
              rdd=[t.detach() for t in RDD] if RDD is not None else None, cored=[t.detach() for t in CORED] if CORED is not None else None,
              rng=dict(py=random.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    torch.save(st, RESUME_F + '.tmp'); os.replace(RESUME_F + '.tmp', RESUME_F)
    print(f'resume checkpoint @u{u} -> {RESUME_F}', flush=True)


START_U = 1
if a.resume and os.path.exists(RESUME_F):
    st = torch.load(RESUME_F, map_location=dev, weights_only=False)
    with torch.no_grad():
        for k in names: P32[k].copy_(st['host'][k])
        if st['links'] is not None:
            for k, lk in links.items(): lk.load_state_dict(st['links'][str(k)])
        if st['inw'] is not None: INW.load_state_dict(st['inw'])
        if st['Dp'] is not None: Dp.copy_(st['Dp'])
        if st['rdd'] is not None:
            for t, t1 in zip(RDD, st['rdd']): t.copy_(t1)
        if st['cored'] is not None:
            for t, t1 in zip(CORED, st['cored']): t.copy_(t1)
    opt.load_state_dict(st['opt']); SGD_LR.update(st.get('sgd_lr', {}))
    random.setstate(st['rng']['py']); torch.set_rng_state(st['rng']['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in st['rng']['cuda']])
    START_U = st['u'] + 1; del st
    sync_weights()
    if TRAIN_READER: materialize()
    log({'resumed_from': START_U - 1}); print('RESUMED from u', START_U - 1, flush=True)
if START_U == 1:
    log({'params': PARAMS})
if START_U == 1 and imp is not None and imp.tagf and (TRAIN_READER or TRAIN_CORE):
    mx = module_exact(); log({'module_exact': {**mx, 'u': 0}}); print('module exact@0', mx, flush=True)
if START_U == 1:
    res0 = evaluate(devset); res0.update(u=0, t=0); log({'dev': res0}); print('dev@0', res0, flush=True)
if START_U == 1 and imp is not None:
    r = evaluate(devset, mode='hostonly', tag='devhost'); r.update(u=0); log({'dev_host_curve': r}); print('host-only@0', r, flush=True)
    if HAS_EXTRA_LINKS:
        r = evaluate(devset, mode='linksoff', tag='devlinksoff'); r.update(u=0); log({'dev_linksoff_curve': r})
TRACK_DRIFT = (TRAIN_IMPL or WARM_FINAL is not None) and digit_rows is not None   # no digit map for deranged implants
if TRACK_DRIFT and START_U == 1: d0 = writer_drift(); log({'drift': {**d0, 'u': 0}}); print('drift@0', d0, flush=True)
for u in range(START_U, a.updates + 1):
    for g, b in zip(opt.param_groups, base_lrs): g['lr'] = b * min(1.0, u / a.warmup)
    if a.arm == 'CWinB' and u == a.break_at + 1:      # flip the calculator in place (no graph re-capture)
        eng.partial_t.fill_(0 if a.break_dir == 'on2off' else 999); log({'break_switch': u}); print('BREAK SWITCH at', u, flush=True)
    batch = random.sample(pool, a.prompts)
    qids = [C.q_ids(tok, r['q']) for r in batch]
    t0 = time.time()
    eng.drop_p = a.tool_dropout
    out = eng.generate(qids, G=a.G, T=a.T, max_new=a.max_new)
    eng.drop_p = 0.0
    t_roll = time.time() - t0
    R = []; fmt = 0
    for j, t in enumerate(out['tokens']):
        s = C.score(tok, t, batch[j // a.G]['gold']); R.append(s[0]); fmt += s[1]
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
    if idx:
        st = train_step([qids[j // a.G] for j in idx], [out['tokens'][j] for j in idx], [out['logp'][j] for j in idx],
                        [float(advt.view(-1)[j]) for j in idx], len(R),
                        batch_dec=[out['dec'][j] for j in idx] if INW is not None else None,
                        batch_on=[out['on'][j] for j in idx] if a.tool_dropout > 0 else None)
        rec.update(st)
        sync_weights()
    if INW is not None:                               # rollout fire rate by the call's largest operand (tracker)
        bk_, bo_ = {}, {}
        for d, o in zip(out['dec'], out['ops']):
            for x, n_ in zip(d, o):
                if x <= 0: continue
                nd_, op_ = n_ & 15, n_ >> 4
                b_ = '0' if nd_ <= 0 else ('1-2' if nd_ <= 2 else ('3-4' if nd_ <= 4 else ('5-6' if nd_ <= 6 else '7+')))
                c_ = bk_.setdefault(b_, [0, 0]); c_[0] += 1; c_[1] += int((x & 3) == 1)
                c_ = bo_.setdefault('?+-*/'[op_] if op_ < 5 else '?', [0, 0]); c_[0] += 1; c_[1] += int((x & 3) == 1)
        rec['iw_bucket'] = bk_; rec['iw_op'] = bo_; rec['reward_raw'] = sum(R_raw) / len(R_raw)
    if INW is not None:                               # rollout in-wire: fire rate at hand-trigger burst starts, by kind
        for kd in ('gsm', 'scaled'):
            ds_ = [out['dec'][j] for j in range(len(R)) if kinds[j] == kd]
            ns = sum(sum(1 for x in d if x > 0) for d in ds_); nf = sum(sum(1 for x in d if (x & 3) == 1) for d in ds_)
            rec[f'iw_fire_{kd}'] = nf / max(1, ns); rec[f'iw_starts_{kd}'] = ns / max(1, len(ds_))
    if links:
        rec['link_gate_l2'], rec['link_W_l2'], rec['link_eff'] = link_norms()
    if WARM_FINAL is not None and WARM_FINAL.log_gain is not None: rec['gain'] = float(torch.exp(WARM_FINAL.log_gain.detach()))
    if TRAIN_IMPL:
        rec['Dp_delta'] = float((Dp.detach() - imp.D.float()).norm() / imp.D.float().norm())
    if a.tool_dropout > 0:
        rec['tool_on_frac'] = sum(out['on']) / len(out['on'])
        onr = [R_raw[j] for j in range(len(R_raw)) if out['on'][j] > 0]; offr = [R_raw[j] for j in range(len(R_raw)) if out['on'][j] == 0]
        rec['reward_on'] = sum(onr) / max(1, len(onr)); rec['reward_off'] = sum(offr) / max(1, len(offr))
    if 'practice_n' in rec: rec['practice_ce'] = rec['practice_ce'] / max(1, rec['practice_n'])
    if TRAIN_CORE:
        rec['core_drift'] = {nm: float(sum((t.detach() - t0).pow(2).sum() for (n_, _, _), t, t0 in zip(CORE_NAMES, CORE, CORE0) if n_ == nm) ** 0.5 /
                                        sum(t0.pow(2).sum() for (n_, _, _), t0 in zip(CORE_NAMES, CORE0) if n_ == nm) ** 0.5) for nm in ('stack1', 'stack2')}
    if TRAIN_READER:
        rec['reader_drift'] = [float((t.detach() - t0).norm() / t0.norm().clamp(min=1e-12)) for t, t0 in zip(RD, RD0)]
    if WARM_FINAL is not None:
        rec['writer_dir_drift'] = float((WARM_FINAL.W.detach() - WARM_FINAL.W0).norm() / WARM_FINAL.W0.norm())
    log(rec)
    if u % 5 == 0 or u == 1:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                          if k not in ('link_W_l2',)}), flush=True)
    if imp is not None and imp.tagf and (TRAIN_READER or TRAIN_CORE) and (u % a.exact_every == 0 or (u <= 25 and u % 5 == 0) or u == a.updates):
        mx = module_exact(); log({'module_exact': {**mx, 'u': u}}); print('module exact', u, mx, flush=True)
    if u % a.dev_every == 0 or u == a.updates or u in DEV_EXTRA:
        r = evaluate(devset); r.update(u=u, t=time.time() - t_start); log({'dev': r}); print('dev', r, flush=True)
        if CW_ARM and (u % 100 == 0 or u == a.updates):
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
    if u % a.ckpt_every == 0 or u == a.updates:
        save(f'step_{u:06d}')
    if a.resume_every and u % a.resume_every == 0 and u < a.updates:
        resume_save(u)
# end-of-run ablations
if imp is not None:
    r = evaluate(devset, mode='hostonly', tag='devhostonly'); r.update(u=a.updates); log({'dev_hostonly': r}); print('host-only', r, flush=True)
    if HAS_EXTRA_LINKS:
        r = evaluate(devset, mode='linksoff', tag='devlinksoff_final'); r.update(u=a.updates); log({'dev_linksoff_final': r}); print('links-off', r, flush=True)
if TRACK_DRIFT:
    df = writer_drift('drift_final'); log({'drift_final': {**df, 'u': a.updates}}); print('drift-final', {k: df[k] for k in df if 'digits' not in k}, flush=True)
log({'DONE_RUN': time.time() - t_start}); print('DONE', time.time() - t_start, flush=True)
if os.path.exists(RESUME_F): os.remove(RESUME_F)
