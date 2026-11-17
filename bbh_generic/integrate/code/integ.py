"""BBH integration step (protocol rules 14/15): ONE host, ALL generic tools of Teams A, B, C attached at once.

Tools (shared envelope <tool_call>NAME ARGS</tool_call><tool_response>RESULT</tool_response>):
  Team A: track (track_gen), order (order_gen)
  Team B: interval (interval_gen), walk (turtle_gen), truth (truth_gen)
  Team C: expr (expr_call), bool (bool_call), calc (calc_call)      -- CALL form (decision: see NOTEBOOK.md)
All eight compiled machines live in ONE micromachines.machine.Attached object (one hook before the host's final norm; the
writes of all machines are added; shared output tokens share one solved write direction). Team C's float64 stack and its
'ANY' start-of-text flag are applied exactly as teamc_tools.attach_gen does.

    from integ import attach_all, prep_call_tokens, TOOLS
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _paths import REPO, G                                   # repo micromachines/ + team code on sys.path
import torch
from micromachines.specs import SPECS
from micromachines import machine as MM
import register as _regA                                    # noqa: F401  (Team A specs)
import circuits_B as _regB                                  # noqa: F401  (Team B specs)
import teamc_tools as TT                                    # Team C specs + helpers
TT.register(oe='dotnl', bool_dot=1)

TOOLS_A = ('track_gen', 'order_gen')
TOOLS_B = ('interval_gen', 'turtle_gen', 'truth_gen')
TOOLS_C_CALL = ('expr_call', 'bool_call', 'calc_call')
TOOLS_C_PASSIVE = ('expr_gen', 'calc_gen', 'bool_gen')
TOOLS = TOOLS_A + TOOLS_B + TOOLS_C_CALL
CALL_NAME = {'track_gen': 'track', 'order_gen': 'order', 'interval_gen': 'interval', 'turtle_gen': 'walk', 'truth_gen': 'truth',
             'expr_call': 'expr', 'bool_call': 'bool', 'calc_call': 'calc'}
prep_call_tokens = TT.prep_call_tokens


def attach_all(model, tok, machines=TOOLS, veto=None):
    A = MM.Attached(model, tok, machines=tuple(machines))
    for n in machines:
        if SPECS[n].get('f64'): A.nets[n].stack1 = TT.F64(A.nets[n].stack1).to(model.device)
        if 'ANY' in SPECS[n]['syms']:
            i = SPECS[n]['syms'].index('ANY'); A.nets[n].reader_w[i] = 0; A.nets[n].reader_b[i] = 1
    if veto is not None and 'calc_gen' in A.nets:
        A.set_veto('calc_gen', torch.load(veto)['v'], zero_os=True)
    return A


def ref(call):
    """Python reference reply for a call 'NAME ARGS' (tests / gold only; never at inference)."""
    import callfmt as CA, tools_ref as RB, callfmt_C as CC
    name = call.split(' ', 1)[0]
    if name in ('track', 'order'): return CA.run_ref(call)
    if name in ('interval', 'walk', 'truth'): return RB.run_ref(call)
    if name in ('expr', 'bool', 'calc'): return CC.run_ref(call)
    return None


class GraphStep:
    """Speed only (no change of the computation): replays Attached.step for a fixed batch as ONE CUDA graph.
    The deep compiled ReLU stacks (Team C's expr/bool: ~1000 layers) are launch-bound (~200 ms per decoding step); a CUDA
    graph removes the launch overhead. Same kernels, same inputs, same order -> the same writes (checked bit-exact against
    the plain Attached.step in check_graph.py). Use: after A.prime(x):  gs = GraphStep(A, B); ... gs.step(next_ids)."""

    def __init__(self, A, B):
        self.A = A; dev = A.model.device
        self.E = A.model.get_input_embeddings()
        self.names = list(A.nets)
        self.st = {n: tuple(t.clone() for t in A.nets[n].state) for n in self.names}      # static state buffers
        self.ids = torch.zeros(B, dtype=torch.long, device=dev)
        saved = {n: tuple(t.clone() for t in A.nets[n].state) for n in self.names}
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2): self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g), torch.no_grad():
            self.out = self._body()
        for n in self.names:                                   # undo the warm-up steps
            for t, v in zip(self.st[n], saved[n]): t.copy_(v)

    def _body(self):
        emb = self.E(self.ids); tot = None
        for n in self.names:
            net = self.A.nets[n]
            net.state = self.st[n]
            w, _ = net.step(emb)
            for t, v in zip(self.st[n], net.state): t.copy_(v)
            tot = w if tot is None else tot + w
        return tot

    @torch.no_grad()
    def step(self, next_ids):
        self.ids.copy_(next_ids); self.g.replay()
        self.A.pending = self.out
