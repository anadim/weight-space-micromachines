"""Two micromachines driven as ONE implant in the RL engine: calc_s4 (frozen, unchanged) + chain2 (calc v2, compiled).
Their one-hot outputs are concatenated and their writes added (e @ D, D = both machines' write directions stacked),
as machine.Attached adds the writes of several machines,
with one fixed gate: where chain2 writes, calc_s4 is masked (calc_s4 misfires on some decimal chains, test_chain2.py). Same API as implant.CalcImplant."""
import torch


def _gate(e1, e2):
    """chain2 has priority: where it writes, calc_s4's output is masked (a fixed AND on the two one-hot outputs)."""
    on2 = (e2.abs().sum(-1, keepdim=True) > 0).to(e1.dtype)
    return torch.cat([e1 * (1 - on2), e2], -1)


class _M:
    def __init__(self, parts): self.parts = parts

    def emit(self, emb):
        return _gate(self.parts[0].M.emit(emb)[0], self.parts[1].M.emit(emb)[0]), None


class DualImplant:
    def __init__(self, calc, chain):
        self.parts = [calc, chain]; self.calc, self.chain = calc, chain
        self.tagf = False; self.fk = None; self.random = None; self.ring = False; self.last_w = None
        self.K = max(p.K for p in self.parts); self.W = max(p.W for p in self.parts)
        self.D = torch.cat([p.D for p in self.parts], 0)
        self.n_out = self.D.shape[0]; self.n_calc = calc.n_out
        self.M = _M(self.parts); self.margins = calc.margins
        self.sym_h, self.S_h, self.key_h, self.fst = calc.sym_h, calc.S_h, calc.key_h, calc.fst

    def states(self):
        return [t for p in self.parts for t in (p.sym_h, p.S_h, p.key_h, p.fst)]

    @torch.no_grad()
    def prime_rows(self, emb_tail, rows):
        return _gate(self.parts[0].prime_rows(emb_tail, rows), self.parts[1].prime_rows(emb_tail, rows))

    def step(self, emb, B, tok=None):
        return _gate(self.parts[0].step(emb, B, tok), self.parts[1].step(emb, B, tok))

    def gather(self, idx, n):
        for p in self.parts: p.gather(idx, n)

    @torch.no_grad()
    def full(self, emb):
        return self.M.emit(emb)[0]
