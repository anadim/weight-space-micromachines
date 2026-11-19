"""Port v2 (on top of port_ifeval.py output): fixed emitter strings that are ONE token in Qwen but several in the new vocabulary
('***\\n\\n', '******', 'P.S.', 'P.P.S') are written as their token SEQUENCE in the host vocab; emitter prefix/suffix registers
widened 4 -> 6 slots so the sequences fit. Diagnosed from the ODD IFEval outputs (paras 8->2/16: the circuit blocked the host's
'***' but its own '***\\n\\n' token id was -1). Rule logic unchanged.
v2.1: two_responses separator NOT converted (DEV: its once-only latch reads the single token "******" (STAR6) and never sets
when the separator is written as "****"+"**" -> re-fires every end-of-turn, 9->0/34 on DEV, 44 truncations); kept as in v1.  python port_v2.py DIR"""
import sys
d = sys.argv[1]; p = f'{d}/circuit.py'; c = open(p).read()
reps = [
    ("self.pre = zl(B, 4)", "self.pre = zl(B, 6)"),
    ("self.suf = zl(B, 4)", "self.suf = zl(B, 6)"),
    ("a = self.pre.gather(1, j.clamp(0, 3)[:, None])[:, 0]", "a = self.pre.gather(1, j.clamp(0, 5)[:, None])[:, 0]"),
    ("c = self.suf.gather(1, (j - pl - sl).clamp(0, 3)[:, None])[:, 0]", "c = self.suf.gather(1, (j - pl - sl).clamp(0, 5)[:, None])[:, 0]"),
    ("out = torch.full((self.B, 4), -1, dtype=torch.long, device=dev)", "out = torch.full((self.B, 6), -1, dtype=torch.long, device=dev)"),
    ("self.start_prog(s_go, 5, pre=[self.tk['***\\n\\n']])", "self.start_prog(s_go, 5, pre=self.seq('***\\n\\n'))"),
    ("pre=[sep, self.tk['P'], self.tk['.P'], self.tk['.S']], suf=[self.tk['.']])", "pre=[sep, *self.seq('P.P.S')], suf=[self.tk['.']])"),
    ("pre=[sep, self.tk['P'], self.tk['.S'], self.tk['.']])", "pre=[sep, *self.seq('P.S.')])"),
    ("t = self.T['TOK']; self.tk = {k: torch.tensor(v, device=dev) for k, v in t.items()}",
     "t = self.T['TOK']; self.tk = {k: torch.tensor(v, device=dev) for k, v in t.items()}\n"
     "        self.seq = lambda s: [int(i) for i in tok(s, add_special_tokens=False).input_ids]      # v2: fixed string -> host token sequence"),
]
for a, b in reps:
    assert a in c, a; c = c.replace(a, b)
open(p, 'w').write(c); print('v2 applied to', p)
