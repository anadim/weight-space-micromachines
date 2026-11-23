"""calc_tag checks (2026-11-19): (1) exactness of the DECODE path on random '<calc>a op b</calc>' calls (implant writes
' ' + digits, then the end veto; the host's next token must not restart it); (2) decode path == teacher-forced emit_ids;
(3) silence + exact writes along call-form gold solutions (8-shot prefix + dev questions)."""
import os, sys, json, random, torch
os.environ['CALC_MACHINE'] = 'calc_tag'; os.environ['CALLFORM'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForCausalLM
import common as C
from implant import CalcImplant
from micromachines.calcfmt import calc_ref
from micromachines.specs import SPECS
from micromachines.machine import first
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
tok = AutoTokenizer.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base')); dev = 'cuda'
hf = AutoModelForCausalLM.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'), dtype=torch.bfloat16).to(dev).eval()
E = hf.get_input_embeddings().weight.detach()
B = 1000
imp = CalcImplant(hf, tok, B, machine='calc_tag', fused=False); imp.Eref = E
del hf
outs = SPECS['calc_tag']['out']; ids_map = imp.A.all_ids['calc_tag']
out_tok = [first(ids_map[s]) for s in outs]; OE = outs.index('OE')
print('out symbols', outs, 'tokens', [tok.decode([t]) for t in out_tok], flush=True)
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
rng = random.Random(0)
def rnum(maxd):
    nd = rng.randint(1, maxd); x = str(rng.randint(10 ** (nd - 1) if nd > 1 else 0, 10 ** nd - 1))
    if rng.random() < 0.15: x += '.' + str(rng.randint(0, 999)).rstrip('0') if rng.random() < 1 else ''
    return x.rstrip('.')
cases = []
for i in range(N):
    op = rng.choice('+-*/'); a = rnum(9); b = rnum(9 if op in '+-' else 6)
    if op == '/' and float(b) == 0: b = '7'
    sp = rng.random() < 0.8
    cases.append((a, op, b, f"<calc>{a} {op} {b}</calc>" if sp else f"<calc>{a}{op}{b}</calc>", calc_ref(f'{a}{op}{b}')))
pad = T(' and')[0]
@torch.no_grad()
def run(batch):
    seqs = [T('So we get ' + c[3]) for c in batch]; L = max(map(len, seqs)) + imp.K + imp.W
    seqs = [[pad] * (L - len(s)) + s for s in seqs]
    n = len(seqs); imp.sym_h.zero_(); imp.S_h.zero_(); imp.key_h.zero_(); imp.fst.zero_()
    X = torch.tensor(seqs, device=dev); e = None
    for t in range(L): e = imp.step(E[X[:, t]], n, X[:, t])
    written = [[] for _ in range(n)]; done = torch.zeros(n, dtype=torch.bool, device=dev); after = torch.zeros(n, device=dev)
    host_next = torch.tensor(T('.'), device=dev)[0]
    full = [list(s) for s in seqs]
    for k in range(40):
        fire = e.abs().sum(-1) > 0; am = e.argmax(-1)
        is_oe = fire & (am == OE)
        nt = torch.where(fire & ~is_oe, torch.tensor(out_tok, device=dev)[am], host_next)
        for j in range(n):
            if not bool(done[j]):
                if bool(fire[j]) and not bool(is_oe[j]): written[j].append(int(nt[j]))
                elif bool(is_oe[j]) or not bool(fire[j]): done[j] = True
            else:
                after[j] += float(fire[j] & ~is_oe[j])
        for j in range(n): full[j].append(int(nt[j]))
        e = imp.step(E[nt], n, nt)
    # teacher-forced reference on the same final token streams
    ef = imp.emit_ids(torch.tensor(full, device=dev))
    return written, after, full, ef
ok = tot = bad_after = 0; mism = []; tf_mismatch = 0
for i in range(0, N, B):
    batch = cases[i:i + B]
    written, after, full, ef = run(batch)
    for j, c in enumerate(batch):
        got = tok.decode(written[j]); want = (' ' + c[4]) if c[4] else ''
        tot += 1; ok += int(got == want); bad_after += int(after[j] > 0)
        if got != want and len(mism) < 12: mism.append((c[3], c[4], got))
print(f'DECODE exact {ok}/{tot} = {100*ok/tot:.2f}%   writes after the end veto: {bad_after}', flush=True)
for m in mism: print('  MISMATCH', m)
# (2) decode vs teacher-forced: recompute decode e along the final streams with the step path and compare
@torch.no_grad()
def decode_e(X):
    n, L = X.shape; imp.sym_h.zero_(); imp.S_h.zero_(); imp.key_h.zero_(); imp.fst.zero_(); es = []
    for t in range(L): es.append(imp.step(E[X[:, t]], n, X[:, t]))
    return torch.stack(es, 1)
X = torch.tensor(run(cases[:B])[2], device=dev)
ed = decode_e(X); ef = imp.emit_ids(X)
print('decode == teacher-forced (first', B, 'streams):', torch.equal(ed, ef), 'max diff', float((ed - ef).abs().max()), flush=True)
# (3) gold call-form solutions: writes only at the call outputs, and exactly the text's tokens
pre, shot_q = C.prefix_ids(tok)
tr = [json.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')]
rows = [r for r in tr if C.part(r['question']) == 'B'][:400]
nw = nbad = ncalls = nos = 0; OS = outs.index('OS')
for r in rows:
    body, fin, n = C.callform_solution(r['answer'])
    q = C.q_ids(tok, r['question']); a = T(' ' + body + '\n#### ' + fin)
    ids = torch.tensor([pre + q + a], device=dev)
    e = imp.emit_ids(ids)[0]; full = ids[0].tolist(); Lq = len(pre) + len(q)
    fire = (e.abs().sum(-1) > 0)
    ncalls += n
    for p in range(Lq - 1, len(full) - 1):
        if bool(fire[p]):
            k = int(e[p].argmax())
            if k == OE: continue
            nos += int(k == OS); nw += 1
            if out_tok[k] != full[p + 1]: nbad += 1
print(f'GOLD answers (400 dev, call form): calls in text {ncalls}, OS fires {nos}, implant writes {nw}, writes != text token {nbad}', flush=True)
