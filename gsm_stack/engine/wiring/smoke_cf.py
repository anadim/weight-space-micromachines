"""Call-form smoke (2026-11-19): greedy answers on N dev items with the calc_tag implant; checks that calls parse, the
implant fires on every well-formed call and only on calls, results are exact, and how the host uses the results."""
import os, sys, re, json, torch
os.environ['CALC_MACHINE'] = 'calc_tag'; os.environ['CALLFORM'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForCausalLM
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from micromachines.calcfmt import calc_ref
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
dev = 'cuda'; tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, shot_q = C.prefix_ids(tok)
W = Q.fuse({k: v for k, v in Q.load_hf_params(dtype=torch.float32).items()}, torch.bfloat16)
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to(dev).eval()
imp = CalcImplant(hf, tok, 64, fused=False); del hf
eng = Engine(W, pre, tok, implant=imp, B_max=64, S_max=544, use_write=True)
devset = C.load_train('B', shot_q=shot_q)
import random; rng = random.Random(123)
dg = [r for r in devset if r['kind'] == 'gsm']; ds = [r for r in devset if r['kind'] == 'scaled']; rng.shuffle(dg); rng.shuffle(ds)
items = dg[:N // 2] + ds[:N // 2]
out = eng.generate([C.q_ids(tok, r['q']) for r in items], G=1, greedy=True, max_new=320)
CALL = re.compile(r'<calc>(.*?)</calc>')
def _dbg(t, r, j):
    full = pre + C.q_ids(tok, r['q']) + list(t)
    e = imp.emit_ids(torch.tensor([full], device=dev))[0]
    p = len(pre) + len(C.q_ids(tok, r['q'])) + j - 1           # position of the closing '>' (predicts token j)
    prev = [i - len(pre) - len(C.q_ids(tok, r['q'])) for i in range(len(pre), p) if e[i].abs().sum() > 0]
    print('TF at CE fires:', bool(e[p].abs().sum() > 0), '| earlier answer positions with writes:', [x for x in prev if x >= 0][-6:], 'j', j)
st = dict(answers=N, calls=0, wellformed=0, impl_fired=0, impl_exact=0, impl_fired_malformed=0, host_typed_result=0, writes_outside=0, correct=0)
ex = []; mal = []
for t, wf, r in zip(out['tokens'], out['wflag'], items):
    dec = [tok.decode([x]) for x in t]; s_ = ''.join(dec)
    cut = s_.find('\n\n'); cut = len(s_) if cut < 0 else cut
    st['correct'] += int(C.score(tok, t, r['gold'])[0])
    starts = []; c = 0
    for d in dec: starts.append(c); c += len(d)
    tok_at = lambda off: max(i for i, b in enumerate(starts) if b <= off)
    covered = set()
    for m in CALL.finditer(s_[:cut]):
        st['calls'] += 1; want = calc_ref(m.group(1).replace(' ', '').replace(',', ''))
        st['wellformed'] += int(bool(want))
        j = tok_at(m.end() - 1) + 1                      # first token after the closing '>'
        k = j; w = []
        while k < len(t) and wf[k]: w.append(dec[k]); covered.add(k); k += 1
        fired = len(w) > 0; mg = re.match(r' ?(-?[\d.]+)', ''.join(w)); got = mg.group(1) if mg else ''
        st['impl_fired'] += int(fired and bool(want)); st['impl_fired_malformed'] += int(fired and not want)
        st['impl_exact'] += int(fired and bool(want) and got == want)
        if not fired and re.match(r' ?-?\d', s_[m.end():m.end() + 3]): st['host_typed_result'] += 1
        if not want and len(mal) < 5: mal.append(m.group(0))
        if len(ex) < 4: ex.append((m.group(0), repr(''.join(w)), want))
        if want and (not fired or got != want): _dbg(t, r, j) if not fired else None; print('ODD', 'fired' if fired else 'NOFIRE', repr(s_[max(0, m.start() - 15):m.end() + 12]), 'got', repr(got), 'want', want)
    st['writes_outside'] += sum(1 for i, x in enumerate(wf) if x and i not in covered and starts[i] < cut)
print(json.dumps(st, indent=1)); print('examples', ex); print('malformed examples', mal)
print('calls/answer', round(st['calls'] / N, 2), '| implant fires on well-formed', round(st['impl_fired'] / max(1, st['wellformed']), 4),
      '| exact when fired', round(st['impl_exact'] / max(1, st['impl_fired']), 4), '| dev acc (64)', st['correct'] / N)
