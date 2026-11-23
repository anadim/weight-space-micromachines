"""[38] verify the implant core against the arm-T tool (tooltext.calc_eval) on every <calc> call of the [35] X traces
(+ an adversarial set). The expression of a call = text between the LAST '<calc>' before a '</calc>' and it (the tool's
own extraction); a '</calc>' with no open tag -> None (ERR). Writes out/verify_core.json + out/mismatches.txt (shipped as
res/). python3 verify_core.py [DIR]: DIR holds the C35 eval dirs ev_Xs*/ ev_baseX/ with the generated text (not shipped;
default ../tooltext/out, where ../tooltext/job.sh copies them)."""
import os, sys, json, glob, time, collections, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'tt/code/rl/gsm_grpo'))
from tooltext import calc_eval, OPEN, CLOSE
from core import ExactCore, sym_str

data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../tooltext/out')
calls = []
for f in sorted(glob.glob(f'{data}/ev_Xs*/*_s*.jsonl') + glob.glob(f'{data}/ev_baseX/*_s*.jsonl')):
    for l in open(f):
        t = json.loads(l)['text']; p = 0
        while True:
            k = t.find(CLOSE, p)
            if k < 0: break
            o = t.rfind(OPEN, 0, k)
            calls.append(t[o + len(OPEN):k] if o >= 0 else None); p = k + len(CLOSE)
adv = ['48/2', '6*60', '10/3', '2,000*3', '(3+4)*2', '-5+2', '1/0', 'abc', '12*0.5', '2^3', '50%*80', '0.1+0.2', '2/3',
       '05', '00', '0', '05.5', '.5', '5.', '1..2', '1.2.3', '2(3)', '(3)(4)', '3//2', '2**3**2', '-2**2', '2**-1', '0**-1',
       '2**13', '2**12', '2**0.5', '(-2)**3', '--5', '+5', '3*-2', '3+-+2', ' 3 + 4 ', '3\n+4', '(3\n+4)', '', '  ', '()',
       '(1+2', '1+2)', '((((1))))', '10%5', '10%', '$5*3', '1,234+1', '1,23+1', '3x4', '3×4', '1e5', '7/7', '-0.0000001',
       '0.0000005', '-0.0000005', '1/3*3', '999999999999999999+1', '1000000000000000000', '1000000000000000001',
       '1000000000000000000.5', '123456789012345678*10', '0.1*0.1*0.1*0.1*0.1*0.1*0.1', '1/7', '22/7', '-1/7', '2.5*4',
       '100-99.99', '3 * 4', '3* *4', '3 ** 2', '1-', '*3', '3+', '4/(2-2)', '(2+3)*(4-1)/5', '7.0', '7.50', '1.234567',
       '1.2345675', '1.2345665', '0.5^2', '1/8', '-1/8', '1/16', '1/1024', '12.5%', '\t5', '5\f+1', None, '1' * 130,
       '9' * 18, '9' * 19, '0.' + '1' * 15, '0.' + '1' * 16, '(' * 30 + '1' + ')' * 30, '1+' * 40 + '1', '3\r+4', '(3\r+4)', '3\v+4', '\v3', '\xa03\xa0', '3\x0c+4', '\u20033', '3\x1c',
       '(3\n)', '\n\n7\n', '2**3**-1', '-2**-2', '(-8)**-1', '0**0', '7**12', '7**13', '1.5**2', '2**(1+1)', '1/3+1/6',
       '1/9999991*1/9999993', '0.5 x 4', '٣+1', '1,٣٣٣', '1_000', '1e3', '0x10', '1j', 'True', '3%%', '(((', ')))', '+-+-3',
       '3 - - 3', '0.000000499999', '0.0000005', '2.0000005', '-2.0000005', '999999999999999999.9999999',
       '1000000000000000000-0.5', '-1000000000000000000', '-1000000000000000001', '123456.7890125', '1.0000000000000002']
import random
rnd = random.Random(0)
def num():
    k = rnd.random()
    if k < .5: return str(rnd.choice([rnd.randint(0, 20), rnd.randint(0, 1000), rnd.randint(0, 10 ** rnd.randint(1, 12))]))
    if k < .85: return f'{rnd.randint(0, 10 ** rnd.randint(0, 6))}.{rnd.randint(0, 10 ** rnd.randint(1, 6))}'
    return rnd.choice(['0', '00', '05', '.5', '5.', '0.0', '1,000', '$3', '50%', '2^2', '3x4', '7.50', '1e2'])
def expr(d=0):
    k = rnd.random()
    if d > 3 or k < .35: e = num()
    elif k < .45: e = '(' + expr(d + 1) + ')'
    elif k < .5: e = rnd.choice(['-', '+']) + expr(d + 1)
    else: e = expr(d + 1) + rnd.choice([' ', '']) + rnd.choice(['+', '-', '*', '/', '*', '/', '**', '//', '+']) + rnd.choice([' ', '']) + expr(d + 1)
    return e
fuzz = [expr() for _ in range(300000)]
adv = adv + fuzz

calls_all = calls + adv
uniq = list(dict.fromkeys(calls_all))
print('calls', len(calls), 'distinct', len(set(calls)), 'with adversarial', len(uniq), flush=True)
core = ExactCore(os.environ.get('CORE_DEV', 'cuda'))
got = {}
t0 = time.time()
BS = 4096
for i in range(0, len(uniq), BS):
    ch = uniq[i:i + BS]
    cp, ln = core.encode(ch)
    sym, slen, feat, inex = core(cp, ln)
    for e, s, x in zip(ch, sym_str(sym, slen), inex.tolist()): got[e] = (s, x)
dt = time.time() - t0
mis = []
for e in uniq:
    want = calc_eval(e)
    if len(want) > 24: want = want[:24]                      # the tool forces at most FMAX tokens (1 token per char here)
    if got[e][0] != want: mis.append((e, want, got[e][0], got[e][1]))
cnt = collections.Counter(calls)
mis_calls = sum(cnt[m[0]] for m in mis if m[0] in cnt)
res = dict(calls=len(calls), distinct=len(set(calls)), adversarial=len(adv), seconds=dt,
           mismatched_distinct=len(mis), mismatched_calls=mis_calls,
           reproduced_calls_pct=100 * (1 - mis_calls / max(1, len(calls))),
           mismatched_adversarial=sum(1 for m in mis if m[0] in set(adv) and m[0] not in cnt),
           fuzz_distinct=len(set(fuzz)), fuzz_err_frac=sum(calc_eval(e) == 'ERR' for e in fuzz[:20000]) / 20000)
os.makedirs('out', exist_ok=True)
json.dump(res, open('out/verify_core.json', 'w'), indent=1)
with open('out/mismatches.txt', 'w') as f:
    for e, w, g, x in mis: f.write(f'{cnt.get(e, 0)}\t{e!r}\twant={w!r}\tgot={g!r}\tinexact={x}\n')
print(json.dumps(res, indent=1)); print(open('out/mismatches.txt').read()[:6000])
