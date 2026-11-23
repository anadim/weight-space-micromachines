"""rl24/bbh driver: GRPO on BBH look-alike train data with the fast BBH engine (bbheng) + packed trainer (bbhtrain),
plus the exactness checks and the BBH-27 eval. Same host install for every mode:
  host = HF checkpoint (bf16 weights, tied head); tools (--tools 1) = integ.attach_all(TOOLS_V2, calc veto) solved ONCE on
  the u0 host (train_library recipe: if the 4 envelope rows are shared/untrained, distinct rows are given in the tools'
  snapshot E0 only), then frozen; the RL'd host keeps reading through E0 (the install-time snapshot).
  gate (--gate PROBE.pt) = cg_compiled.CompiledCallGate math (fixed tensors + 1-float latch per row).

modes:
  rl    : --run NAME, 64 prompts x G8 per update, Dr.GRPO (kl 0), AdamW, resume file every --resume_every updates (default 10, written asynchronously)
          (chain across timeouts: re-run the same command), dev greedy every --dev_every, bf16 ckpts every --ckpt_every.
  check : exactness on the dev slice: greedy vs the HF harness outputs (--hf_json), continuous vs wave, teacher-forced
          ratio + tool write flags vs decode, tool fuzz (reference Micromachine.prime/step vs engine tool step), speed.
  eval  : BBH-27 (harness prompts / extract / norm), greedy, max_new 600 -> --out json (+ per-row outputs).
"""
import os, sys, json, time, math, random, argparse, re, copy, threading
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('WSM_REPO', os.environ.get('WSM_ROOT') or os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))); os.environ.setdefault('WSM_REPO_ROOT', os.environ['WSM_REPO'])   # repo root
sys.path.insert(0, os.path.join(os.environ['WSM_REPO'], 'bbh_generic', 'callgate', 'code'))
from _cgpaths import CODE  # noqa: E402
import integ  # noqa: E402
sys.path.insert(0, HERE)
import envs_ref as ENV  # noqa: E402
import fastqwen as Q  # noqa: E402
import bbheng as BE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['rl', 'check', 'eval'])
ap.add_argument('--host', required=True); ap.add_argument('--weights', default='')
ap.add_argument('--tools', type=int, default=0); ap.add_argument('--gate', default='')
ap.add_argument('--run', default=''); ap.add_argument('--runs_dir', default=os.path.join(os.environ.get('RL24_DATA', 'runs/rl24'), 'bbh', 'runs'))
ap.add_argument('--train', default=''); ap.add_argument('--dev', default='')
ap.add_argument('--updates', type=int, default=300); ap.add_argument('--P', type=int, default=64); ap.add_argument('--G', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-6); ap.add_argument('--T', type=float, default=1.0); ap.add_argument('--clip', type=float, default=0.2)
ap.add_argument('--warmup', type=int, default=5); ap.add_argument('--grad_clip', type=float, default=1.0)
ap.add_argument('--max_new', type=int, default=600); ap.add_argument('--min_tokens', type=int, default=4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--B_max', type=int, default=256); ap.add_argument('--S_max', type=int, default=0)
ap.add_argument('--budget', type=int, default=16384); ap.add_argument('--ckpt', default='auto')
ap.add_argument('--resume_every', type=int, default=10); ap.add_argument('--dev_every', type=int, default=25)
ap.add_argument('--ckpt_every', type=int, default=100); ap.add_argument('--max_hours', type=float, default=3.7)
ap.add_argument('--out', default=''); ap.add_argument('--hf_json', default=''); ap.add_argument('--n_check', type=int, default=128)
ap.add_argument('--bbh_dir', default=os.environ.get('BBH_DIR', '')); ap.add_argument('--bbh_cot', default=os.environ.get('BBH_COT', ''))
ap.add_argument('--tasks', default=''); ap.add_argument('--grad_check', type=int, default=1)
ap.add_argument('--mem_frac', type=float, default=0.0)  # per-process GPU memory cap when several processes share a GPU
a = ap.parse_args()
dev = 'cuda'
if a.mem_frac > 0: torch.cuda.set_per_process_memory_fraction(a.mem_frac)
T_START = time.time()
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
log = lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True)


# ------------------------------------------------------------------------------------------------ host + implants
def install():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.host)
    hf = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.bfloat16).to(dev).eval()
    Ew = hf.get_input_embeddings().weight
    assert hf.lm_head.weight.data_ptr() == Ew.data_ptr() or torch.equal(hf.lm_head.weight, Ew), 'untied head'
    P32 = {k: v.detach().float().clone() for k, v in hf.state_dict().items() if k != 'lm_head.weight'}
    tools, E0, info = [], None, {}
    if a.tools:
        env = [tok.convert_tokens_to_ids(t) for t in ('<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>')]
        shared = bool((Ew[env][1:].float() - Ew[env][0].float()).abs().max() < 1e-6)
        info['envelope_rows_shared'] = shared
        if shared:
            with torch.no_grad(): integ.prep_call_tokens(hf, tok)
        E0 = Ew.detach().clone()
        t0 = time.time(); A = integ.attach_all(hf, tok, integ.TOOLS_V2, veto=integ.VETO); A.detach()
        info['attach_s'] = time.time() - t0
        for n, M in A.nets.items():
            for t_ in list(M.parameters()) + list(M.buffers()): t_.requires_grad_(False)
            tools.append(BE.BBHTool(n, M, BE.B_BUCKETS[-1]))
        info['tools'] = [t.name for t in tools]
        info['dense_rand_exact'] = {t.name: getattr(t, 'fast1_check_rand', None) for t in tools}
    del hf; torch.cuda.empty_cache()
    if a.weights:
        sd = torch.load(a.weights, map_location=dev)
        sd = sd.get('model', sd)
        for k in P32: P32[k].copy_(sd[k].float())
        info['weights'] = a.weights
    gate = None
    if a.gate:
        pr = torch.load(a.gate, map_location=dev); gate = dict(w=pr['w'], b=float(pr['b'])); info['gate'] = a.gate
    return tok, P32, tools, E0, gate, info


tok, P32, TOOLS, E0, GATE, INFO = install()
names = list(P32)
log('install', json.dumps(INFO))


def tids(s): return tok(s, add_special_tokens=False)['input_ids']


def load_rows(path):
    rows = [json.loads(l) for l in open(path)]
    for r in rows: r['ids'] = tids(r['prompt'])
    return rows


def make_engine(rows_for_len, B_max):
    S = a.S_max or (max(len(r['ids']) for r in rows_for_len) + a.max_new + 8)
    S = (S + 63) // 64 * 64
    W = Q.fuse({k: v for k, v in P32.items()}, torch.bfloat16)
    eng = BE.BBHEngine(W, tok, tools=TOOLS, E0=E0, gate=GATE, B_max=B_max, S_max=S, min_tokens=a.min_tokens)
    log(f'engine B_max {eng.B_max} S_max {S} tools {len(TOOLS)} gate {GATE is not None}')
    return eng


@torch.no_grad()
def sync_weights(eng):
    W = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16)
    eng.W['E'].copy_(W['E']); eng.W['lnf'].copy_(W['lnf'])
    for l0, l1 in zip(eng.W['L'], W['L']):
        for k in l0: l0[k].copy_(l1[k])


def text_of(toks): return tok.decode(toks).replace('<|endoftext|>', '')


def calls_of(wf): return sum(1 for i in range(len(wf)) if wf[i] and (i == 0 or not wf[i - 1]))


def greedy_eval(eng, rows, tag):
    out = eng.generate([r['ids'] for r in rows], G=1, max_new=a.max_new, greedy=True)
    per = {}; res = []
    for r, t, w, la in zip(rows, out['tokens'], out['wflag'], out['lat']):
        txt = text_of(t); ok = bool(ENV.score(txt, r))
        per.setdefault(r['task'], []).append(ok)
        res.append(dict(task=r['task'], ok=ok, ntok=len(t), writes=sum(w), calls=calls_of(w), latched=int(any(la)), tc=t.count(BE.TC)))
    acc = {k: 100 * sum(v) / len(v) for k, v in per.items()}
    s = dict(tag=tag, acc=sum(acc.values()) / len(acc), per_task=acc, tok_q=sum(x['ntok'] for x in res) / len(res),
             writes_q=sum(x['writes'] for x in res) / len(res), fire_rows=100 * sum(x['writes'] > 0 for x in res) / len(res),
             latch_rows=100 * sum(x['latched'] for x in res) / len(res), tc_rows=100 * sum(x['tc'] > 0 for x in res) / len(res),
             tok_s=out['stats']['tok_s'], seconds=out['stats']['seconds'])
    return s, out, res


# ------------------------------------------------------------------------------------------------ docs for the trainer
def docs_of(qids_list, out, P, G, adv, all_rows=False):
    docs = []
    for r in range(P * G):
        if not all_rows and adv[r] == 0.0: continue                 # Dr.GRPO: A = 0 rows give exactly zero loss/grad
        t = out['tokens'][r]
        if not t: continue
        masked = [(i, eng.eos) for i in range(min(a.min_tokens, len(t)))]
        masked += [(i, BE.TC) for i, l in enumerate(out['lat'][r]) if l]
        docs.append(dict(qi=r // G, q=qids_list[r // G], r=t, lp=out['logp'][r], adv=float(adv[r]), masked=masked, row=r))
    return docs


# ================================================================================================ eval (BBH-27)
if a.mode == 'eval':
    def extract(text):
        m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''

    def norm(x, task):
        x = x.strip().strip('.').strip()
        if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x) and task not in ('word_sorting', 'dyck_languages'): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
        if task in ('multistep_arithmetic_two', 'object_counting'): m = re.search(r'-?\d+', x); return m.group() if m else x
        return x.lower()
    tasks = a.tasks.split(',') if a.tasks else sorted(f[:-5] for f in os.listdir(a.bbh_dir) if f.endswith('.json'))
    pre = lambda t: open(f'{a.bbh_cot}/{t}.txt').read().split('-----', 1)[1].strip() + '\n\n'
    rows = []
    for t in tasks:
        for e in json.load(open(f'{a.bbh_dir}/{t}.json'))['examples']:
            rows.append(dict(task=t, gold=e['target'], ids=tids(pre(t) + f"Q: {e['input']}\nA: Let's think step by step.")))
    eng = make_engine(rows, a.B_max)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['ids']))
    out = eng.generate([rows[i]['ids'] for i in order], G=1, max_new=a.max_new, greedy=True)
    res = [None] * len(rows)
    for j, i in enumerate(order):
        t = out['tokens'][j]; txt = text_of(t); p = extract(txt); r = rows[i]
        res[i] = dict(task=r['task'], gold=r['gold'], pred=p, ok=norm(p, r['task']) == norm(r['gold'], r['task']), out=txt,
                      ntok=len(t), writes=sum(out['wflag'][j]), latched=int(any(out['lat'][j])), ncall=t.count(BE.TC))
    per = {}
    for r in res: per.setdefault(r['task'], []).append(r['ok'])
    acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}
    TOOL12 = ['tracking_shuffled_objects_three_objects', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
              'logical_deduction_three_objects', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'temporal_sequences',
              'navigate', 'web_of_lies', 'multistep_arithmetic_two', 'boolean_expressions', 'object_counting']
    t12 = [t for t in TOOL12 if t in acc]; o15 = [t for t in acc if t not in TOOL12]
    summ = dict(host=a.host, weights=a.weights, tools=a.tools, gate=a.gate, n=len(rows), mean27=sum(acc.values()) / len(acc),
                tool12=sum(acc[t] for t in t12) / max(1, len(t12)), other15=sum(acc[t] for t in o15) / max(1, len(o15)),
                tok_q=sum(r['ntok'] for r in res) / len(res), fire_rows=100 * sum(r['writes'] > 0 for r in res) / len(res),
                call_rows=100 * sum(r['ncall'] > 0 for r in res) / len(res), latch_rows=100 * sum(r['latched'] for r in res) / len(res),
                per_task=acc, stats=out['stats'], install=INFO)
    log('SUMMARY', json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summ.items() if k not in ('per_task', 'install', 'stats')}))
    log('PER_TASK', json.dumps({k: round(v, 1) for k, v in acc.items()}))
    log('STATS', json.dumps(out['stats']))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(summary=summ, rows=res), open(a.out, 'w'))
    sys.exit(0)

# ================================================================================================ check
if a.mode == 'check':
    rows = load_rows(a.dev)
    hf = json.load(open(a.hf_json))['rows'] if a.hf_json else None
    n = min(a.n_check, len(rows))
    rng = random.Random(0); sel = sorted(rng.sample(range(len(rows)), n))
    R = [rows[i] for i in sel]
    eng = make_engine(rows, a.B_max)
    rep = dict(install=INFO, n=n)
    # (1) greedy vs the HF harness (hostcheck.py outputs, same rows), continuous batching
    out = eng.generate([r['ids'] for r in R], G=1, max_new=a.max_new, greedy=True)
    rep['speed_greedy_cont'] = out['stats']
    if hf is not None:
        same = 0; div = []
        for k, i in enumerate(sel):
            mine = text_of(out['tokens'][k]); ref = hf[i]['out']
            if mine == ref: same += 1
            else:
                j = next((p for p in range(min(len(mine), len(ref))) if mine[p] != ref[p]), min(len(mine), len(ref)))
                div.append(dict(i=i, task=rows[i]['task'], char=j, mine=mine[max(0, j - 40):j + 40], ref=ref[max(0, j - 40):j + 40],
                                ok_mine=bool(ENV.score(mine, rows[i])), ok_ref=bool(hf[i]['ok'])))
        rep['vs_hf'] = dict(identical=same, n=n, acc_mine=100 * sum(ENV.score(text_of(t), r) for t, r in zip(out['tokens'], R)) / n,
                            acc_ref=100 * sum(hf[i]['ok'] for i in sel) / n, diverged=div[:40])
        # near-tie test at the first divergent token: top-2 logit gap of the engine's own (teacher-forced) logits
        gaps = []
        for d in div[:40]:
            k = sel.index(d['i']); mt = out['tokens'][k]
            reft = tids(hf[d['i']]['out'])
            p = next((j for j in range(min(len(mt), len(reft))) if mt[j] != reft[j]), None)
            if p is None: continue
            ids = torch.tensor([R[k]['ids'] + mt[:p]], device=dev)
            with torch.no_grad():
                hn = Q.fwd_chunk(eng.W, ids, 0, eng.cos_t, eng.sin_t)
                lg = torch.nn.functional.linear(hn[0, -1], eng.W['E']).float()
            top = lg.topk(2).values
            gaps.append(dict(i=d['i'], pos=p, gap=float(top[0] - top[1]), mine=mt[p], ref=reft[p], lg_mine=float(lg[mt[p]]), lg_ref=float(lg[reft[p]])))
        rep['div_gaps'] = gaps
    # (2) batch-shape invariance: the same rows through 32-slot continuous batching (admission + compaction) vs all at once
    outw = eng.generate([r['ids'] for r in R], G=1, max_new=a.max_new, greedy=True, B_use=32)
    rep['speed_greedy_b32'] = outw['stats']
    rep['b32_vs_all_identical'] = sum(x == y for x, y in zip(out['tokens'], outw['tokens']))
    # (3) sampled rollout (RL shape) -> teacher-forced ratio, write flags vs decode
    P = min(a.P, n); G = a.G
    qids = [r['ids'] for r in R[:P]]
    t0 = time.time(); outs = eng.generate(qids, G=G, T=a.T, max_new=a.max_new, seed=1)
    rep['speed_sample'] = outs['stats']
    import bbhtrain as BT
    tr = BT.BBHTrainer(P32, names, eng, T=a.T, clip=a.clip, budget=a.budget, ckpt=a.ckpt)
    docs = docs_of(qids, outs, P, G, [1.0] * (P * G), all_rows=True)
    st, new_lp, tf_wf = tr.grads(docs, P * G, a.max_new, check_only=True)
    dl = [abs(x - y) for j, d in enumerate(docs) for x, y in zip(new_lp[j], d['lp'])]
    rat = [math.exp(x - y) for j, d in enumerate(docs) for x, y in zip(new_lp[j], d['lp'])]
    rep['tf'] = dict(ntok=len(dl), mean_abs_dlp=sum(dl) / len(dl), max_abs_dlp=max(dl), mean_ratio=sum(rat) / len(rat),
                     frac_dlp_gt_0_05=sum(x > 0.05 for x in dl) / len(dl))
    if TOOLS:
        mm = sum(int(x != y) for j, d in enumerate(docs) for x, y in zip(tf_wf[j], outs['wflag'][d['row']]))
        nw = sum(sum(outs['wflag'][d['row']]) for d in docs)
        rep['wflag_tf_vs_decode'] = dict(mismatch=mm, decode_writes=nw, rows=len(docs))
    # (4) tool fuzz: reference Micromachine.prime_sym/step (its own sparse/F64 stacks, as attached in the HF harness),
    #     batched over 32 sampled rows, vs the trainer's teacher-forced emit (DenseStack) and the decode write flags
    if TOOLS:
        # rows where a tool wrote in the decode first (then the longest), up to 400 response tokens, right-padded
        byt = {}
        for d in sorted(docs, key=lambda d: (-sum(outs['wflag'][d['row']]), -len(d['r']))): byt.setdefault(R[d['row'] // G]['task'], []).append(d)
        dd = []
        while len(dd) < 32 and any(byt.values()):
            for v in byt.values():
                if v and len(dd) < 32: dd.append(v.pop(0))
        Lr = [min(400, len(d['r'])) for d in dd]; Lc = max(Lr)
        valid = torch.tensor([[p < L for p in range(Lc)] for L in Lr], device=dev)
        fz = {}
        for t in TOOLS:
            M = t.M
            tails = torch.tensor([([eng.eos] * eng.Kt + list(d['q']))[-t.Kt:] for d in dd], device=dev)
            resp = torch.tensor([list(d['r'][:Lc]) + [eng.eos] * (Lc - len(d['r'][:Lc])) for d in dd], device=dev)
            with torch.no_grad():
                es = [M.prime_sym(M.symbols(E0[tails]))[:, -1].float()]                   # reference (sparse stacks)
                for p in range(Lc - 1):
                    _, e = M.step(E0[resp[:, p]]); es.append(e.float())
                ref = torch.stack(es, 1)                                                  # (32, Lc, n_out)
                got = BT.emit_tf(t, torch.cat([tails, resp[:, :Lc - 1]], 1), E0)[:, t.Kt - 1:]   # trainer
                rows_ = torch.arange(len(dd), device=dev)                                 # engine prime + graph-step math
                e0, st = t.prime_many(E0[tails]); t.set_rows(rows_, st)
                ge = [e0]
                for p in range(Lc - 1): ge.append(t.step(E0[resp[:, p]], len(dd)))
                ge = torch.stack(ge, 1)
            fire = (ref @ t.Dnz) != 0
            fz[t.name] = dict(steps=int(valid.sum()), mismatch_trainer=int(((ref != got).any(-1) & valid).sum()),
                              mismatch_engine=int(((ref != ge).any(-1) & valid).sum()), fire=int((fire & valid).sum()))
        rep['tool_fuzz'] = fz
        rep['tool_fuzz_total'] = {k: sum(v[k] for v in fz.values()) for k in ('steps', 'mismatch_trainer', 'mismatch_engine', 'fire')}
    # (4b) trainer gradient vs an independent HF autograd reference (bf16 HF model + the same teacher-forced writes added
    #      by a pre-hook on model.norm, plain log_softmax over the full vocabulary with the same masks), 12 docs, adv +-1
    if a.grad_check:
        from transformers import AutoModelForCausalLM
        gdocs = [dict(d, adv=(1.0 if j % 2 == 0 else -1.0)) for j, d in enumerate(sorted(docs, key=lambda d: -sum(outs['wflag'][d['row']]))[:12])]
        for k in names: P32[k].grad = None
        tr.grads(gdocs, len(gdocs), a.max_new)
        g_tr = {k: P32[k].grad.clone() for k in names}
        for k in names: P32[k].grad = None
        def hf_grads(dt):
            hf = AutoModelForCausalLM.from_pretrained(a.host, dtype=dt).to(dev)
            hf.load_state_dict({k: v.to(dt) for k, v in P32.items()}, strict=False); hf.tie_weights(); hf.train(False)
            pend = {}
            hk = hf.model.norm.register_forward_pre_hook(lambda m, x: (x[0] + pend['w'].to(x[0].dtype),) if 'w' in pend else None)
            for d in gdocs:
                q, r = list(d['q']), list(d['r']); ql = len(q); rl = len(r)
                ids = torch.tensor([q + r], device=dev)
                pend.clear()
                if TOOLS:
                    mb = tr.build([(d['qi'], [0], ql + rl)], [d]); wr, _ = tr.writes(mb, [d])
                    w = torch.zeros(1, ql + rl, Q.D, device=dev); w[0, ql - 1:ql - 1 + rl] = wr; pend['w'] = w
                lg = hf(ids).logits[0, ql - 1:ql - 1 + rl].float() / a.T
                for t_, mt in d['masked']:
                    lg = lg.index_put((torch.tensor([t_], device=dev), torch.tensor([mt], device=dev)), torch.tensor(-1e30, device=dev))
                lp = torch.log_softmax(lg, -1).gather(-1, torch.tensor(r, device=dev)[:, None])[:, 0]
                old = torch.tensor(d['lp'], device=dev); ratio = torch.exp(torch.clamp(lp - old, -20, 20))
                A_ = d['adv']; per = -torch.minimum(ratio * A_, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A_)
                (per.sum() / (len(gdocs) * a.max_new)).backward()
            hk.remove()
            hp = dict(hf.named_parameters())
            g = {k: (hp[k].grad.float() if hp[k].grad is not None else torch.zeros_like(g_tr[k])) for k in names}
            del hf, hp; torch.cuda.empty_cache()
            return g

        def cmp(g1, g2):
            num = d1 = d2 = 0.0; per_t = []
            for k in names:
                x, y = g1[k], g2[k]
                num += float((x * y).sum()); d1 += float((x * x).sum()); d2 += float((y * y).sum())
                per_t.append((float((x * y).sum() / (x.norm() * y.norm() + 1e-30)), k))
            per_t.sort()
            return dict(cos=num / (math.sqrt(d1 * d2) + 1e-30), norm_ratio=math.sqrt(d1 / max(d2, 1e-30)), worst=per_t[:3])
        g16 = hf_grads(torch.bfloat16); g32 = hf_grads(torch.float32)
        rep['grad_check'] = dict(docs=len(gdocs), trainer_vs_hf32=cmp(g_tr, g32), hf16_vs_hf32=cmp(g16, g32),
                                 trainer_vs_hf16=cmp(g_tr, g16))
        del g16, g32
        for k in names: P32[k].grad = None
    # (5) a full RL update timing (rollout + grads) at the RL shape, no optimizer step
    if a.P <= n:
        torch.cuda.synchronize(); t0 = time.time()
        outs = eng.generate([r['ids'] for r in R[:a.P]], G=a.G, T=a.T, max_new=a.max_new, seed=2)
        torch.cuda.synchronize(); t1 = time.time()
        Rw = [float(ENV.score(text_of(t), R[r // a.G])) for r, t in enumerate(outs['tokens'])]
        adv = []
        for p in range(a.P):
            g = Rw[p * a.G:(p + 1) * a.G]; m = sum(g) / len(g); adv += [x - m for x in g]
        docs = docs_of(qids, outs, a.P, a.G, adv)
        st = tr.grads(docs, a.P * a.G, a.max_new) if docs else {}
        torch.cuda.synchronize(); t2 = time.time()
        for k in names: P32[k].grad = None
        rep['update_timing'] = dict(rollout_s=t1 - t0, train_s=t2 - t1, total_s=t2 - t0, gen_tokens=outs['stats']['tokens'],
                                    tok_s=outs['stats']['tok_s'], train_stats=st, reward=sum(Rw) / len(Rw))
    log('CHECK', json.dumps({k: v for k, v in rep.items() if k not in ('vs_hf',)}, default=str)[:6000])
    if 'vs_hf' in rep: log('VS_HF', json.dumps({k: v for k, v in rep['vs_hf'].items() if k != 'diverged'}))
    json.dump(rep, open(a.out, 'w'), default=str, indent=1)
    sys.exit(0)

# ================================================================================================ rl
assert a.run
RUN = os.path.join(a.runs_dir, a.run); os.makedirs(RUN, exist_ok=True)
OUTD = os.path.join(HERE, 'out', a.run); os.makedirs(OUTD, exist_ok=True)
train = load_rows(a.train); dev_rows = load_rows(a.dev)
eng = make_engine(train + dev_rows, a.B_max)
import bbhtrain as BT  # noqa: E402
tr = BT.BBHTrainer(P32, names, eng, T=a.T, clip=a.clip, budget=a.budget, ckpt=a.ckpt)
params = [P32[k] for k in names]
for p in params: p.requires_grad_(True)
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
# prompt order: a seeded permutation per epoch
rng = np.random.default_rng(1000 + a.seed); order = []
while len(order) < a.updates * a.P: order += rng.permutation(len(train)).tolist()
u0 = 0; hist = []
rpath = os.path.join(RUN, 'resume.pt')
if os.path.exists(rpath):
    sv = torch.load(rpath, map_location=dev, weights_only=False)
    for k in names: P32[k].data.copy_(sv['P32'][k])
    opt.load_state_dict(sv['opt']); u0 = sv['u']; hist = sv['hist']
    torch.set_rng_state(sv['rng_cpu'].cpu()); torch.cuda.set_rng_state(sv['rng_cuda'].cpu())
    log(f'resumed at u{u0}')
    del sv
logf = open(os.path.join(OUTD, 'log.jsonl'), 'a')
devf = open(os.path.join(OUTD, 'dev.jsonl'), 'a')


_saver = [None]


def save_resume(u, wait=False):
    # snapshot to host memory synchronously, write in a background thread (one write in flight; the write is IO-bound)
    if _saver[0] is not None: _saver[0].join()
    def cpu(x): return x.detach().to('cpu', copy=True) if torch.is_tensor(x) else x
    sd = opt.state_dict()
    obj = dict(P32={k: cpu(P32[k]) for k in names},
               opt=dict(state={i: {kk: cpu(vv) for kk, vv in v.items()} for i, v in sd['state'].items()},
                        param_groups=copy.deepcopy(sd['param_groups'])),
               u=u, hist=list(hist), rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(), args=vars(a))
    tmp = rpath + '.tmp'
    def w(): torch.save(obj, tmp); os.replace(tmp, rpath)
    _saver[0] = threading.Thread(target=w); _saver[0].start()
    if wait: _saver[0].join()


def save_ckpt(u):
    torch.save({k: P32[k].detach().to(torch.bfloat16) for k in names}, os.path.join(RUN, f'u{u}.pt'))


def run_dev(u):
    sync_weights(eng)
    s, _, _ = greedy_eval(eng, dev_rows, f'u{u}')
    s['u'] = u; devf.write(json.dumps(s) + '\n'); devf.flush()
    log(f'DEV u{u} acc {s["acc"]:.2f} tok/q {s["tok_q"]:.0f} fire {s["fire_rows"]:.0f}% latch {s["latch_rows"]:.0f}% tc {s["tc_rows"]:.0f}% ({s["seconds"]:.0f}s)')


if u0 == 0:
    run_dev(0)
for u in range(u0 + 1, a.updates + 1):
    if time.time() - T_START > a.max_hours * 3600:
        if (u - 1) % a.resume_every and u - 1 > u0: save_resume(u - 1)
        if _saver[0] is not None: _saver[0].join()
        log('time budget reached; exiting for the chain'); sys.exit(3)
    t0 = time.time()
    sync_weights(eng)
    idx = order[(u - 1) * a.P: u * a.P]
    qids = [train[i]['ids'] for i in idx]
    out = eng.generate(qids, G=a.G, T=a.T, max_new=a.max_new, seed=a.seed * 1000003 + u)
    t1 = time.time()
    Rw = [float(ENV.score(text_of(t), train[idx[r // a.G]])) for r, t in enumerate(out['tokens'])]
    adv = []
    for p in range(a.P):
        g = Rw[p * a.G:(p + 1) * a.G]; m = sum(g) / len(g); adv += [x - m for x in g]
    rec = dict(u=u, reward=sum(Rw) / len(Rw), rollout_s=t1 - t0, gen_tok=out['stats']['tokens'], tok_s=out['stats']['tok_s'],
               tok_q=out['stats']['tokens'] / len(Rw), occupancy=out['stats']['occupancy'],
               writes_q=sum(map(sum, out['wflag'])) / len(Rw), latch_rows=sum(any(l) for l in out['lat']) / len(Rw),
               tc_rows=sum(BE.TC in t for t in out['tokens']) / len(Rw), trunc=sum(len(t) >= a.max_new for t in out['tokens']) / len(Rw))
    fam = {}
    for r, x in enumerate(Rw): fam.setdefault(train[idx[r // a.G]]['task'], []).append(x)
    rec['reward_task'] = {k: round(sum(v) / len(v), 3) for k, v in fam.items()}
    docs = docs_of(qids, out, a.P, a.G, adv)
    if u == 1 and docs:                                       # teacher-forced ratio of the exact rollout policy (u1 check)
        st0, new_lp, tf_wf = tr.grads(docs, a.P * a.G, a.max_new, check_only=True)
        dl = [abs(x - y) for j, d in enumerate(docs) for x, y in zip(new_lp[j], d['lp'])]
        rec['u1_tf'] = dict(mean_abs_dlp=sum(dl) / len(dl), max_abs_dlp=max(dl), mean_ratio=sum(math.exp(x - y) for j, d in enumerate(docs) for x, y in zip(new_lp[j], d['lp'])) / len(dl))
        if TOOLS: rec['u1_wflag_mismatch'] = sum(int(x != y) for j, d in enumerate(docs) for x, y in zip(tf_wf[j], out['wflag'][d['row']]))
    for g in opt.param_groups: g['lr'] = a.lr * min(1.0, u / a.warmup)
    if docs:
        st = tr.grads(docs, a.P * a.G, a.max_new)
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip))
        opt.step()
    else:
        st, gn = {}, 0.0
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    rec.update(train_s=time.time() - t1, total_s=time.time() - t0, grad_norm=gn, n_docs=len(docs),
               **{k: st[k] for k in ('loss', 'ratio', 'clipfrac', 'n_micro', 'ntok_packed', 'ckpt', 'max_dlp') if k in st})
    hist.append(dict(u=u, reward=rec['reward']))
    logf.write(json.dumps(rec) + '\n'); logf.flush()
    log(f'u{u} R {rec["reward"]:.3f} tok/q {rec["tok_q"]:.0f} roll {rec["rollout_s"]:.1f}s ({rec["tok_s"]:.0f} tok/s) train {rec["train_s"]:.1f}s '
        f'tot {rec["total_s"]:.1f}s docs {len(docs)} ratio {st.get("ratio", 0):.4f} clip {st.get("clipfrac", 0):.4f} gn {gn:.3f}'
        + (f' u1_tf {rec["u1_tf"]}' if 'u1_tf' in rec else ''))
    if u % a.dev_every == 0: run_dev(u)
    if u % a.ckpt_every == 0 or u == a.updates: save_ckpt(u)
    if u % a.resume_every == 0 or u == a.updates: save_resume(u)
if _saver[0] is not None: _saver[0].join()
open(os.path.join(OUTD, 'DONE'), 'w').write('done\n')
open(os.path.join(RUN, 'DONE'), 'w').write('done\n')
log('DONE')
