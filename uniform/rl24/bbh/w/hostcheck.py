"""HOST CHECK (task [26]): accuracy / tok per question / tool fire rates of a host on the BBH look-alike DEV set (fresh
seed 400; never BBH test), BBH-27 harness format and decoding (3-shot CoT prompt of the task, greedy, min_tokens 4,
stop eos / '\\n\\nQ', max_new 600), with and without the 8 v2 tools (integ.TOOLS_V2 + calc veto) attached.
Scored with the team harness norms (envs.score). Tool fires are counted post hoc by teacher-forced Micromachine.emit
over prompt + output (= the decode-time writes).
python hostcheck.py OUT.json --host PATH_OR_HF_ID --tools 0|1 [--data dev.jsonl --bs 64 --max_new 600]
"""
import os, sys, json, time, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('WSM_REPO', os.environ.get('WSM_ROOT') or os.path.abspath(os.path.join(HERE, '..', '..', '..', '..')))   # repo root
os.environ.setdefault('WSM_REPO_ROOT', os.environ['WSM_REPO'])
sys.path.insert(0, os.path.join(os.environ['WSM_REPO'], 'bbh_generic', 'callgate', 'code'))
from _cgpaths import CODE  # noqa: E402  (integration paths)
import integ  # noqa: E402
import cg_gate_gen as CG  # noqa: E402
sys.path.insert(0, HERE)
import envs_ref as E  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--host', required=True); ap.add_argument('--tools', type=int, default=0)
ap.add_argument('--data', default=os.path.join(HERE, 'data', 'dev.jsonl'))
ap.add_argument('--gate', default=''); ap.add_argument('--bs', type=int, default=64); ap.add_argument('--max_new', type=int, default=600)
a = ap.parse_args()
dev = 'cuda'
rows = [json.loads(l) for l in open(a.data)]
tok = AutoTokenizer.from_pretrained(a.host)
model = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.bfloat16).to(dev).eval()
net = None; attach_err = None
if a.tools:
    try:
        t0 = time.time()
        Ew = model.get_input_embeddings().weight
        env = [tok.convert_tokens_to_ids(t) for t in ('<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>')]
        shared = bool((Ew[env][1:].float() - Ew[env][0].float()).abs().max() < 1e-6)
        print('envelope rows shared (untrained):', shared, flush=True)
        if shared:      # = train_library: distinct envelope rows for the tools' install only; host rows restored after
            keep = Ew[env].detach().clone()
            with torch.no_grad(): integ.prep_call_tokens(model, tok)
        net = integ.attach_all(model, tok, integ.TOOLS_V2, veto=integ.VETO)
        if shared:
            with torch.no_grad(): Ew[env] = keep
        print(f'attached v2 tools ({time.time()-t0:.0f}s)', flush=True)
    except Exception as ex:
        attach_err = repr(ex); print('ATTACH FAILED', attach_err, flush=True)
PROBE = torch.load(a.gate, map_location=dev) if a.gate else None      # python call gate (== compiled, cg own sets)
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['prompt']))
res = [None] * len(rows); t0 = time.time(); ntok = 0
if not (a.tools and net is None):
    for b in range(0, len(order), a.bs):
        ch = [rows[i] for i in order[b:b + a.bs]]
        gens, stats = CG.generate(model, tok, net, [r['prompt'] for r in ch], a.max_new, dev, mode='gate' if a.gate else 'none',
                                  probe=PROBE, min_tokens=4)
        for i, g, st in zip(order[b:b + a.bs], gens, stats):
            txt = tok.decode(g).replace('<|endoftext|>', '')
            res[i] = dict(task=rows[i]['task'], ok=E.score(txt, rows[i]), ntok=len(g), ncall=st['ncall'], blocked=st['blocked'], out=txt, gen=g)
            ntok += len(g)
        print(f'{b + len(ch)}/{len(rows)} {time.time()-t0:.0f}s', flush=True)
    dt = time.time() - t0
    fires = {}
    if net is not None:                      # post-hoc teacher-forced writes per tool over prompt + output
        Em = model.get_input_embeddings()
        with torch.no_grad():
            for i, r in enumerate(rows):
                pid = tok(r['prompt'], add_special_tokens=False)['input_ids']; g = res[i]['gen']
                emb = Em(torch.tensor([pid + g], device=dev))
                f = {}
                for n, m in net.nets.items():
                    e, _ = m.emit(emb)
                    nz = (e[0, len(pid) - 1:len(pid) - 1 + len(g)].float().abs().sum(-1) > 0)
                    f[n] = int(nz.sum())
                res[i]['writes'] = f
        for n in net.nets:
            fires[n] = dict(rows_fired=100 * sum(r['writes'][n] > 0 for r in res) / len(res),
                            writes_per_q=sum(r['writes'][n] for r in res) / len(res))
    per = {}
    for r in res: per.setdefault(r['task'], []).append(r['ok'])
    acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}
    summ = dict(host=a.host, tools=a.tools, gate=a.gate, n=len(rows), mean_task_acc=sum(acc.values()) / len(acc), per_task=acc,
                tok_per_q=ntok / len(rows), call_rows=100 * sum(r['ncall'] > 0 for r in res) / len(res),
                calls_per_q=sum(r['ncall'] for r in res) / len(res), fires=fires, seconds=dt,
                any_fire_rows=(100 * sum(any(v > 0 for v in r['writes'].values()) for r in res) / len(res)) if net else None)
    for r in res: r.pop('gen', None)
else:
    summ = dict(host=a.host, tools=a.tools, gate=a.gate, attach_error=attach_err); res = []
print('SUMMARY', json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summ.items() if k != 'per_task'}), flush=True)
print('PER_TASK', json.dumps({k: round(v, 1) for k, v in summ.get('per_task', {}).items()}), flush=True)
json.dump(dict(summary=summ, rows=res), open(a.out, 'w'))
