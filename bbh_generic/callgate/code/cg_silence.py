"""Silence check: rows where the gate/breaker never blocked must be bit-identical to the no-gate run.
python cg_silence.py BASE_DIR RESULTS_DIR   (CPU; reads saved JSONs only; shipped: python cg_silence.py ../../integrate/results ../results)
BBH: arm rows vs merged published v2+tools transcripts (aligned by position within task; gold checked).
Sets / ARC: arm rows vs sets_none_heldout.json / arc_none.json (same harness, same bs 16 / same script)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
base_dir, R = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], base_dir]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import cg_report as C
pub = C.merge_published(base_dir)


def per_task(fns):
    per = {}
    for fn in fns:
        try: d = C._open(f'{R}/{fn}')
        except FileNotFoundError: continue
        for r in d['rows']: per.setdefault(r.get('set', '') + '/' + r['task'] if 'set' in r else r['task'], []).append(r)
    return per


def cmp(name, arm, ref):
    nb = same = blk = gold_bad = 0; diff_tasks = {}
    for t, rs in arm.items():
        if t not in ref: continue
        rf = ref[t]
        for i, r in enumerate(rs):
            if i >= len(rf): continue
            if r['gold'] != rf[i]['gold']: gold_bad += 1; continue
            if r['blocked']: blk += 1; continue
            nb += 1
            key = lambda x: x['out_sha256'] if 'out_sha256' in x else __import__('hashlib').sha256(x['out'].encode()).hexdigest()
            if key(r) == key(rf[i]): same += 1             # shipped results: compare SHA-256 of the text
            else: diff_tasks[t] = diff_tasks.get(t, 0) + 1
    print(f'{name}: not-blocked {nb}, identical {same} ({100*same/max(nb,1):.1f}%), blocked {blk}, gold-misaligned {gold_bad}; differing by task {diff_tasks}')


cmp('BBH gate even (bs32) vs published v2+tools (bs32)', per_task(['bbh_gate_even.json']), pub)
cmp('BBH gate odd (bs16) vs published v2+tools (bs32)', per_task(['bbh_gate_odd.json']), pub)
cmp('BBH breaker (bs16) vs published v2+tools (bs32)', per_task(['bbh_breaker.json', 'bbh_breaker_b.json']), pub)
none_sets = per_task(['sets_none_heldout.json'])
if none_sets:
    cmp('own sets gate vs none (bs16)', per_task(['sets_gate3_heldout.json']), none_sets)
    cmp('own sets breaker vs none (bs16)', per_task(['sets_breaker_heldout.json']), none_sets)
an = {'arc': C._open(f'{R}/arc_none.json')['rows']}
for m in ('gate', 'breaker'):
    try: am = C._open(f'{R}/arc_{m}.json')['rows']
    except FileNotFoundError: continue
    cmp(f'ARC {m} vs none', {'arc': am}, {'arc': [dict(r, gold=r['gold']) for r in an['arc']]})
