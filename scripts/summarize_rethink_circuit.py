"""Summarize the compiled-rethink full runs (results/rethink_circuit/full/*.json chunks) against the plain / calc_s3
baselines (results/gsm_full_implant/<hw>/*.json, other agent) and, optionally, the old Python-state implant on a subset.
Post-hoc analysis only (never used during generation): the phrase is located in the generated tokens by its token ids.
Works on full run outputs and on the reduced files shipped in results/ (scripts/reduce_outputs.py precomputes ph_hits,
first_ok and ok_last with the same rules; the --oldref equivalence check needs full outputs with token ids).
extra tokens = tokens generated from the phrase's 'Hmm' to the end of the generation (phrase + second solution; 0 if no phrase)
first answer = text before the phrase; its first '####' number.

python scripts/summarize_rethink_circuit.py [--full results/rethink_circuit/full] [--base results/gsm_full_implant/a100]
       [--oldref results/rethink_circuit/oldref_test_plain_b4.json]
"""
import os, re, sys, json, glob, argparse, collections
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--full', default='results/rethink_circuit/full')
ap.add_argument('--base', default='results/gsm_full_implant')
ap.add_argument('--oldref', default='')
ap.add_argument('--out', default='results/rethink_circuit/summary.json')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
PH = T("Hmm, wait. Let me rethink this.\n")
NUM = r'####\s*(-?[\d,]*\.?\d+)'


def correct(pred, gold):
    try: return pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


def find_all(ids, pat):
    return [k for k in range(len(ids) - len(pat) + 1) if ids[k:k + len(pat)] == pat]


def load_arm(prefix):
    rows = {}
    for f in sorted(glob.glob(f'{a.full}/{prefix}_*.json')):
        for r in json.load(open(f))['rows']: rows[r['i']] = r
    return rows


summary = {}
for split in ('test', 'hard'):
    for mode in ('plain', 'calc'):
        base = None
        bf = f'{a.base}/{split}_{mode}.json'
        if os.path.exists(bf):
            B = json.load(open(bf))['rows']; base = {r['i']: r for r in B}
            b_first = 100 * sum(r['ok'] for r in B) / len(B)
            last_ok = lambda r: r['ok_last'] if 'ok_last' in r else correct((lambda m: m[-1].replace(',', '') if m else None)(re.findall(NUM, r['out'])), r['gold'])
            b_last = 100 * sum(last_ok(r) for r in B) / len(B)
            summary[f'{split}_{mode}_baseline'] = dict(n=len(B), acc=b_first, acc_last_hash_rule=b_last)
        for imp in ('always', 'gated'):
            rows = load_arm(f'{split}_{mode}_{imp}')
            if not rows: continue
            n = len(rows); ok = sum(r['ok'] for r in rows.values())
            fired = multi = 0; extra = []; first_ok = 0
            for r in rows.values():
                hits = r['ph_hits'] if 'ph_hits' in r else find_all(r['ids'], PH)
                nid = r['n_tok'] if 'ids' not in r else len(r['ids'])
                if hits:
                    fired += 1; multi += len(hits) > 1; extra.append(nid - hits[0])
                if 'first_ok' in r: first_ok += r['first_ok']; continue
                t1 = tok.decode(r['ids'][:hits[0]]) if hits else r['out']
                m = re.search(NUM, t1); first_ok += correct(m.group(1).replace(',', '') if m else None, r['gold'])
            s = dict(n=n, acc=100 * ok / n, first_answer_acc=100 * first_ok / n, fired=fired, fired_twice=multi,
                     mean_extra_tokens=sum(extra) / n, mean_gen_tokens=sum(r['n_tok'] for r in rows.values()) / n,
                     complete=(n == 1319))
            if base is not None:
                common = [i for i in rows if i in base]
                s['vs_baseline'] = dict(n_common=len(common), wrong_to_right=sum(rows[i]['ok'] and not base[i]['ok'] for i in common),
                                        right_to_wrong=sum(base[i]['ok'] and not rows[i]['ok'] for i in common),
                                        baseline_acc_on_common=100 * sum(base[i]['ok'] for i in common) / max(1, len(common)))
            summary[f'{split}_{mode}_{imp}'] = s
if a.oldref and os.path.exists(a.oldref):
    O = {r['i']: r for r in json.load(open(a.oldref))['rows']}
    N = load_arm('test_plain_always')
    c = collections.Counter()
    for i, o in O.items():
        if i not in N: continue
        c['n'] += 1; ids = N[i]['ids']; hits = find_all(ids, PH)
        w0 = o['write_span'][0]
        old_fired = w0 is not None; new_fired = bool(hits)
        c['both_fired'] += old_fired and new_fired; c['neither'] += (not old_fired) and (not new_fired)
        if old_fired and new_fired:
            oh = find_all(o['ids'], PH)
            same = oh and oh[0] == hits[0] and o['ids'][:oh[0]] == ids[:hits[0]]
            c['same_place_same_prefix'] += bool(same)
            c['identical_through_phrase'] += bool(same) and o['ids'][:oh[0] + len(PH)] == ids[:hits[0] + len(PH)]
            c['identical_whole_generation'] += o['ids'][:len(ids)] == ids or ids[:len(o['ids'])] == o['ids']
        c['same_final_correctness'] += o['ok'] == N[i]['ok']
    summary['equivalence_vs_old_python_implant'] = dict(c)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(summary, open(a.out, 'w'), indent=1)
print(json.dumps(summary, indent=1))
