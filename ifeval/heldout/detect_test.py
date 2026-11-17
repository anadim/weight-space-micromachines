"""Detection check: run the circuit over the PROMPT tokens only and compare latched rules / captured arguments with the
generator's ground truth. python detect_test.py data/dev.jsonl [--show type]"""
import argparse, json, re, os, torch
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from circuit import Circuit, HP
ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('--show', default=''); ap.add_argument('--bs', type=int, default=64)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16)   # CPU: only shapes are used
C = Circuit(tok, model)
docs = [json.loads(l) for l in open(a.data)]
MAP = {'punctuation:no_comma': 'nc', 'change_case:english_lowercase': 'lower', 'change_case:english_capital': 'caps', 'startend:quotation': 'quote',
       'startend:end_checker': 'end', 'detectable_content:postscript': 'ps', 'keywords:forbidden_words': 'forbid', 'combination:repeat_prompt': 'repeat',
       'combination:two_responses': 'two', 'length_constraints:number_paragraphs': 'para', 'length_constraints:nth_paragraph_first_word': 'nth',
       'detectable_format:title': 'title', 'detectable_format:number_highlighted_sections': 'high', 'detectable_format:number_bullet_lists': 'bul'}
def wh(w):
    h = 0
    for ch in w.lower(): h = (h * 131 + ord(ch)) % HP
    return h
def msgs(d):
    m = []
    if d.get('system'): m.append({'role': 'system', 'content': d['system']})
    m += d.get('history') or []; m.append({'role': 'user', 'content': d['prompt']}); return m
stat = defaultdict(Counter); bad = defaultdict(list)
for b in range(0, len(docs), a.bs):
    batch = docs[b:b + a.bs]
    P = [tok(tok.apply_chat_template(msgs(d), tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False).input_ids for d in batch]
    L = max(map(len, P)); x = torch.tensor([[151643] * (L - len(p)) + p for p in P], device=os.environ.get('CIRC_DEV', 'cuda'))
    C.reset(len(P))
    for t in range(L): C.step(x[:, t])
    rep = C.report()
    for d, r in zip(batch, rep):
        truth = {MAP[i]: kw for i, kw in zip(d['instruction_id_list'], d['kwargs']) if i in MAP}
        for k in MAP.values():
            det = k in r; tr = k in truth
            if k == 'para': det = det and 'star3' in r and max(r['N_par'], r['N_sec']) >= 2
            stat[k]['tp' if det and tr else 'fn' if tr else 'fp' if det else 'tn'] += 1
            if det != tr: bad[k].append((d['key'], 'FN' if tr else 'FP', d['prompt'][-300:]))
            if det and tr:     # argument checks
                ok = True
                if k == 'end': ok = r['end_phrase'].strip() == truth[k]['end_phrase'].strip()
                if k == 'ps': ok = ('pps' in r) == (truth[k]['postscript_marker'] == 'P.P.S')
                if k == 'forbid': ok = sorted(r['forbid_h']) == sorted(wh(w) for w in truth[k]['forbidden_words'])
                if k == 'repeat': ok = r['rep'].strip() == truth[k]['prompt_to_repeat'].strip()
                if k == 'para': ok = (r['N_par'] or r['N_sec']) == truth[k]['num_paragraphs'] and 'star3' in r
                if k == 'nth':
                    kk = r['nth_k'] if not r['nth_last'] else r['N_par']
                    ok = kk == truth[k]['nth_paragraph'] and r['nth_word'].strip().lower() == truth[k]['first_word'] and r['N_par'] == truth[k]['num_paragraphs'] and 'star3' not in r
                if k == 'high': ok = r['N_high'] == truth[k]['num_highlights']
                if k == 'bul': ok = r['N_bul'] == truth[k]['num_bullets']
                stat[k]['arg_ok' if ok else 'arg_bad'] += 1
                if not ok: bad[k].append((d['key'], 'ARG', {kk: r[kk] for kk in r if kk in ('end_phrase', 'rep', 'forbid_h', 'N_par', 'N_sec', 'N_bul', 'N_high', 'nth_k', 'nth_word', 'nth_last')}, truth[k], d['prompt'][-300:]))
for k in MAP.values():
    s = stat[k]; print(f"{k:8s} tp {s['tp']:4d} fn {s['fn']:3d} fp {s['fp']:3d} tn {s['tn']:4d} | args ok {s['arg_ok']:4d} bad {s['arg_bad']:3d}")
fires_any = sum(1 for k in []) 
for k in (a.show.split(',') if a.show else []):
    for x in bad[k][:40]: print(k, x)
