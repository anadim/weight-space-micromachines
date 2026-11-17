"""Reduce a GSM run output (scripts/run_gsm.py, run_gsm_circuit.py, rethink_v2/run_gsm_v2.py) to the compact form kept
in results/: per problem only the index, gold, prediction, correctness and token counts, plus the few post-hoc fields the
summary scripts need. Generated text, token ids and question text are dropped. The derived fields are computed here from
the full output, with the same rules the summary scripts use on full outputs:

  n_tok        number of generated tokens (len(ids) if the run did not store it)
  ok_last      correctness of the LAST '####' number in the generated text (scripts/summarize_rethink_circuit.py)
  ph_hits      token positions where the rethink phrase "Hmm, wait. Let me rethink this.\\n" starts (by token ids)
  first_ok     correctness of the first '####' number in the text before the first phrase (whole text if no phrase)
  fire_b       1 if the phrase occurs before the first '####' of the text (v2 'rethink at ####'), else 0
  fire_a       1 if the phrase occurs after the first '####' (rethink after the committed answer), else 0
  out_sha256   SHA-256 of the generated text (to check two runs for identical text without shipping it)

The file keeps the run's own metadata and records source_file / source_sha256 (of the full output it was reduced from).
python scripts/reduce_outputs.py FULL.json OUT.json [--command "python scripts/... "]
"""
import os, re, sys, json, hashlib, argparse

PH_IDS = [80022, 11, 3783, 13, 6771, 752, 75655, 419, 624]     # Qwen3 tokenizer: "Hmm, wait. Let me rethink this.\n"
PH = 'Hmm, wait. Let me rethink this.'
NUM = r'####\s*(-?[\d,]*\.?\d+)'
DROP = {'q', 'out', 'out2', 'ids', 'forced', 'rt_text', 'question', 'text'}


def correct(pred, gold):
    try: return pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


def find_all(ids, pat):
    return [k for k in range(len(ids) - len(pat) + 1) if ids[k:k + len(pat)] == pat]


def reduce_row(r, tok=None):
    o = {k: v for k, v in r.items() if k not in DROP}
    out = r.get('out')
    ids = r.get('ids')
    if ids is not None:
        o.setdefault('n_tok', len(ids))
        hits = find_all(ids, PH_IDS); o['ph_hits'] = hits
        if tok is not None and 'first_ok' not in r:
            t1 = tok.decode(ids[:hits[0]]) if hits else out
            m = re.search(NUM, t1 or ''); o['first_ok'] = bool(correct(m.group(1).replace(',', '') if m else None, r['gold']))
    if out is not None:
        m = re.findall(NUM, out); o['ok_last'] = bool(correct(m[-1].replace(',', '') if m else None, r['gold']))
        k = out.find(PH); h = out.find('####')
        o['fire_b'], o['fire_a'] = ((0, 0) if k < 0 else ((1, 0) if (h < 0 or k < h) else (0, 1)))
        o['out_sha256'] = hashlib.sha256(out.encode()).hexdigest()
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--command', default='')
    ap.add_argument('--no_tokenizer', action='store_true')
    a = ap.parse_args()
    raw = open(a.src, 'rb').read(); d = json.loads(raw)
    tok = None
    if not a.no_tokenizer and any('ids' in r for r in d.get('rows', [])[:1]):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')      # same tokenizer for all Qwen3 sizes
    meta = {k: v for k, v in d.items() if k != 'rows'}
    meta['source_file'] = os.path.basename(a.src); meta['source_sha256'] = hashlib.sha256(raw).hexdigest()
    if a.command: meta['command'] = a.command
    meta['reduced_by'] = 'scripts/reduce_outputs.py (generated text, token ids and question text removed)'
    meta['rows'] = [reduce_row(r, tok) for r in d['rows']]
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    with open(a.dst, 'w') as f: json.dump(meta, f, separators=(',', ':'))
    print(a.src, '->', a.dst, f'{len(raw) / 1e6:.2f} MB -> {os.path.getsize(a.dst) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
