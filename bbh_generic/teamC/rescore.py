"""Re-score saved runs. Lenient extraction for the out-of-style sets (own prompts; the host often ends with 'So, X gets $395'
instead of 'So the answer is 395'): if no 'the answer is', take the last number (num_*) / last True|False (bool_*).
Applied identically to plain and tool runs.   python rescore.py out/a.json out/b.json ..."""
import json, re, sys
def lenient(row):
    t = row['out'].split('\n\nQ:')[0]; t = t.replace('<|endoftext|>', '')
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', t)
    p = m[-1] if m else ''
    if row['task'].startswith('num'):
        nums = re.findall(r'-?\d+', (p or t).replace(',', '').replace('$', ''))
        return nums[-1] if nums else '' if not p else (re.findall(r'-?\d+', t.replace(',', '')) or [''])[-1]
    b = re.findall(r'True|False', p or t); return b[-1] if b else ''
for f in sys.argv[1:]:
    d = json.load(open(f)); rows = d['rows']
    strict = sum(r['ok'] for r in rows); len_ok = sum(lenient(r).lower() == r['gold'].lower() for r in rows)
    print(f'{f:45s} strict {100 * strict / len(rows):5.1f}  lenient {100 * len_ok / len(rows):5.1f}  (n={len(rows)})')
