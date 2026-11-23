"""Reduce the per-item off-task rows written by ot_hook.py (OT_OUT/<ROW>_<task>.jsonl: log-likelihoods, NQ answer text)
to the per-item fields ana_ot.py needs (no log-likelihoods, no generated text):
    python3 mk_items.py SRC_DIR DST_DIR [ROW ...]
  SRC_DIR  ot_hook output dir (OT_OUT, e.g. $WSM_WORK/offtask/j06/ot)
  DST_DIR  e.g. j06/res; writes one file per row (all five tasks, ~30k items, 3-4 MB of text, so gzip'd):
           DST_DIR/<ROW>.items.jsonl.gz
Fields (h = host only, a = implants attached, both from the same forward pass); every record has task (mmlu | arcc |
hs | wg | nq), in benchmark-file order:
  multiple choice (mmlu, arcc, hs, wg): i (item index in the benchmark file; mmlu: subject instead), gold, pred_h / pred_a
  (argmax option by summed log-lik), predn_h / predn_a (argmax by log-lik per character; arcc, hs), ok_h / ok_a /
  okn_h / okn_a, fsc (implant writes on scored positions), fall (writes anywhere in prompt + continuation), npos (scored
  positions), end (items / options whose serial END trigger fires on a scored position);
  nq: i, ok_h, ok_a (EM), fsc (implant fires while the attached run generates), fh (same, host-only run), end,
  same (1 = host-only and attached generations are byte-identical)."""
import glob, gzip, json, os, sys

src, dst = sys.argv[1], sys.argv[2]
rows = set(sys.argv[3:])
os.makedirs(dst, exist_ok=True)


def am(v):
    return max(range(len(v)), key=lambda q: v[q])


TASKS = ('mmlu', 'arcc', 'hs', 'wg', 'nq')
files = {}
for f in sorted(glob.glob(f'{src}/*.jsonl')):
    tag, task = os.path.basename(f)[:-6].rsplit('_', 1)
    if task in TASKS and (not rows or tag in rows): files.setdefault(tag, {})[task] = f
for tag, fs in sorted(files.items()):
  out = []
  for task in TASKS:
    if task not in fs: continue
    f = fs[task]
    for k, line in enumerate(open(f)):
        x = json.loads(line)
        if task == 'nq':
            y = dict(task=task, i=k, ok_h=x['ok_h'], ok_a=x['ok_a'], fsc=x['fsc'], fh=x['fh'], end=x['end'], same=x['same'])
        else:
            y = dict(task=task, i=k, subject=x['subject']) if 'subject' in x else dict(task=task, i=x.get('i', k))
            y.update(gold=x['gold'], pred_h=am(x['lp_h']), pred_a=am(x['lp_a']))
            if 'nch' in x:
                y['predn_h'] = am([l / n for l, n in zip(x['lp_h'], x['nch'])])
                y['predn_a'] = am([l / n for l, n in zip(x['lp_a'], x['nch'])])
            for kk in ('ok_h', 'ok_a', 'okn_h', 'okn_a', 'fsc', 'fall', 'npos', 'end'):
                if kk in x: y[kk] = x[kk]
        out.append(json.dumps(y, separators=(',', ':')))
  p = f'{dst}/{tag}.items.jsonl.gz'
  with open(p, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as fh:
      fh.write(('\n'.join(out) + '\n').encode())
  print(p, len(out), os.path.getsize(p))
