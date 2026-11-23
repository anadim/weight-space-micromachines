"""Audit P1-19 (§1.2): "Claude checked that [the calculator's Read layer] works with all of ~151k tokens in Qwen's vocabulary."
Re-runs micromachines.make_reader (the reader the implants use) on the host's full input-embedding matrix and logs:
  (1) the exhaustive check: clamp(E @ W.T + b, 0, 1) == the intended 0/1 flag table, on every row x every symbol;
  (2) pre-activation margins (worst non-target row, worst target row);
  (3) a regex cross-check on the decoded vocabulary: which tokens contain a digit / operator character but are NOT read
      as that symbol (they are not the host's own single-character digit/operator tokens).
PYTHONPATH=$WSM_ROOT python3 checks/sourced/vocabjob/vocab_reader.py MODEL [MODEL ...]
  -> checks/sourced/vocabjob/res/vocab_reader_<model>.json + stdout   (micromachines/ from the release root)"""
import sys, json, os, re, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.machine import make_reader, symbol_token_ids
from micromachines.specs import SPECS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res'); os.makedirs(RES, exist_ok=True)
for model_name in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    E = m.get_input_embeddings().weight.detach().float()
    V = E.shape[0]
    dec = [tok.decode([i]) for i in range(len(tok))]
    rep = dict(model=model_name, vocab_rows=V, tokenizer_len=len(tok), machines={})
    print(f'== {model_name}: embedding rows {V}, tokenizer entries {len(tok)}')
    for name in ('calc_s3', 'calc_s4'):
        syms = SPECS[name]['syms']; ids = symbol_token_ids(tok, name)
        W, b, gaps = make_reader(E, ids, syms)                      # asserts exactness on all V rows
        pre = E @ W.T + b
        got = torch.clamp(pre, 0, 1)
        want = torch.zeros_like(got)
        for i, s in enumerate(syms): want[ids[s], i] = 1
        mism = int((got != want).sum())
        tmask = want.bool()
        worst_non = pre.masked_fill(tmask, float('-inf')).max(0).values
        worst_tgt = pre.masked_fill(~tmask, float('inf')).min(0).values
        per = {s: dict(tokens=[dec[t] for t in (ids[s] if isinstance(ids[s], list) else [ids[s]])],
                       n_tokens=int(tmask[:, i].sum()), max_preact_other=round(worst_non[i].item(), 3),
                       min_preact_target=round(worst_tgt[i].item(), 3), rel_gap=round(gaps[s], 4))
               for i, s in enumerate(syms)}
        rep['machines'][name] = dict(n_syms=len(syms), cells_checked=V * len(syms), mismatches=mism,
                                     rows_flagged_any=int(tmask.any(1).sum()), per_symbol=per)
        print(f'{name}: {len(syms)} symbols x {V} rows = {V*len(syms):,} cells, mismatches {mism}; '
              f'rows read as some symbol {int(tmask.any(1).sum())}; worst non-target pre-act {worst_non.max().item():.3f} (clamps to 0), '
              f'worst target pre-act {worst_tgt.min().item():.3f} (clamps to 1)')
    # regex cross-check (calc_s4 symbol table): tokens with a digit / operator char that are not read as that symbol
    ids = symbol_token_ids(tok, 'calc_s4'); readset = collections.defaultdict(set)
    for s, t in ids.items():
        for x in (t if isinstance(t, list) else [t]): readset[x].add(s)
    xc = {}
    for label, rx in (('ascii digit 0-9', r'[0-9]'), ('any unicode digit (\\d)', r'\d'), ('+ - * / = x ×', r'[+\-*/=×]')):
        hit = [i for i in range(len(tok)) if re.search(rx, dec[i])]
        unread = [i for i in hit if i not in readset]
        xc[label] = dict(tokens_containing=len(hit), read_by_calculator=len(hit) - len(unread), not_read=len(unread),
                         examples_not_read=[dec[i] for i in unread[:40]])
        print(f'  regex {label}: {len(hit)} tokens contain it; {len(hit)-len(unread)} are reader symbols; {len(unread)} are not, e.g. '
              f'{[dec[i] for i in unread[:15]]}')
    single = [i for i in range(len(tok)) if re.fullmatch(r'[0-9]', dec[i])]
    multi = [i for i in range(len(tok)) if re.search(r'[0-9]', dec[i]) and not re.fullmatch(r'\s?[0-9]', dec[i])]
    rep['regex'] = xc
    rep['ascii_single_digit_tokens'] = [dec[i] for i in single]
    rep['tokens_with_ascii_digit_and_other_chars'] = len(multi)
    print(f'  ASCII single-digit tokens: {len(single)}; tokens mixing an ASCII digit with other chars: {len(multi)} e.g. {[dec[i] for i in multi[:15]]}')
    fn = RES + '/vocab_reader_' + model_name.rstrip('/').split('/')[-1] + '.json'
    json.dump(rep, open(fn, 'w'), indent=1, ensure_ascii=False)
    print('wrote', fn)
