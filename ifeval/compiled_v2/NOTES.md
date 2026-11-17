# compiled_v2: IFEval rule-keepers, compiled-only run (2026-11-17)

Scope (deadline cut by the author): only no_comma, lower (upper veto x2), quote (open + close).
repeat, forbid, two, paras, nth were DROPPED (not run at all).

Code: gen_c.py (harness + circuit), uses compiled.py (reader + ReLU stacks, unchanged) and dirs.pt (unchanged write vectors).
No detect.py, no rules.py, no regex, no Python rule objects, every odd prompt goes through the circuit (no subset).

Circuit per token (tensor ops only):
- reader: linear readers on the host token embedding (13 groups) + 0/1 lookup rows (12 groups) -> 25 flags (exact on all 151,936 rows)
- stack1: ReLU net over a 16-token window -> matches (no_comma, lower, caps, quote), role-header flags, first-answer marker
- latch heads (max of matches so far), turn head (role at most recent header) -> in-assistant flag
- quote-close tracker (tensor cache): answer index n, last non-whitespace answer token ends with '"' (lookup tables over the vocab), first-token exception
- gates: nc=AND(latch,asst), lo=AND(latch,asst), q=AND(latch,asst); open=AND(q,first); close=ReLU(q-first-ReLU(lq-lx))
- priority as tensor gates: W0 = open*F + (1-open)*(nc*Vcomma + lo*2*Vupper); s = (1-open)*close
- end-of-turn readout (DISCLOSED: second readout of rmsnorm(h+W0) through the frozen LM head): e = clamp(1e6*(max EOS logit - max other), 0, 1);
  W = (1-s*e)*W0 + s*e*F
Remaining non-weight pieces: the harness's greedy decode loop (same as baseline); a Python on/off switch that turns the hook on for
generation (not a rule decision); the second LM-head readout (disclosed, fixed linear readout of the host state).

Held-out run: one GPU, odd keys split into 4 shards; compiled, baseline (gen.py --rules none) and the
Python reference (gen.py --rules no_comma,lower,quote --vscale upper=2 --subset 0) use the SAME shards and batches.

## HELD-OUT RESULT (odd keys, 299 prompts, Qwen3-0.6B, one GPU, 4 shards, bs16, greedy, max_new 2048)
Same shards and batches for all three runs (runs/odd_none.jsonl = gen.py --rules none; runs/odd_pyref.jsonl = Python reference
3 rules; runs/odd_comp.jsonl = gen_c.py compiled-only).
- P-strict / P-loose / I-strict / I-loose: baseline 56.52-56.86 / 61.54-61.87 / 67.99-68.21 / 71.30-71.52;
  compiled 61.54-61.87 / 66.22-66.56 / 72.19-72.41 / 75.50-75.72. (Ranges: the lm-eval scorer is not deterministic on key 1129,
  language check; the same text scores differently between calls.)
- prompt-strict vs baseline: +18 / -3 (lost 279, 1021, 1781); net +15 prompts. Key 1129 is a silent prompt with identical text in both runs
  whose score flips between scorer calls (excluded from gained).
- per rule strict passes: no_comma 18 -> 34/36; lowercase 18 -> 21/23; quotation 21 -> 22/23.
- silent prompts (circuit never fired): 222/222 byte-identical to the baseline. Active 77 prompts; truncated 6 (baseline 3 on the same prompts);
  all prompts truncated 18 vs 15.
- compiled vs Python reference: 299/299 responses byte-identical (77/77 active).
- circuit detection vs regex (odd): no_comma 33/33 exact; lower 24 vs 25 (misses 1571); quote 21 vs 22 (misses 1325); no circuit-only fires.
  The 2 misses did not change any output (the reference's write there gave the same text).
