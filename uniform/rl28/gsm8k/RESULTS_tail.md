## What was refit for 1.7B (circuits unchanged, nothing new climbed)
- Host: Qwen/Qwen3-1.7B-Base (hidden 2048, FF 6144, 28 layers, 16 q / 8 kv heads, head_dim 128, tied embeddings; tokenizer identical to 0.6B).
  Code: one shape line in wiring/fastqwen.py made per-host (env HOST_MODEL), model literals -> HOST_MODEL; patched copies only (patch17.py).
- Readers (calculator / chain2 reader tables, tk symbol tables): rebuilt closed-form from the 1.7B embedding at attach time.
- Writers (implant make_out_dirs, comp_serial and comp_sanity _solve_dirs): re-solved on the 1.7B final norm + LM head at attach time,
  unit-rms margin target SW_DMARG=40.
- End-of-number veto: existing results/gsm_bigger/veto_q17.pt (1.7B-Base, same solve_veto.py, GSM8K train rows 2000-2299).
- Reserved stop row (rung 2): stopcal_hook hn_mineig on 1.7B-Base, dev600 rung-2 token rows (train split): s = 27.9, forced margin
  min 39.4, normal states top - z_c >= 16.5. Guard on degenerate pad rows relaxed via SW_PAD_NONDEG_OK=1 (1.7B pad rows are distinct
  small vectors, max norm 0.375; the original row c = 151669 has z_c - lse <= -19.2 on 669k positions, i.e. never emitted).
- Verifier probe L18 (rung-2 vote, AW_ADAPT, CW reward vote): StandardScaler + L2 logistic regression C = 0.001 (torch float64,
  exact optimum) on 11,992 1.7B-Base samples of GSM8K train split C (T = 1, K = 8, calc v2 on); within-question AUROC 0.86.
- dfa_tables.pt (tokenizer only): reused.

## Checks
- Idle bit-identity (greedy dev600, host vs calc_s4 + chain2 attached): 73/73 no-implant rows token-identical in full, 527/527 rows
  identical up to their first implant step, 0 earlier divergences.
- Calc-v2 stack silence (stack_hook, greedy dev600): stock engine == stack off 600/600; stack on: silent rows identical 595/595.
- Aw fuzz (test_aw) PASS before every rung-2 eval.
- Forced-token / write audit, rung 2: 0 sampled != written (row 3: 151,900 test + 206,955 Hard writer steps; row 4: see NOTES.md).
- RL engine: RL_FASTCHECK greedy old-vs-fast 600/600 token-identical (max |dlogp| 0), first-update TF ratio ~1 (P17: 0.99995 / 1.0003).
- RL packing: 2 arms per GPU, mem_frac 0.45-0.49, B_max 96 (rollout concurrency only), CW micro-batch budget
  FT_BUDGET 16384 (gradient accumulation only); recipe unchanged.

Files: items/{test,hard,gsmplus}.items.jsonl.gz (reduced per-item rows, read via uniform/items.py), res/audit.txt (FE audit lines), rows.json (rl24 uniform.json schema), NOTES.md (log).
