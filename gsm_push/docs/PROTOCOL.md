# GSM8K push: implant loop past ~71% (no new RL)

Goal: raise GSM8K accuracy of the RL+calc host beyond its 71.2% test score using new implants
(exact circuits or small machines attached to the frozen host), following the blog's agent loop:
run model -> read failures -> find a pattern -> write a generic implant -> attach, test on dev -> keep if it helps.

## Host (frozen, no further RL in this study)
- RL+calc host: `rl_implants_v2/runs/long/a2_u300_s2` (Qwen3-0.6B-Base after 300 GRPO updates with the "=" calculator).
  GSM8K test 71.2 (71.3 in its own eval), GSM-Hard 55.8. Same 8-shot prompt as the failure analysis.
- Secondary check at the end only: base host (u0) + calculator, GSM8K test 53.6.
- Harness to reuse: dig/scripts/fail/ (tw_dump_lean.py, dump_hook.py, classify.py, pass8.py) and the
  sanity engine dig/sanity/san_engine.py (exact decode-time hooks inside the RL engine's CUDA graph).
- Failure taxonomy and numbers: dig/gsm8k_failures.md (READ FIRST).

## Data discipline (anti reward-hacking, as in the blog's loop)
- DEV = GSM8K train items NOT in the RL pool (split A was the RL pool; use the remaining train items, "split B").
  Take a fixed 600-item dev sample (seed 0) and a separate 300-item "dev2" for confirming before declaring a win.
  If split B is not identifiable, use GSM8K train items not in the RL pool file, and document how.
- TEST (GSM8K test, 1319) is NOT used for development. Only the coordinator runs it once, on the final stack.
- Implants must be generic: say what f(x) computes without referring to GSM8K, and where else it would help.
- Report greedy accuracy on dev, fixes vs breaks, fire rate, extra tokens, and the same on GSM-Hard dev slice if relevant.

## Compute
- Machines: 2x H100, 2x A40, 1x A100. Overnight RL replicates finish by ~08:30 CT; after that
  GPUs are free for this study. Before then, use free memory only, with torch.cuda.set_per_process_memory_fraction caps;
  never OOM a running job. Keep >=6 GB disk free per pod. Never run model jobs on the local laptop.
- GPU ownership after 08:30 CT: calculator-v2 agent -> A100; selection/verifier agent -> A40 (both A40s);
  bookkeeping/units/re-read agent -> H100 (both H100s).

## Output
Each agent writes gsm_push/<topic>.md with: hypothesis, implant design, dev results (600 + 300 confirm),
fixes/breaks examples, fire rate, cost, and a recommendation (keep / drop) plus how to stack it.
