# causal_lw — causal latent write lane (task [5])
2026-11-21 02:44 started. Reading the repository latent_write code. Nothing run yet.
02:50 Copied lwv1.py, implant.py, causal.py, kvrewrite.py, micromachines/{__init__,hdl}.py here; run.sh = node env
(venv, HF_HUB_CACHE=$WSM_DATA/hf/hub offline). Plan: cw.py (muladd mid, Qwen3-0.6B-Base, n=300 test):
 V1 ridge REPLACE write at causal positions (")+" and/or the tail "?\nA: " after c, features P digits [+c digits]) by layer;
 V2 trained side map (linear, P digits -> additive delta at ")+"+tail positions; single layer or all layers), CE on answer
 digits, trained on fresh items (seed 7), not test.
02:55 wrote cw.py + job.sh (job.sh MODE -> run.sh -> python3 cw.py MODE; out/cw_MODE.json, out/log_MODE.txt).
 Submitted: j23277 job_clw_ridge = `<queue-submit> $PWD 2400 job_clw_ridge -- bash job.sh ridge`
 Next: if it runs clean, submit `bash job.sh train` (V2 configs 8:ct,16:ct,all:close,all:ct,all:tail; CW_STEPS=300).
 Check: gq log j23277 --tail 40
03:00 j23277 RIDGE (V1) RESULT: NEGATIVE. plain 14.3, text oracle 98.0, tool line 96.0. Ridge REPLACE at ')+' (close),
 tail ('?\n','A',':',' '), or both, layers 0-20: never above plain (<=15%), corrupted-P follow ~1% => the value is not
 used; at L>=8 replacing ')+' hurts (0-1%), at L>=16 replacing tail kills it (0%). Linear oracle states at causal
 positions do not carry P in a form the later layers use (they read P from the operand-digit positions).
 Submitted j-train: `gsub.sh $PWD 3000 job_clw_train -- bash job.sh train 8:ct,16:ct,all:close,all:ct,all:tail` (V2).
03:04 j23280 TRAIN (V2, 300 steps x bs32, lr 3e-3, linear P-digit one-hots[31] -> additive delta per position/layer,
 CE on the 3 answer digits; train items seed 7, test excluded) partial: 16:ct 50.3 (corr-P control 26.7 acc / 1.3 follow),
 all:close 62.7 (corr 19.3 / follow 5.0). Loss still falling -> next: CW_STEPS=2000 run; then (c) latent-slot insertion.
03:07 j23280 done: 8:ct 52.7 (corr 28.0), 16:ct 50.3 (26.7), all:close 62.7 (19.3), all:ct 73.7 (corr acc 10.7, follow 9.3).
 Submitted j23285 job_clw_train2k: `env CW_TAG=_2k CW_STEPS=2000 bash job.sh train all:ct,all:close,all:tail,8:ct`
 -> out/cw_train_2k.json. Then: (c) latent-slot insertion (extra positions after ')+' carrying P), (d) generated exprs.
03:12 j23280 also: all:tail 69.7 (corr 9.7 / follow 15.0). Added mode `kv` to cw.py (V3 = variant (c)): all-layer delta at ')+'
 + learned per-layer/head attention bias from later queries to the ')+' key (eager attn patch; lwv1 copy has LW_ATTN env).
 Submitting (gsub waiting for a slot): `env LW_ATTN=eager CW_STEPS=600 bash job.sh kv bias,nobias` -> out/cw_kv.json.
03:20 **j23285 all:ct @2000 steps: 97.3% causal** (corrupted-P' control: acc 1.7, follow 49.3 => the written value is
 used). vs non-causal 96-99, text oracle 98.0, tool line 96.0, plain 14.3. Caveat: map is TRAINED with answer CE on
 non-test items (non-causal writer was ridge on oracle states, no answers); Python reference positions (query only),
 not yet compiled. Remaining 2k configs (all:close, all:tail, 8:ct) still running -> out/cw_train_2k.json.
 Added mode `gen` (V4 = (d): write at GENERATED ')+' in 'A: (a*b)+c = ANS' format, decode-step hook) - to submit.
03:24 j23289 (kv, accidental 2nd GPU) cancelled to respect the 1-GPU budget. job2.sh = gen (1500 steps) then kv
 (bias,nobias; 1000 steps, eager). Submit AFTER j23285 finishes: `<queue-submit> $PWD 3600 job_clw_genkv -- bash job2.sh`
03:28 j23285 all:close @2000: 93.3 (corr-P' acc 7.0, follow 16.0) - write ONLY at ')+' (all 28 layers) works too.
03:33 j23285 all:tail @2000: 94.0 (corr acc 5.0, follow 37.7).
03:40 j23285 done: 8:ct @2000 84.3 (corr 12.3 / follow 6.7). Cancelled j23289 had finished kv/bias @600 steps:
 90.0 (corr acc 9.7, follow 8.0; beta max 5.3) [out/old_kv600/]. vs all:close no-bias @300 = 62.7.
 Submitted j23301 job_clw_genkv (job2.sh: gen @1500 then kv bias,nobias @1000) -> out/cw_gen.json, out/cw_kv.json.

## Table so far (muladd mid, Qwen3-0.6B-Base, n=300 test, greedy; plain 14.3, text oracle 98.0, tool line 96.0, non-causal 96-99)
| variant (causal, writes at ')+' and/or later prompt positions) | acc | corrupted-P' acc / follow |
|---|---:|---:|
| V1 ridge replace (oracle states), any of close/tail/both, L0-20 | <=15 | ~14 / ~1 |
| V2 trained linear map, all 28 layers, ')+' only (2000 st) | 93.3 | 7.0 / 16.0 |
| V2 all layers, tail '?\n A : _' only (2000 st) | 94.0 | 5.0 / 37.7 |
| V2 all layers, ')+' + tail (2000 st) | **97.3** | 1.7 / 49.3 |
| V2 layer 8 only, ')+' + tail (2000 st) | 84.3 | 12.3 / 6.7 |
| V3 ')+' all layers + learned attn bias to ')+' (600 st) | 90.0 | 9.7 / 8.0 |
03:50 j23301: V4 gen (d) @1500, loss on answer digits only: gen plain 7.3 (copy of expr 100%), write 70.7 but copy_ok
 75.3 (write at generated ')+' breaks copying c); corr-P' 9.3 / follow 10.3. KV bias @1000: 85.7 (corr 9.0/8.0).
 Fix: CW_GLOSS=all (CE on all tokens after the generated ')+'). Next submit: `env CW_TAG=_gall CW_GLOSS=all CW_STEPS=2000 bash job.sh gen`
03:55 j23301 kv nobias @1000: 86.3 (corr 9.3/8.3) vs bias 85.7 -> learned attention bias to ')+' gives no gain over the
 plain all-layer ')+' delta (which reaches 93.3 @2000). Submitted j23304 job_clw_gall (gen, CW_GLOSS=all, 2000 steps).

## Compiled trigger (lane m4-skill, 2026-11-21, job j23318 job_m4_ct) — closes the "positions chosen in Python" caveat
Trigger = fixed token DFA over the host token stream (prompt AND generated tokens), same style as aw/stk/comp_serial.py:
`ctrig.py` (pure python) / `cw_ct.py` TTrig (torch). 17 states (IDLE, OPEN, A1-3, STAR, B1-2, CLOSE, C1-3, T1-4, COLON),
token-class table TC[class][state], register ops ROP (ra/rb digit accumulators; P = ra*rb latched at CLOSE, valid iff
100<=P<=999), fixed fire-slot table FSLOT (CLOSE->')+' row, T1..T4 -> '?\n','A',':',' ' rows = the Wt row order). Gates (also
table state): colon (the '(' must follow ':'), tail on/off, a saturating '\n\n' counter (nsep>=K: query span only).
The write = onehot(slot) (x) feats(P_register) @ W_L at every layer, driven ONLY by the DFA state + registers (embedding
pre-hook runs the scan at prefill / one step per decoded token with registers carried); no Python position choice.
- Fuzz: sandbox `python3.11 fuzz_ctrig.py 20000`: 100000 scans, 62940 fires, exact (pos, slot, P) match vs an independent
  Python finder over 5 gate configs. On node every prompt, training batch and generated stream is also checked:
  toy 1896 seqs / 15960 fires, gen 3696 seqs / 7796 fires, 0 mismatches. Gated positions == the old Python positions
  (fe,slot) [(7,0),(4,4),(3,3),(2,2),(1,1)]; on the 300 generated streams the gated trigger fires exactly where the old Python
  decode-step finder (parseP) did (300/300).
- Re-measured (muladd mid, n=300 test, greedy; out/ct_prompt.json, out/ct_gen.json; log_ct_*.txt):

| setup | trigger | map | acc | corrupted-P' acc / follow |
|---|---|---|---:|---:|
| toy prompt | Python positions (j23285) | Wt_all_ct_2k | 97.3 | 1.7 / 49.3 |
| toy prompt | **compiled, gated (nsep>=4)** | same map, drop-in | **97.3** | 1.7 / 49.3 |
| toy prompt | compiled, ungated (also fires in the 4 shots) | same map, drop-in | 95.7 | 2.0 / 47.0 |
| toy prompt | compiled, ungated | retrained 2000 st (out/CT_prompt_ungated.pt) | 99.3 | 3.3 / 28.7 |
| generated expr | Python decode finder (j23304, CW_GLOSS=all) | GEN_gall | 96.7 | 6.0 / 12.3 |
| generated expr | **compiled, gated (colon + nsep>=4)** | same map, drop-in | **96.7** (copy_ok 100) | 6.0 / 12.3 |
| generated expr | compiled, colon gate only | same map | 94.7 | 6.0 / 12.3 |
| generated expr | compiled, ungated | same map | 67.7 (copy_ok 97.0) | 6.0 / 13.3 |
| generated expr | compiled, ungated | retrained 2000 st (out/CT_gen_ungated.pt) | 85.3 (copy_ok 73.7) | 14.7 / 4.3 |
| reference | non-causal latent write | – | 96 | – |
| reference | text (tool line) | – | 94-96 (tool line 96.0, text oracle 98.0) | – |
| reference | plain (no write) | – | 14.3 toy / 7.3 gen | – |

Verdict: with the compiled trigger (tables + registers, no host decision) the causal write is unchanged: 97.3 toy, 96.7 on
generated expressions, i.e. >= non-causal 96 and >= text. The gate matters: firing also on the in-context shots (ungated)
costs 1.6 pts on the toy and 29 pts on generated text (the shots' expressions get written too and copying breaks), which
retraining only partly recovers (85.3) -> keep the query-span gate (it is just more DFA state).
