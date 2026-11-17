# Topic fence implant (weight-space micromachine) — NOTEBOOK

Ran on rented GPUs (A40 / H100), one GPU per process.
Host: Qwen/Qwen3-1.7B chat, enable_thinking=False, frozen. Two fences tested separately: math ("never solve math") and ml ("never help build/train/debug ML models").

## Design (2026-11-17)
- Gate = logistic probe on hidden_states[L] at the last prompt token (ReLU-threshold unit in weights). Selected by
  leave-one-STYLE-out AUROC on dev data only (pos vs neg+hard), with leave-one-source-out (gen7b <-> public) reported.
- Actions: text (solved output directions before the final norm write the decline phrase, then the host continues; the earlier
  refusal writer), latent (fixed decline vector added after layer L2 > L when gated), generation-time monitor (per-token probe
  on the host's own reply states, k=3 consecutive hits -> writes "\n\n<decline phrase><|im_end|>").
- Baselines: plain host, system prompt (as specified), strong system prompt (gives the exact decline sentence), keyword regex gate
  + the same text writer.
- Data. DEV (all selection): Qwen2.5-7B-Instruct writes user messages from (fence, label, topic, style) specs (specs.py, 10 styles,
  8 languages), two sampling runs (run a had a per-request seed bug -> duplicates, run b unseeded), deduped -> 9275 rows incl.
  GSM8K train 200 and Alpaca train 1500. Every row labelled for both fences by the 7B (question-level YES/NO logit gap); generator
  mistakes filtered by |gap|>0 (pos) / <0 (neg, hard).
  TEST (never scored during development): 640 prompts hand-written by a separate Claude agent (independent author; 10 styles;
  pos 120/fence 12 per style, neg 70, hard 70, drift 30 per fence, general 60) + GSM8K test 100 + MATH-500 100 + MMLU math MC 100
  (math pos) + Alpaca test 600 (labelled per fence by the 7B: pos if gap>2, retention if both gaps<-2) + MMLU non-STEM 400 for
  capability retention.

## Log
- Dev data: run a 4348 rows (half of the 5130 generations lost as exact duplicates: SamplingParams(seed=0) applies the same seed to
  every request), run b unseeded; merged/deduped dev_all = 9275. Generator noise: some "user messages" are written in assistant
  voice (e.g. a casual answer about why the sky is blue); kept, they are still on-topic/off-topic text.
- Math probe sweep (dev, Qwen3-1.7B): every layer 4-24 has LOSO-style AUROC >= 0.999 (pos vs neg+hard) and >= 0.999 vs hard only.
  Best L16 lam 0.01: LOSO 0.9997, worst held-out style 0.9967, train-on-public-only (GSM8K+Alpaca) -> gen7b 0.9993 (vs hard 0.9988).
  Threshold (OOF): TPR 99.9%, FPR neg 1.0%, FPR hard 1.0%; per held-out style FPR max = alpaca 2.5% (Alpaca has borderline items).
  Math is close to lexically separable in-distribution; the independent test set is the real check.
- System prompt is nearly ignored by Qwen3-1.7B: 3% of 200 dev math positives declined (string check) under the "must not solve math"
  system prompt; spot check 12/12 math+ML replies fully comply. Added a stronger variant (gives the exact decline sentence).
- Latent sweep v1 (string judge, gate forced on, 64 dev math positives, vector added at the last prompt token and every generated
  token): generic refusal direction v_ref (AdvBench - Alpaca) at L16-20 x1 gives 23-62% "declines", but they are CONFABULATED
  HARM refusals ("I cannot provide any content that involves unethical, illegal..."); x2+ degenerates ("**** ** **"). Prefill
  topic direction v_pf (teacher-forced decline opener - compliance opener) L10 x2: 67% but "The determinant of a matrix is not be used
  to spread harm". The host's decline concept is entangled with harm. v_sys (system-prompt delta) dropped: too few declines.
- Math monitor (dev): reply-level LOSO AUROC >= 0.9993 at every layer 6-26; chosen L18, thr 0.475, k=3. OOF catch 99.8% at 2.0% FPR
  on not-math replies. Host does math in 87% of replies to dev math positives, 0% for non-positives, 69% for drift prompts
  (budget/recipe/loan). Monitor fires on 98% of drift replies that do math, and on 74% of those the labeller called not-math (most
  are cut-off replies that are about to compute, e.g. "Calcoliamo quanto potresti risparmiare...").
- JUDGE VALIDATION (40 dev items, 10 each plain / strong-sysprompt / text-writer / latent, hand-labelled by me reading full replies):
  LLM judge (Qwen2.5-7B, "does the reply actually help with what was asked... even after an apology") 38/40 agree; string check 23/40.
  Disagreements: a plain reply with a fabricated price table (judge said not-helping), a latent rant (judge said helping).
- KEY: the text writer + continue does NOT block benign topics. After the forced "I can't help with math problems." the host
  continues "But I can explain how to do it step by step." and solves it (8/9 text-writer positives in the validation set). Unlike
  harmful requests (earlier refusal project: forced opener was decisive), here the host has no reason to stay declined.
  => added text_stop (phrase + <|im_end|> written by the same solved directions) and kw_stop.
- Latent v_ref x1 at L18 (validation set): every fired reply is a confabulated legal/ethics rant ("I'm not a criminal...", "You are
  not allowed to access the internal state of the neural network"), sometimes after restating the problem. It does block, but
  it is not a decline about the topic and is not coherent with the request.
- ML probe sweep (dev): best L10 lam 1e-4, LOSO 0.9990 (hard 0.9998), worst style 0.990; mid layers 14-18 have a weak held-out
  style (worst 0.92). Thr (OOF): TPR 99.8%, FPR neg 1.0%, hard 3.1%; held-out-style FPR highest on long 4.6% and snippet 4.3%.
- ML monitor (dev): reply-level LOSO AUROC 0.998-0.9995 all layers; chosen L14 thr 0.85 k=3: OOF catch 99.8% at 2.0% FPR.
  Drift: host gives hands-on ML help in only 11% of 255 ML-drift replies (it answers "detect spam" with keyword rules, "auto-tag
  photos" with CMS metadata); monitor catches 39% of those 28, fires on 8% of the rest. ML drift is rare at 120 tokens.
- Latent sweep v2 (s3b, LLM judge + coherence judge, 64 dev math positives, gate forced on; grid v_ref/v_pf x L2 14-20 x scale
  1/1.5 x active steps 1/6/all). Writing only at the prompt token (ns=1) does nothing (<=20%). All steps: declines 70-100% but
  coherence drops to 25-75% (harm rants, "I'm not an image processor", garbage). Best decline-and-coherent = v_ref (AdvBench-Alpaca
  diff of means) at L16, x1.5, first 6 positions: 80% decline, 100% coherent, e.g. "I'm sorry, but I cannot assist with this
  request. The question involves calculating the determinant of a 2x2 matrix... it may be considered inappropriate". Other
  "declines" in that config are deflections ("this code is incorrect", "the question is ambiguous").
  The latent write must be strictly after the gate read; math probe L16 -> refit math probe at L12 for the latent gate
  (LOSO 0.9996, OOF TPR 99.8%, FPR neg 1.0%, hard 1.9%).
- ML latent sweep (s3b): best decline-and-coherent = v_ref L18 x1 all steps: 97% decline, 86% "coherent" by the judge, but the
  declines are fluent legality/ethics rants ("You are not engaging in any form of ethical hacking... I am not a criminal").
  Chosen by the pre-set criterion anyway (gate = ML probe L10 < 18). v_ref L16 x1.5 first-6 (math winner) gives only 41% on ML.
- Compute: shared GPUs; two runs went out of memory and were rerun. The final 1.7B eval ran math and ml fences in separate
  processes / files (res/test_q17_gen_{math,ml}.json; each condition's plain and fenced runs on the same machine). A reduced
  0.6B pipeline (lam 1e-3, 9 layers, no latent) ran separately.
- BUG caught (test run, before any judging): latent fence gate was re-read at every generated token (the latent kind has no
  norm hook, so the step counter never advanced) -> the L10 probe, applied to generated-token states it was never trained on,
  fired on 100% of test rows at some token. Fixed (gate read only on the prompt pass via the latent step counter); both
  probe_latent test runs discarded and rerun. Other conditions unaffected (text/monitor kinds advance the counter).
- Silent checks on the test set (first 640 test prompts, 40 steps, mixed batches, per-step logits vs plain host in the same
  batches): kw 484/484 (math) 505/505 (ml), probe_text 491/491 & 503/503, probe_text_stop same, ML monitor 511/511 non-firing rows
  bit-identical (max |diff| 0.0).
