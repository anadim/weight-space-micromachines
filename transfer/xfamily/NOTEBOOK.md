# Weight-Space Micromachines — cross-FAMILY transfer (xfamily)

Question: do our implants (reader tables + ReLU/attention circuits + solved write directions) transfer to small models
from OTHER families than Qwen? All prior results are on Qwen (tokenizer splits numbers into single digits).

Started 2026-11-17. Pods: A40 pod (2xA40, ~/xfamily), H100 pod GPU0 (<workdir>/xfamily). Never ran on the laptop.
Code reused from <repo> (release-prep): scripts/run_gsm.py, scripts/solve_veto.py,
micromachines/{machine,specs,calcfmt,circuits}. Harness = same 8-shot plain-CoT GSM prompt as the Qwen runs.

## Models chosen (all base, non-Qwen)
| model | family | params | released | tokenizer: "12345" | calc symbols | verdict |
|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-1.7B | HuggingFace | 1.7B | Nov 2024 | [3,3,3,3,3] single digit | space-attached ops single-token (like Qwen) | CLEAN transfer |
| ibm-granite/granite-3.3-2b-base | IBM | 2.0B | 2025 | single digit | space-attached ops single-token | CLEAN transfer |
| tiiuae/Falcon3-3B-Base | TII | 3.1B | Dec 2024 | single digit | **space is a SEPARATE token** from the operator | needs reader rebuilt (calc_s3f) |
| LiquidAI/LFM2-1.2B | Liquid AI | 1.2B | 2025 | [10293,2637] MULTI-digit chunk | n/a | calculator BREAKS (digits not single tokens) |

Rejected / not available:
- meta-llama/Llama-3.2-1B, google/gemma-3-1b-pt, gemma-2-2b: **gated**, no HF access token on the pods.
- allenai/tmax-4b: architecture is `Qwen3_5ForConditionalGeneration` (a Qwen3.5 derivative) -> NOT a different family. Dropped.
- SmolLM3-3B-Base, allenai/OLMo-2-0425-1B, microsoft/Phi-4-mini, openbmb/MiniCPM4-0.5B: all MULTI-digit chunk tokenizers
  (same break mode as LFM2). Kept LFM2 as the single documented chunk-tokenizer negative (explicitly suggested in the brief).

## Why the tokenizer decides everything (calculator)
The calculator reader is a 0/1 lookup table over the vocabulary: each symbol (each digit 0-9, + - * / . = space newline) must be
ONE token, and the host must EMIT numbers one digit per token, so the circuit can read the operands digit-by-digit and the write
directions can emit the answer one digit per token. Two independent tokenizer axes matter:
  1. single-digit vs multi-digit chunks. Multi-digit (LFM2, SmolLM3, OLMo-2, Phi, MiniCPM) => the reader cannot see or write
     individual digits. Breaks; no generic single-digit extension exists without re-deriving the whole number circuit per vocab.
  2. how a space attaches to an operator. Qwen/SmolLM2/granite encode " +" " =" " /" as ONE token (space glued to operator), which
     is what calc_s4's spaced build assumes. Falcon3 is single-digit but encodes the space as a SEPARATE token, so the window
     geometry differs; the space-prefixed symbol alternatives must be dropped and the space read by the SP symbol (spec calc_s3f).

## What was rebuilt per model (vs unchanged)
- UNCHANGED: every ReLU/attention circuit (micromachines/circuits/*.py) — tokenizer-independent by construction.
- REBUILT (automatic, closed-form in machine.make_reader): the reader W,b from the new embedding matrix; checked EXACT on the
  whole vocabulary by the assert in make_reader (each symbol flag is 1 exactly on its token ids, 0 elsewhere). Build fails loudly
  if a symbol is not a single token or not linearly separable.
- REBUILT (Adam on the host's own norm+LM head, no task data): the write directions (make_out_dirs) and the end-of-number veto
  (solve_veto.py, fit on GSM8K TRAIN rows 2000-2299 only — training data, never test).
- SPEC EDIT: calc_s3x = calc_s3 with CSN trimmed to ('\n',' \n') for SmolLM2/granite (the combined '.\n'/':\n'/',\n' are two
  tokens there; harmless, a period before a newline is its own token). calc_s3f for Falcon3 (drop space-prefixed op alternatives).

## Harness
- GSM: scripts/run_gsm.py, 8 worked GSM8K-train solutions as the prompt (plain CoT), greedy, bs 32, full test sets (1319 each).
  plain vs calc (calc = the calculator implant reads the model's own "a op b =" and writes the exact result, ending with the veto).
- veto solved per host: 100% "no digit wins" at number endings, ~99% the model's own non-digit choice kept.

## HARNESS CHANGE (coordinator audit, 2026-11-17 ~04:00) — all earlier calc_s3 runs discarded
Coordinator: Qwen 1.7B/4B calc runs used old calc_s3 (digit runaways); blank-line stop truncated outputs. New harness
code/run_gsm_x.py (repo scripts/run_gsm_x.py on pods):
- calculator = calc_s4 (BBH calculator; model writes the space after '=' itself; veto ends the number), attached zero_os=True.
- generic per-vocab sanitizer: each symbol keeps only its single-token alternatives; a symbol with none -> CANNOT_ATTACH (exit 3).
  SmolLM2/granite: only CSN trimmed to ('\n',' \n'). Falcon3: + - * / CE CS OS SP CSN lose their space-prefixed forms
  (same as the hand-made calc_s3f, now produced by the rule). LFM2: CANNOT_ATTACH on N* ('*-').
- stop: '\n\nQuestion:' / eos / 400 tokens / first complete '#### <number>' line (score = first '####'; checked equivalent:
  16/16 identical preds vs running to 400 tokens, SmolLM2 GSM-Hard batch 1, mean tokens 368 -> 133).
- logged: unextracted (no '####') rate per arm; GSM-Hard accuracy on test items 200-1318 separately (shots chosen on 0-199).
- calc_s4 attach margins: SmolLM2 14.25, granite 16.0, Falcon3 11.6 (each write wins the host argmax).
Partial calc_s3 / blank-line results (none had completed) moved to old_calc_s3/ and are not used.

## Results (filled as runs report) — plain -> calc
(Qwen reference, same harness: GSM8K 52->53 / 73->72 / 84->85 ; GSM-Hard 27->35 / 45->51 / 59->60 for 0.6B/1.7B/4B.)

| model | GSM8K plain | GSM8K calc | GSM-Hard plain | GSM-Hard calc |
|---|---|---|---|---|
| SmolLM2-1.7B | running | running | running | running |
| granite-3.3-2b | running | running | running | running |
| Falcon3-3B (calc_s3f) | running | running | running | running |
| LFM2-1.2B | pending | pending | pending | pending |

Smoke checks (1 batch, GSM-Hard, calc): SmolLM2 calc fires + exact arithmetic on the model's own steps
(e.g. "0.2 * 7871124 = 1574224.8", "8300329 - 50 = 8300279.0" — both exact; misses are the model's setup, not the implant).
Falcon3 calc_s3f also fires + exact ("0.20 x 7871124 = 1574224.8" via ' x', "$8300329 + $50 = $8300379").

### LFM2-1.2B — calculator cannot transfer (documented, no run needed)
LFM2 attaches the space to the operator like Qwen (' /' ' +' ' =' are single tokens), BUT chunks digits:
  "48 / 2 = 24"  -> ['48',' /',' ','2',' =',' ','24']
  "0.2 * 7871124 = 1574224.8" -> ['0','.','2',' *',' ','787','112','4',' =',' ','157','422','4','.','8']
  "123 + 456 = 579" -> ['123',' +',' ','456',' =',' ','579']
Two concrete failures:
  (1) FUNDAMENTAL: multi-digit operands/answers are single chunk tokens ('48','787','112'...). The calculator reads operands
      digit-by-digit and emits the answer digit-by-digit; with chunked numbers there are no per-digit tokens to read or write.
  (2) BUILD-TIME: the calc_s3x reader cannot even be constructed — symbol N* ('*-') has NO single-token form in LFM2, so
      machine.symbol_token_ids asserts before attach. (The same single-digit requirement that make_reader enforces.)
No generic extension recovers this without re-deriving the entire number circuit for chunk tokens (operands of variable chunk
length, chunk-boundary arithmetic) — out of scope; it is a different machine. Same break mode for SmolLM3/OLMo-2/Phi/MiniCPM.

## IFEval / BBH — see below (lower priority; IFEval needs chat models + full vocab-table rebuild; BBH Team C reuses calc).

## IFEval port (Team D frozen 14-rule circuit -> SmolLM2-1.7B-Instruct)
Host choice: IFEval used the CHAT model Qwen3-0.6B (thinking off), so the transfer host is an instruct model. SmolLM2-1.7B-Instruct
is the closest port: ChatML template (<|im_start|>/<|im_end|>) and GPT-2 byte-level BPE like Qwen. Granite-instruct
(<|start_of_role|>) and Falcon3-instruct (<|user|>) would need a new role/answer-start rule; not attempted.
The frozen code (md5 matches <workdir>/ifeval_heldout/frozen/MD5) hard-codes Qwen in 4 ways; port_ifeval.py replaces ONLY these
by a host config (no rule, word list, detector, threshold or table definition changes):
  1. EOS / role ids (<|im_start|> 151644, 'assistant' 77091, 'user' 872, 'system' 8948, '\n' 198, ' ' 220, <|im_end|> 151645).
  2. the special-token boundary 'id >= 151643' in tables.py -> the tokenizer's special/added id set (SmolLM2 specials are ids 0-16).
  3. answer start: Qwen '</think>' '\n\n' -> ChatML '<|im_start|>' 'ass' 'istant' '\n' ('assistant' is TWO tokens in SmolLM2).
  4. vecs2.pt force_fix directions are keyed by Qwen token ids -> re-solved on SmolLM2 (solve_force.py, force part of solve_vecs2.py)
     from the host's own plain answers to OUR dev prompts (data/dev.jsonl, 250 rows); never IFEval data.
Port check (CPU): the ported tables.py with the Qwen config rebuilds the original Qwen tables: vocab strings identical (151936),
65/66 tables torch.equal, 'P' (n-gram patterns) = the same 7167 (pattern, class) rows, permuted within a class by Python set order
(PC identical; the matcher sums over patterns) -> semantically identical.
Rebuilt automatically for SmolLM2: all vocab tables (word classes, n-gram matcher for split words, case maps UPSEQ/LOWSEQ,
rolling-hash word tables, number readers).

## Incidents (honest log)
- 04:00-04:50: after the harness change, my first `pkill -f` attempts did not kill the old calc_s3 drivers (pkill matched/killed my own
  ssh session instead); two old-harness processes kept running ~30 min more (gr_test_calc, gr_hard_plain, fal_test_calc). All killed
  by PID; their outputs are in old_harness/ and NOT used.
- ~04:45: a Team C CPU symbol check (tcchk.py) for Falcon3 unexpectedly allocated on A40 pod GPU0 (teamc register used CUDA),
  briefly making two of my processes on that GPU; together with another agent's 23 GB job it caused a CUDA OOM that killed MY
  SmolLM2 GSM-Hard calc+plain runs (not the other agent's jobs). Killed it; drivers replaced by drvr.sh (bs 24, expandable
  segments, each arm retried until its json exists). bs 24 vs 32 changes only padding/batching numerics (bf16).

## Falcon3: the calculator ATTACHES but (likely) rarely FIRES — tokenizer space axis
Context tokenization (Falcon3 vs SmolLM2) of model-written equations:
  Falcon3 " has 48 + 24 = 72" -> ['4','8',' ','+',' ','2','4',' ','=',' ','7','2']
  SmolLM2 " has 48 + 24 = 72" -> ['4','8',' +',' ','2','4',' =',' ','7','2']      (= Qwen layout, the one calc_s4's grammar encodes)
Falcon3 puts a SEPARATE space token between the number and the operator and before '='. The sanitizer makes every symbol a single
token, so the reader builds and is exact, but the circuit's grammar (operand digits, then ' op', then operand, then ' =') does not
match the Falcon token sequence, so spaced equations are not recognised; only unspaced ones ('$100/2 =' ...) can fire.
Audit of the Falcon GSM-Hard calc arm (code/audit_eq.py; fresh 'a op b = c' segments in outputs): 2195 equations, 87.4% exact;
misses like '0.60 x 560233 = 336139' (implant would write 336139.8), '12.8 / 3 = 4.2667' (implant: 4.266666) -> implant silent there.
=> a real "needs rebuilding" case: the circuit grammar must accept an optional space token before operators and '=' (a circuit
change, not a reader-table change). Not done tonight (time); reported as such. Fire rate confirmed against the plain arm below.
CONFIRMED (05:30): Falcon3 GSM-Hard plain vs calc = 546/1319 both, outputs byte-identical 1319/1319 -> the calculator NEVER fired
on Falcon3 (attached, reader exact, but the grammar never matches Falcon's spaced tokens). Same equation audit in both arms (87.4%
exact = the host's own arithmetic). Doubles as a large silence check: zero spurious writes over 1319 generations.
granite GSM-Hard calc: equations in outputs 98.9% exact (2784/2815) -> the calculator fires on granite.

## GSM results, NEW harness (calc_s4, Question-stop, n=1319 each) — interim 07:00
| model | GSM8K plain | GSM8K calc | GSM-Hard plain | GSM-Hard calc | GSM-Hard 200-1318 plain -> calc |
|---|---|---|---|---|---|
| SmolLM2-1.7B | (running) | (running) | 13.95 (unx 8.42) | 19.86 (unx 7.96) | 13.67 -> 18.50 |
| granite-3.3-2b-base | 50.57 (unx 10.99) | 50.80 (unx 10.84) | 23.20 (unx 16.38) | 30.40 (unx 16.00) | 22.34 -> 29.31 |
| Falcon3-3B-Base | 68.99 (unx 1.36) | 68.99 (unx 1.36) | 41.39 (unx 4.02) | 41.39 (unx 4.02) | 40.04 -> 40.04 (calc never fires) |
| LFM2-1.2B | - | CANNOT_ATTACH | - | CANNOT_ATTACH | - |
Plain vs calc diffs: SmolLM2 hard identical outputs 761/1319, calc fixes 83 / breaks 5; granite hard 745/1319, +99/-4; granite
GSM8K 1102/1319, +17/-14; Falcon3 hard 1319/1319 identical.
Equation audit (fresh 'a op b = c' in outputs, exact %): SmolLM2 hard plain 72.6 -> calc 96.9; granite hard 80.7 -> 98.9;
granite GSM8K 96.0 -> 99.1. (The <100% residue in calc arms = equations the grammar does not read, e.g. '2 * 1031335' inside a
longer chain, '3x12647810' without spaces, or host-written values before a parse; no digit runaways: >20-digit results 7/3/0,
all inherited from the host's own expressions.)

## BBH Team C (frozen host, passive expr/calc/bool, no FT), official 3-shot CoT, greedy, max_new 400, n=250 (causal 187)
Port: run27x.py = run27.py with (a) per-host veto, (b) the generic single-token sanitizer, (c) eos = host eos_token_id,
(d) expr_gen end-write. Team C on Qwen ends an expression value with the single token '.\n' ("so the model does not re-expand a
bracket the machine already evaluated"). '.\n' is TWO tokens in SmolLM2/granite/Falcon3 -> cannot be written as one write.
  variant oe=veto (Team C's own alternative): end with the solved no-digit veto. Falcon3 result: multistep 75.6 -> 55.6 (-20):
    the tool's values are exact ('(-1 * 7 * -6 + -2) = 40', '(-5 * -7 * -6 + 9) = -201') but with no terminator the host keeps
    expanding ('= 40 + -2 = 38') -> 81 breaks / 31 fixes. Same failure mode Team C documented on Qwen.
  variant oe='.' (closest single-token form of '.\n'; register(oe) takes the OE text): running.
Falcon3 (oe=veto)  base -> +C: multistep 75.6 -> 55.6, boolean 90.0 -> 98.4, object_counting 82.4 -> 85.6;
  no-tool tasks unchanged (date 76.0, sports 57.2, causal 50.8, hyperbaton 68.0 in both arms). 7-task mean 71.43 -> 70.23.
  Note: unlike calc_s4 on GSM, Team C's expr_gen DOES fire on Falcon3 (its punctuation class includes space-only tokens).

## IFEval ODD (299 never-opened keys) — SmolLM2-1.7B-Instruct, port v1 (constants only), 08:40
force solve (dev answers only): 39 tokens needed a solved force direction, 39/39 fixed.
| arm | P-strict | P-loose | I-strict | I-loose | truncated (2048 tok) |
|---|---|---|---|---|---|
| plain | 48.49 | 51.51 | 59.82 | 62.47 | 59 |
| +implants (v1) | 60.54 | 64.21 | 71.08 | 73.51 | 68 |
prompt-strict: gained 42, lost 6. Silent prompts (circuit wrote nothing) 164/164 byte-identical; unlatched 114/114 identical.
Per rule (strict pass, plain -> implant): no_comma 14->32/36, forbid 11->23/27, repeat 4->16/21, quote 16->20/23, high 20->24/28,
caps 6->8/12, lower 18->19/23, end 9->11/13, title 18->19/19, bullets 12->13/14, ps 12->11/14, two 0->0/11, nth 1->1/6,
paras 8->2/16 (REGRESSION). (Qwen3-0.6B reference: 56.19 -> 71.91, +49/-2.)
Diagnosis of paras/two/ps (from ODD outputs, so any fix is post-diagnosis): four FIXED EMITTER strings are one token in Qwen but
not in SmolLM2: '***\n\n' -> ['***','\n\n'], '******' -> ['****','**'], '.S'/'.P' (P.S. writer) -> none. tid() returns -1, so
the paragraph-divider program is empty while the host's own '***' is still BLOCKED -> the implant deletes the host's dividers
(6 prompts lost: 1845, 2357, 2921, 2943, 3315, 3565 all had correct *** dividers in plain, none in the implant arm).
Port v2 (port_v2.py): fixed strings emitted as their host token sequence; emitter registers 4 -> 6 slots. For Qwen the sequences
equal the original token lists exactly (no-op). Validated on our DEV paras/two/ps prompts, then ODD rerun labelled post-diagnosis.
DEV validation (OUR dev data; the 118 dev prompts with paras / two_responses / postscript; plain vs v2 implant):
  P-strict 38.14 -> 54.24; paras 18 -> 30/38 (fixed), ps 49 -> 48/49, two 9 -> 0/34 (BROKEN: separator latch reads the
  single token '******' (STAR6), never set by '****'+'**' -> separator re-written every end-of-turn, 44/118 hit 2048 tokens).
  => v2.1 = v2 without the separator conversion (two_responses behaves exactly as v1). ODD rerun of v2.1 = post-diagnosis.

## BBH SmolLM2-1.7B (oe='.'), base -> +C, n=250 (causal 187)
multistep 18.4 -> 95.6, boolean 76.8 -> 95.2, object_counting 50.0 -> 73.6; date 49.2 / sports 71.2 / causal 46.0 /
hyperbaton 61.6 identical in both arms. 7-task mean 53.31 -> 70.34. 3 Team-C tasks 48.4 -> 88.1.
## BBH Falcon3-3B (oe='.') base -> +C: multistep 75.6 -> 99.6, boolean 90.0 -> 98.4, counting 82.4 -> 85.6; no-tool identical.
7-task mean 71.43 -> 76.51; 3 Team-C tasks 82.7 -> 94.5. (oe=veto variant: multistep 55.6, mean 70.23 — re-expansion.)

## Silence checks (code/check_silent_calc.py; calc_s4 sanitized, attached to the frozen host)
SmolLM2, granite, Falcon3: host weights hash unchanged; logits bit-identical (torch.equal) at every non-writing position on 8 prose
/code texts + 1 GSM text; 0 writes on prose/code. Plus at scale: Falcon3 GSM 2x1319 generations byte-identical plain vs calc.
IFEval: silent prompts byte-identical 164/164 (ODD); BBH: no-tool tasks identical per-task accuracy in all three hosts.

## BBH granite-3.3-2b (oe='.') base -> +C
multistep 32.8 -> 97.2, boolean 81.6 -> 78.0 (-3.6; 244 outputs differ, 34 fixes / 43 breaks: the tool writes the value early,
granite then re-derives and sometimes contradicts it), object_counting 70.0 -> 80.4; date 56.8 -> 57.2 (3 outputs differ:
expr fired inside date arithmetic, '364 - 3 = 361.' and the host read the '.' as a decimal point -> '361.5'; +1 fix / 0 break),
sports 89.6, causal 52.4, hyperbaton 61.6 identical. 7-task mean 63.54 -> 73.77; 3 Team-C tasks 61.5 -> 85.2.
Cost of oe='.' (vs Qwen's single '.\n'): a period after a value can be read as a decimal point (seen once on granite/date).
27-task BBH NOT run (time; ~3 h per arm per host on the shared GPUs).

==========================================================================================================================
# FINAL SUMMARY (2026-11-18 ~10:40 UTC)
| | SmolLM2-1.7B (HF) | granite-3.3-2b (IBM) | Falcon3-3B (TII) | LFM2-1.2B (Liquid) |
|---|---|---|---|---|
| tokenizer numbers | single digit, space glued to op | single digit, space glued | single digit, SPACE SEPARATE | multi-digit chunks |
| GSM8K plain -> calc | 32.75 -> 34.65 | 50.57 -> 50.80 | 68.99 -> 68.99 (never fires) | cannot attach |
| GSM-Hard plain -> calc | 13.95 -> 19.86 | 23.20 -> 30.40 | 41.39 -> 41.39 (never fires) | cannot attach |
| GSM-Hard 200-1318 | 13.67 -> 18.50 | 22.34 -> 29.31 | 40.04 -> 40.04 | - |
| unextracted (8k / hard) plain;calc | 4.93;4.55 / 8.42;7.96 | 10.99;10.84 / 16.38;16.00 | 1.36;1.36 / 4.02;4.02 | - |
| BBH multistep base -> +C | 18.4 -> 95.6 | 32.8 -> 97.2 | 75.6 -> 99.6 | not run |
| BBH boolean | 76.8 -> 95.2 | 81.6 -> 78.0 | 90.0 -> 98.4 | not run |
| BBH object_counting | 50.0 -> 73.6 | 70.0 -> 80.4 | 82.4 -> 85.6 | not run |
| BBH 4 no-tool tasks | identical | identical except date +0.4 | identical | - |
| IFEval ODD P-strict | 48.49 -> 60.54 (Instruct, v1) | not run | not run | not run |
| silence | bit-identical quiet logits, 164/164 IFEval silent prompts identical | bit-identical | bit-identical; 2638 GSM gens identical | - |

## IFEval ODD port v2.1 (post-diagnosis) — provisional 11:10
P-strict 65.22, P-loose 68.23, I-strict 74.17, I-loose 76.38 (plain 48.49 / v1 60.54); paras 2 -> 11/16 (plain 8), two 1/11,
gained 53 / lost 3. CAVEAT found: shard s0 was generated on h100 (H100, system python) while plain ran on A40 pod (A40, venv):
silent prompts byte-identical s1 (A40) 75/75 but s0 (H100) 19/87 -> hardware/library bf16 numerics, not the implant.
Rerunning s0's batches on A40 pod (shards 0/4 + 2/4 = s0's batches) for a same-hardware comparison.
FINAL v2.1 (same hardware as plain: all batches on A40 pod A40, venv): P-strict 48.49 -> 62.54 (v1 60.87 in this scoring pass;
60.54 earlier = langdetect flake, 1 prompt), P-loose 51.51 -> 65.89, I-strict 60.04 -> 72.85, I-loose 62.47 -> 75.06; gained 44 /
lost 2; silent prompts 164/164 byte-identical, unlatched 114/114; paras 8 -> 10/16 (v1: 2/16). The 65.22 above (half on H100) is
NOT used. Headline = v1 (never-opened ODD, pure constant port): 48.49 -> 60.54/60.87; v2.1 is post-diagnosis.
Remaining port gaps on SmolLM2: two_responses (separator latch is a single-token detector), P.S. marker reader ('.s'/'.p'),
nth paragraph 1/6 (as on Qwen). Truncations 59 -> 70 (2048 tok).

## Negative result: tokenizer families where the calculator cannot attach / cannot work (code/attachchk.py = run_gsm_x sanitizer)
| tokenizer | "12345" | calc_s4 attach | why it fails |
|---|---|---|---|
| LiquidAI/LFM2-1.2B | ['123','45'] | CANNOT_ATTACH (N* '*-' has no single-token form) | + numbers in 3-digit chunks |
| HuggingFaceTB/SmolLM3-3B-Base | ['123','45'] | builds | host writes numbers as 3-digit chunk tokens: digit reader never sees multi-digit operands; writer emits per-digit (off-distribution) |
| allenai/OLMo-2-0425-1B | ['123','45'] | builds | same (3-digit chunks) |
| microsoft/Phi-4-mini | ['123','45'] | builds | same (3-digit chunks) |
| openbmb/MiniCPM4-0.5B | ['▁','1','2',...] | CANNOT_ATTACH (every symbol: digits, ops, '=', newline) | SentencePiece prepends a '▁' piece to every standalone string; no symbol is one token |
| tiiuae/Falcon3-3B-Base | single digits | builds, exact reader | space is a separate token before ops/'=' -> grammar never matches -> never fires (measured: 2638/2638 GSM gens identical) |
(Correction: earlier table listed MiniCPM as multi-digit; it is single digits behind a SentencePiece prefix piece.)
Not checked (gated, no HF token on pods): Llama-3.2, Gemma-2/3.

## Process cleanup
All my processes on A40 pod and H100 pod killed/finished (checked 11:35 UTC). Pod dirs kept: A40 pod ~/xfamily, H100 pod
<workdir>/xfamily. Laptop mirror: <notes>/xfamily/{code,results}.
