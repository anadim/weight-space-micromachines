# C35: text-calculator-tool baseline

**Question.** If a plain text calculator tool gets the same RL, does it do about as well as the calculator implant?

All arms start from Qwen3-0.6B-Base, use the GSM8K 8-shot harness (max_new 320) and train with the same RL recipe: 300
updates, 64 prompts × 8, T 1, lr 1e-6, Dr.GRPO, clip 0.2, warmup 5, dev every 25 updates on 800 items. Only arm X is
new.

| arm | what |
|---|---|
| 1 | plain RL, plain 8-shot prompt |
| 2 | RL + the calculator implant, one attempt (as in the main RL results, `rl/gsm_grpo`) |
| X | RL + a **text tool**, described below |

How arm X works:
- **Prompt.** The 8 shots are the same, but GSM8K's own `<<E=R>>` annotations are rewritten as `<calc>E</calc>R`.
- **Tool.** When a row emits `</calc>`, the host evaluates E exactly (`fractions.Fraction` AST over `+ - * / ** %` and
  parentheses; non-integers rounded half-up to 6 decimals; otherwise `ERR`). The result tokens are forced into the
  decode. At most 16 calls per row.
- **Sync.** The batched engine pauses a row that samples `>` within 4 tokens of `/` and checks it at the next host sync,
  every 4 decode steps for X (`--sync_every 4`).
- **Token accounting.** Tool tokens count toward the 320-token cap and toward tok/q, but are masked out of the policy
  loss. "tok/q (model)" excludes them.
- **No call fine-tune is needed.** On Base, 97% of rows call the tool.

Base-model checks (`res/check_tool.json`, greedy, 256 dev items):
- the tool engine on the plain prompt is identical to the stock engine on 99.2% of rows;
- tool ERR 1.7%;
- teacher-forced HF agreement is 99.6% on rows with calls and 99.9% on rows without;
- sync 4 vs 16 gives identical outputs and is 1.75× faster.

The bracket tags [35] / [37](a) in code and labels are experiment labels from the lab log: [35] = this experiment,
[37](a) = the model-token-cap re-eval. They are kept so that `rows.json` regenerates byte-identically.

## Results
Eval: GSM8K test (1319) / GSM-Hard (1319) / GSM-Plus testmini (2100), T 1, one attempt, eval seeds 7/11/13. Results
are pooled over train seeds 0–11, so 36 evals per arm. Source: `rows.json`; full table in `res/ana_tt_12seeds.txt`.

| row | test | Hard | GSM-Plus | tok/q model (total) | calls/q | ms/q batched | ms/q B=1 |
|---|---|---|---|---|---|---|---|
| 0 (Base, plain prompt) | 28.20 | 11.25 | 19.54 | 101 | 0 | 4.6 | 282 |
| 02 (Base + implant) | 29.06 | 14.89 | 19.83 | 101 | 1.89 | 7.3 | 342 |
| 0X (Base + text tool) | 26.18 | 15.77 | 17.32 | 124 (131) | 2.83 | 5.9 | 379 |
| 1 (plain RL) | 68.37 | 41.54 | 54.32 | 133 | 0 | 4.8 | 361 |
| 2 (RL + implant) | 69.74 | 54.08 | 55.02 | 128 | 3.00 | 7.4 | 422 |
| X (RL + text tool) | 69.29 | 55.97 | 54.27 | 141 (149) | 3.09 | 5.9 | 432 |

Paired differences (12 train seeds; train-seed × item bootstrap 95% CI):

| difference | test | Hard | GSM-Plus |
|---|---|---|---|
| X − 2 | −0.45 [−1.50, +0.54] | +1.90 [+0.65, +3.19] | −0.75 [−1.60, +0.13] |
| X − 1 | +0.92 [−0.20, +2.03] | +14.44 [+12.82, +16.22] | −0.05 [−1.00, +1.00] |
| 2 − 1 | +1.37 [+0.40, +2.35] | +12.54 [+11.01, +14.16] | +0.70 [−0.04, +1.51] |

**[37](a) re-eval.** The arm-X u300 checkpoints of train seeds 0–4 were re-evaluated with the cap on *model* tokens
only (`--cap_extra 128`: up to 128 tool tokens on top of 320). These are rows Xm·5 vs X·5 and 2·5:

| | test | Hard | GSM-Plus |
|---|---|---|---|
| X·5 | 69.09 | 55.86 | 54.12 |
| Xm·5 | 69.16 | 57.26 | 54.43 |
| Xm·5 − X·5 | +0.07 [−1.13, +1.26] | +1.39 [+0.35, +2.47] | +0.31 [−0.90, +1.52] |
| Xm·5 − 2·5 | −0.99 [−2.34, +0.33] | +3.17 [+1.76, +4.63] | −0.59 [−1.82, +0.59] |

This change, plus putting each arm at its own dev-selected config, led to the equal-footing comparison C36
(`../calcimp`).

The GSM-Plus breakdown by perturbation type is in the same text output. `res/ana_tt_8seeds.txt` is the earlier 8-seed
version.

## Layout
| path | what |
|---|---|
| `code/rl/gsm_grpo/` | **overlay** on the release `rl/gsm_grpo/` (see "Code" below) |
| `mk_code.sh`, `code_data.sha256` | build the runnable tree `code/` (release `rl/gsm_grpo` + `micromachines` + data + veto, then the overlay) and check data hashes |
| `job.sh` | job runner: `check`, `train ARM SEED NAME`, `eval ARM RUN NAME` (one GPU) |
| `commands.sh` | prints the exact command line of every job behind the shipped results |
| `res/items.jsonl.gz` | per eval dir: summary + per-item correctness (`ok` bit string in item order) + per-run token / call sums |
| `res/configs.json`, `res/train_steps.jsonl.gz` | per train run: config.json, and the dev records plus main per-update fields of steps.jsonl |
| `res/gsmplus_ptype.json` | GSM-Plus testmini perturbation type per item (for the by-type breakdown) |
| `ana_tt.py` | the table → `rows.json` (reads `res/`) |
| `why/why.py` | arithmetic-slip split of wrong answers; needs the generated text (not shipped) |

### Code
`code/rl/gsm_grpo/` holds only files that are new or differ from the release `rl/gsm_grpo/`.

New:
- `tooltext.py`: the `<calc>` prompt and the exact evaluator.
- `fe_tool.py`: eval of arms 1 / 2 / X, plus single-stream timing.
- `check_tool.py`: the Base checks.

Modified copies:
- `engine.py`: `tool=`, `_book_tool`, `_tool_sync`.
- `train2.py`: arm `X`, `--sync_every`.

`../calcimp/tt/code/rl/gsm_grpo/` is the later C36 version: same `tooltext.py` and `check_tool.py`, plus
`--cap_extra` and the implant arms. With `--cap_extra 0` it reproduces this folder's arm-X eval byte-identically; the
[37](a) re-evals `e37a_Xs*` were run with that code (`../calcimp/tt/commands.sh`, last block). Their results are
included in `res/items.jsonl.gz` here.

`bash mk_code.sh` builds `code/` in place. It needs:
- `data/gsm8k/` from `data/prepare_gsm.py`, and `data/gsmhard/`;
- GSM-Plus testmini from `uniform/final_eval/data/prepare_gsmplus.py` (env `GSMPLUS`);
- `runs/veto_qwen3.pt` from `scripts/solve_veto.py` (env `WSM_VETO`).

All are checked against `code_data.sha256`.

## Rerun
```
cd checks/tooltext
bash mk_code.sh                          # once
bash commands.sh                         # list the jobs; run each line on one GPU:
CUDA_VISIBLE_DEVICES=0 bash -c "$(bash commands.sh | grep ' aX_s3 ')"
python3 ../calcimp/mk_res.py --out out --res res --evals '^(ev_|e37a_)' --runs '^a[12X]_s'
python3 ana_tt.py --write                # -> rows.json (CPU, a few minutes)
```

`job.sh` runs with `WSM_ROOT=code`. It sources the release `rl/gsm_grpo/env_local.sh`, which you edit for your Python
environment. Run and eval dirs (checkpoints included) go under `$WSM_DATA`, default `<release>/runs/tooltext`; small
outputs are copied to `out/`, which is git-ignored.

To include the [37](a) rows, copy `../calcimp/tt/out/e37a_Xs*` into `out/` before `mk_res.py`.

**Hardware and time.** One B200 per job. Training takes ~24 / 33 / 30 min for arms 1 / 2 / X. Each eval job takes ~2–3
min. The original runs went one job per GPU through a cluster queue; the eval was submitted when its train finished.

**Verified.** `ana_tt.py --write` on the shipped `res/` regenerates the shipped `rows.json` byte-identically. The
working-tree original differed only in a `notes` string with internal paths. It also prints `res/ana_tt_12seeds.txt`
identically.

## Not shipped
- Checkpoints. Regenerate them with `commands.sh`.
- Generated text: the fe_tool.py per-item `*.jsonl` with model output.
- The 6 example traces in `check_tool.json`.
- Training stdout.

`why/why.py` (and `../calcimp/verify_core.py`, `forms.py`, `tokprobe.py`) need the regenerated eval dirs in `out/`.
