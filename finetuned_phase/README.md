# Earlier phase: fine-tuned calls, the "eyes" implant, and the first frozen calculator (snapshot, not re-run)

Before the zero-training work in the rest of this repository, the project taught Qwen2.5-0.5B to **call** implants:
the model was fine-tuned on transcripts in which special tokens open and close a call, and the implant answers inside the
forward pass. This folder keeps that code and its numbers, **separately and as-is**:

* nothing here was re-run for this repository (`VERIFY.md` covers only the main code);
* the code is a snapshot of the research directories (older versions of the circuits and of `calcnet.py`, special-token
  call formats, eleven machines); it does not share code with `micromachines/`;
* some raw result files are on a machine that was not reachable while this repository was assembled; those numbers are
  marked "raw files offline" below and in `results/reported_numbers.json`.

Unlike the main results, the Qwen model here **is fine-tuned** (except `calc_icl/`), and the eyes implant contains a
**trained** CNN.

## Contents

```
eyes_and_finetune/     snapshot: eyes.py (MNIST CNN + hand-written plumbing: pixel-token reader, image-end detection,
                       slot memory, lookup), ft.py (fine-tuning), bbh_gen.py / gen_eval.py (evaluation), make_aug.py,
                       calcnet.py + circuits*.py (the eleven special-token machines of that phase), seedjobs.sh, run_eyes.sh
calc_icl/              snapshot: calc_icl.py, the first zero-training test (frozen Qwen2.5-0.5B, calls taught in context)
results/               eyes.txt (eyes, seeds 1-4), comp.txt (composition, seeds 1-4), calc_icl_100.json,
                       reported_numbers.json (every number below, with its source)
```

Data: the eyes experiments use MNIST (LeCun, Cortes and Burges), downloaded by `eyes.py` from the Hugging Face dataset
`ylecun/mnist` (CC BY-SA 3.0); it is not redistributed here. GSM8K and BBH as in the main README.

Usage, from inside each folder (paths in the scripts are relative; BBH via `BBH=...`):
`python eyes.py cnn` (train the CNN), `python eyes.py data N` (questions), then `ft.py` / `bbh_gen.py` as in `seedjobs.sh`;
`python calc_icl.py 100` (CPU, frozen Qwen2.5-0.5B).

## Numbers

**First frozen calculator** (`calc_icl/`, Qwen2.5-0.5B not fine-tuned, 6-shot, N = 100 per set; `results/calc_icl_100.json`):

| set | no implant | calculator implant |
|---|---:|---:|
| bare multiplication (3-6 digit x 2-4 digit) | 4.0 | 100.0 |
| word problems (x, +, -) | 46.0 | 98.0 |

**Eyes** (Qwen2.5-0.5B fine-tuned on questions about 3-4 pasted MNIST digits; with implants = eyes + calculator; without =
same data, the model reads the pixels and does the arithmetic itself):

| training questions | eyes + calculator | no implants | source |
|---:|---:|---:|---|
| 8 | 78.5 (76.2-81.4) | 4.0 | 4-seed sweep |
| 32 | 94.7 (89.1-97.2) | 3.7 | 4-seed sweep |
| 4,000 | 97.67 | 93.26 | mean of 5 seeds: seed 0 offline, seeds 1-4 in `results/eyes.txt` |
| 16,000 | 97.67 | 97.52 | mean of 5 seeds: seed 0 offline, seeds 1-4 in `results/eyes.txt` |

The 8- and 32-question rows are the **4-seed sweep** (same recipe, 4 seeds; ranges are min-max over seeds). An earlier
**single-seed** draft reported 88.0 / 96.0 (implants) and 6.5 / 1.5 (no implants) for these two rows; the single-seed 88
at 8 examples **did not reproduce** (the recipe was not recorded), so the 4-seed means above replace it. The sweep's
per-run summaries, script and aggregator are in [`evidence/eyes_sweep/`](../evidence/README.md) (`results/eyes.txt` here
holds only the 4k/16k seeds). The no-implant 4,000/16,000 means are computed from seed 0 (93.4 / 97.4; raw files offline) and seeds 1-4
(`results/eyes.txt`). An earlier blog figure showed 93.24 / 97.56; those were arithmetic errors. With implants, seeds 1-4
all scored 97.67 at 4,000 and 16,000 (seed 0: 97.7).

**Fine-tune comparison** (Qwen2.5-0.5B fine-tuned on GSM8K solutions three ways; 5 seeds, mean ± std; raw per-seed
files offline; seed-0 values are in the lab-notebook table copied into `reported_numbers.json`):

| | plain solutions | calls, model writes the answer | calls + calculator implant |
|---|---:|---:|---:|
| GSM8K test | 33.6 ± 0.6 | 35.2 ± 0.2 | 35.1 ± 1.2 |
| GSM-Hard | 13.4 ± 0.4 | 14.9 ± 0.9 | 23.3 ± 0.6 |
| bare arithmetic, unseen phrasings | 50.8 ± 3.4 | 54.5 ± 1.5 | 99.0 ± 0.4 |

The figures for these two tables are `figures/6_eyes.png` and `figures/8_finetune.png` (`figures/make_figs.py`, from
`results/reported_numbers.json`).
