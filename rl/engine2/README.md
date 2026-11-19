# engine2: megakernel rollout + trainer for RL with the full implant library (work in progress)

A faster rollout and training engine for the "library" RL arm (`../library/`: Qwen3-0.6B host + `calc_s4` + 8 BBH
implants + the 14-rule IFEval implant, each behind its own trainable in-wire and out-adapters, Dr.GRPO).
**No number in the blog post comes from engine2.** The blog's RL runs used the v2 engine (`../gsm_grpo/engine.py`,
`../wiring/engine.py`, `../library/lib_engine.py`); engine2 was built afterwards and is shipped as a snapshot of the
work in progress. Its hard rule is the same as everywhere else in this repo: implant outputs are bit-exact, and any
mismatch is a bug.

## Kernels: CUDA C++ through NVRTC, no nvcc

* `tk3_kernel.cu` (current), `tk2_kernel.cu`, `tk_kernel.cu`, `tkf_kernel.cu`: the 9 implants (calc_s4 + 8 BBH tools)
  as **one kernel launch per decode step**. Every implant weight and wire is an integer, so the kernel computes the
  sparse ReLU stacks in exact integer arithmetic, pushing only deviations from a per-step baseline (`DESIGN.md` §2-5).
* `lmhead.cu`: fused log-prob head for training (max / sum-exp / target in one pass, logit gradient in place).
* `tk.py` / `tk_compile.py` / `lmhead.py`: compile the `.cu` source at load time with **NVRTC** (CUDA C++ -> cubin for
  the GPU's own `sm_XX`, `-default-device`) via `ctypes`, then launch with the driver API (`cuModuleLoadData`,
  `cuLaunchKernel`) on torch's current stream, so the launches are CUDA-graph capturable. No nvcc and no custom
  build step: only the NVRTC library that ships with the CUDA 12.x torch wheels (`nvidia-cuda-nvrtc-cu12`).
* The host model's decode kernels are the v2 engine's Triton kernels (`../gsm_grpo/hostkern.py`); engine2 replaces the
  implant side, the IFEval step (lazy: only rows that can write), the trainer's feature pass and the log-prob head.
* `vmm.py`: KV cache in CUDA virtual memory (fixed addresses so graphs stay valid; pages released during training).

## Status (see `NOTEBOOK.md`, `DESIGN.md`)

| check | result |
|---|---|
| tools kernel vs the reference step functions, all 9 tools, 32 x 690 reference + 256 x 1076 sampled positions | 0 mismatches |
| engine2 vs the old engine, greedy reference (tokens, log-probs, fire flags, in-wire codes, IFEval writes), A40 and H100 | 32/32 rows bitwise equal |
| trainer: recorded vs teacher-forced tool features; one update old vs new | features equal at all positions; every parameter bitwise equal (before the fused log-prob head; with it, update cosine 0.99999988) |
| full update, m1 recipe (64 prompts x 8, max_new 512) | A40 ~150 s -> 36-43 s; H100 ~70 s -> 17-20 s |
| 25-update smoke vs the old engine (seed 1, same prompt batches) | reward 0.362 vs 0.358, dev@25 51.0 vs 52.25 |
| open | a seed-0 rerun drifted (calc tokens per call 8 -> 4.5 from u6; replays of the guard streams are bitwise); slot refill is exact but slow |

## Running it

engine2 builds the old engine by executing `../library/train_library.py` up to its main loop (`ref_env.py`) and then
swaps parts. Set `E2_LIB` to a directory laid out like the run directory it was developed in: `train_library.py` and
the rest of `../library/`, the SFT'd host in `mixed/sft_host`, and `engine2/` (this folder) inside it. `E2_REF` points
at a bit-exactness reference dump (`../library/mixed/engine_reference_A40.json` is the A40 one; the H100 one is
rebuilt with `train_library.py --dump_ref 32 --max_new 128`). Then, from `$E2_LIB/engine2`:

```bash
python test_tools.py          # tools kernel vs reference step functions
python test_eng2.py           # engine2 vs the reference dump (add --lazy 1 / --roll 1 variants, see NOTEBOOK)
python test_train2.py         # trainer: recorded features + one update vs the old trainer
python run2.py --run_dir runs/smoke25 --updates 25 --B_max 256   # m1-recipe smoke run
```
Requirements as for `../library/` plus a CUDA 12.x driver; tested on A40 (sm_86) and H100 (sm_90).
`runs/` holds the small logs and JSON outputs of the tests and benchmarks above (no checkpoints; `tables.pt`,
118 MB, and `vocab_strs.json` are caches rebuilt on first use).
`eng2_next.py` is the in-progress next version of `eng2.py`.
