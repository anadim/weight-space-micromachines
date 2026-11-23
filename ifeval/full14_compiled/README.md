# The 14 IFEval rules compiled into plain weight tensors

`ifeval/heldout/circuit.py` (the 14-rule implant, md5 `cd15769c...`) runs as batched tensor code. This folder compiles it, line
by line, into plain tensors that an executor with only four operations runs: **matmul, ReLU, argmax / hard attention, gather of
fixed rows** (plus one disclosed op, the RMSNorm of the LM-head readout). The compiled implant produces **byte-identical text to
`circuit.py` on all 541 IFEval prompts**, so the 14-rule numbers (57.86 -> 72.09 prompt-strict on this GPU) carry over.

Size: 414 registers, **9,304 ReLU units, 77,887 nonzero weights** of logic; plus vocabulary tables. The exported state_dict is
1.23 GB (348M numbers), of which 311M are the copied LM-head rows (bf16, all 151,936, the readout's masked argmax ranges over the
whole vocabulary) and the force table F (fp32). See `NOTES.md` for the construction (exact mod-(2^31-1) hash with a ReLU
staircase, prefix-count compaction instead of `torch.sort`, static < 2^52 exactness certificate).

## Files

| file | role |
|---|---|
| `sym.py` | symbolic integer wires (affine forms + proven intervals) lowered to exact ReLU units; uses `wsm_hdl.py` (same file as `../compiled_v2/wsm_hdl.py`) |
| `circ14.py` | transliteration of `circuit.py` (line numbers in comments) into two netlists: STEP and WRITE |
| `execu.py` | export to a state_dict of plain tensors, and the generic 4-op executor |
| `export.py` | builds the netlists + force table and saves `implant14_sd.pt` (`--shared 1`: shared-weights export, see below) |
| `check_shared.py` | bitwise identity of the shared-weights export against the full export |
| `build_test.py` | prints ReLU units / wires / the exactness bound per netlist (CPU, seconds) |
| `run_c.py` | IFEval generation driven by the compiled implant (same decode loop/batches as `../transfer/run_t.py`); `--shadow 1` runs `circuit.py` alongside and compares all registers, the blocked mask, decisions and writes at every step |
| `compare.py` | byte-identity of two runs + silence vs the plain run |
| `lockstep.py`, `lockstep_util.py`, `hash_state_test.py`, `stress_gen.py`, `prof.py` | CPU equivalence / stress tests and profiling |
| `run_all.sh` | every command, in order |
| `results/full14_comp_imp.json`, `results/full14_ref2_imp.json` | text-free per-prompt records (seeded scorer) of the compiled run and the `circuit.py` rerun |
| `results/verify_logs/` | shadow / lockstep / hash / stress logs (paths sanitized) |
| `results/stress/*.jsonl.gz` | the 3 x 240 random stress sequences (token ids) used by `lockstep.py --seqs` |
| `NOTES.md` | the lab notebook (the first half is an earlier, superseded "cannot be compiled" finding; kept for the record) |

## Weights (not in the repo)

`implant14_sd.pt`, 1,230,023,169 bytes, 1,827 tensors.
sha256 `e85650adb761df2a292c5d1ce4aa836ddb9b2a32f3d2dbfc77b37a849787c200` (md5 `b71f2342612616269b9147c006f028fc`).
Rebuild (about 20 s plus loading the host), from `ifeval/heldout` with `PYTHONPATH=.`:

```bash
python ../full14_compiled/export.py implant14_sd.pt --device cuda --dtype bf16
```

Inputs it reads from the working directory: `tables.py` + its cache `tables.pt` (built on first use from the tokenizer;
original sha256 `be07bb5b...a52647`), `vocab_strs.json` (sha256 `4a3b1686...a05e`), `vecs2_forcefix.pt` (in the repo; sha256
`2d5ee108...e6bd`), and `Qwen/Qwen3-0.6B` from Hugging Face. The netlist build is deterministic; the force table is computed from the
host weights with the same code as `circuit.Circuit.force`.

**Shared-weights export** (`export.py OUT.pt --shared 1`): stores neither the LM-head copy `voc.E` nor the force table
`force.F`; the executor reads the host's own (tied) LM-head / embedding matrix (`Executor(sd, host_E=model.lm_head.weight)`,
done by `run_c.py`) and computes the force rows of the gathered tokens on the fly with the ops that built the table.
348.2M -> 38.7M numbers (1.23 GB -> 0.30 GB). `python check_shared.py FULL.pt SHARED.pt` (from `ifeval/heldout`, same
PYTHONPATH) checks the tensors, `voc.E` == the host matrix and every force row bitwise; a full run on all 541 prompts gave
text identical to the full export and to `circuit.py` (silent prompts byte-identical).

## Reproduce

```bash
python ifeval/data/prepare_ifeval.py
bash ifeval/full14_compiled/run_all.sh        # build, export, reference + compiled + plain runs, compare, stress tests
```

Expected (A100 80GB PCIe, torch 2.8.0+cu128, transformers 4.56.2, lm_eval 0.4.13, Python 3.12.3):

| check | expected |
|---|---|
| compiled vs `circuit.py` rerun, 541 prompts | 541/541 responses byte-identical; circuit reports 541/541 identical |
| `circuit.py` vs itself (noise floor, same GPU) | 541/541 |
| shadow mode (26,242 steps, 24,891 writes) | 0 register / blocked-mask / decision / write mismatches |
| silent prompts vs plain | 324/324 byte-identical |
| IFEval prompt-strict (seeded scorer) | plain 57.86 -> compiled 72.09 (all 541); odd 57.53 -> 71.24 |
| `hash_state_test.py --trials 400` | 0 mismatches over 9,600 rows |
| lockstep stress (forbid / mixed / random, 240 seqs each) | 0 mismatches |

Runtime: compiled run with shadow ~1.7 h on a shared A100 (STEP 17 ms + WRITE 18 ms per token at B = 24); the `circuit.py`
rerun ~1.4 h; each CPU stress set 1-3 h. Deterministic on one GPU type; across GPU types bf16 greedy text can differ (compare
the compiled and reference runs on the same GPU).

Known non-identities (see `NOTES.md`): the digit accumulator saturates at 10^6 where `circuit.py`'s float32 would overflow to
NaN after >= ~39 consecutive digits (never happens in IFEval; seen once in a synthetic CPU lockstep, `results/verify_logs/lockstep_cpu_b7.log`);
and the readout RMSNorm is computed as in `circuit.py`, not as ReLU/matmul.
Not in weights, as before: the decode loop, the forward pre-hook with its on/off switch, and run_c's diagnostics.
