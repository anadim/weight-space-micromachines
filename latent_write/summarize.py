"""Print markdown tables from results/*.json (local, no model). Usage: python summarize.py [results_dir] > summary.md"""
import json, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
pc = lambda x: f"{100 * x:.1f}"
for f in sorted(glob.glob(f"{RES}/*.json")):
    d = json.load(open(f))
    Ls = sorted(int(k[1:].split("_")[0]) for k in d if k.startswith("L") and k.endswith("_comp_true"))
    print(f"\n### {d['task']} {d['range']} {d['model']} (n={d['n']}, circuit {d['units']} units / {d['nnz']} nnz)")
    print(f"plain {pc(d['plain']['acc'])} | text oracle {pc(d['text_oracle_true']['acc'])} (corrupted text followed "
          f"{pc(d['text_oracle_corr']['follow'])}) | text tool line {pc(d['text_tool_line']['acc'])}")
    cc = d["circuit_check_task"]
    print(f"circuit check: spans {cc['spans_fired']}, slot positions {cc['slot_positions']}, mismatches pos/F/P "
          f"{cc['pos_mismatch']}/{cc['F_mismatch_positions']}/{cc['P_mismatch']}")
    for k in d:
        if k.startswith("silent"): print(k, d[k])
    print("\n| L | " + " | ".join(map(str, Ls)) + " |\n|---|" + "---|" * len(Ls))
    rows = [("compiled: acc", "comp_true", "acc"), ("compiled, corrupted P': follow", "comp_corr", "follow"),
            ("compiled, mean-value control (query): acc", "comp_mean", "acc"),
            ("compiled, fixed mean writer at every span: acc", "comp_meanall", "acc"),
            ("Python ref (all spans): acc", "ref_true", "acc"), ("Python ref, corrupted: follow", "ref_corr", "follow"),
            ("old Python write (query only): acc", "refq_true", "acc"), ("old Python write (query only), corrupted: follow", "refq_corr", "follow")]
    for name, k, m in rows:
        print(f"| {name} | " + " | ".join(pc(d[f"L{l}_{k}"][m]) for l in Ls) + " |")
    print("| compiled == ref outputs (true/corr/mean, of n) | " + " | ".join(
        "/".join(str(v) for v in d[f"L{l}_outputs_identical"].values()) for l in Ls) + " |")
    print("| residual: non-slot bit-equal / slot max abs diff | " + " | ".join(
        f"{'Y' if d[f'L{l}_residual_check']['nonslot_bit_equal'] else 'N'}/{d[f'L{l}_residual_check']['slot_max_abs_diff']:.0e}" for l in Ls) + " |")
    ne = sum(d[f"L{l}_residual_check"]["slot_elems"] - d[f"L{l}_residual_check"]["slot_elems_bit_equal"] for l in Ls)
    tot = sum(d[f"L{l}_residual_check"]["slot_elems"] for l in Ls)
    print(f"slot elements not bit-equal to the Python reference: {ne} of {tot} ({100*ne/tot:.4f}%); "
          f"max |h| seen {max(d[f'L{l}_residual_check']['max_abs_h'] for l in Ls)} (big-M = 1e6)")
