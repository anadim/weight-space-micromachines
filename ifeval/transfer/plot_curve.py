"""IFEval add-one-rule curve (all 541 prompts, prompt-strict), three Qwen3 hosts -> ifeval_curve.png.
Data: results/ifeval_curves.json (= the 'ALL P-strict' column of results/{c06,c17,c4}_curve_score.txt).
The blog figure (22_ifeval_curve.png) was drawn from the same JSON with the blog's house style (figures/blog/ if present)."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(H, "results", "ifeval_curves.json")))
labs = ["none", "+ no commas", "+ N paragraphs", "+ repeat request", "+ forbid words", "+ para k starts", "+ all caps",
        "+ two answers", "+ lowercase", "+ end phrase", "+ highlights", "+ quotes", "+ bullets", "+ title", "+ P.S."]
fig, ax = plt.subplots(figsize=(11, 6))
for key, nm in (("c4", "Qwen3-4B"), ("c17", "Qwen3-1.7B"), ("c06", "Qwen3-0.6B")):
    ax.plot(range(15), C[key], "-o", label=f"{nm}: {C[key][0]:.1f} -> {C[key][-1]:.1f}")
ax.set_xticks(range(15)); ax.set_xticklabels(labs, rotation=40, ha="right")
ax.set_ylabel("IFEval prompt-strict accuracy (%)"); ax.set_ylim(50, 90); ax.grid(axis="y", alpha=.3); ax.legend()
ax.set_title("Add one rule implant at a time (order picked on our own FRESH prompts)")
fig.tight_layout(); fig.savefig(os.path.join(H, "ifeval_curve.png"), dpi=120); print("wrote ifeval_curve.png")
