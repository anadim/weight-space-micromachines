"""Key figure: accuracy of the latent write vs the layer it enters, across value counts, prompts and models.
Reads results/*.json (copied from the pods)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "results")
J = lambda f: json.load(open(os.path.join(R, f)))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# panel 1: number of latent values (0.6B)
q4 = J("q4_mid_Qwen3-0.6B-Base.json"); s2 = J("q1_sum2_Qwen3-0.6B-Base.json"); s3 = J("q1_sum3_Qwen3-0.6B-Base.json")
Ls = [0, 4, 8, 12, 14, 16]
ax[0].plot(Ls, [100 * q4[f"L{L}:slot:true"][0] for L in Ls], "o-", label="1 value  (a*b)+c")
ax[0].plot(Ls, [100 * s2[f"all:L{L}:synth:true"]["true"] for L in Ls], "s-", label="2 values (a*b)+(c*d)")
ax[0].plot(Ls[:-1], [100 * s3[f"all:L{L}:synth:true"]["true"] for L in Ls[:-1]], "^-", label="3 values (+(e*f))")
ax[0].plot(Ls[:-1], [100 * s3[f"all:L{L}:synth:corrall"]["corrall"] for L in Ls[:-1]], "^--", c="C2", alpha=.6,
           label="3 values, all corrupted: follow")
ax[0].axhline(100 * s3["text_oracle_all(true,..)"][0], c="C2", ls=":", lw=1)
ax[0].set_title("Qwen3-0.6B: several latent values")

# panel 2: word problems, where to write
for f, nm in [("q2_rows_Qwen3-0.6B-Base.json", "rows"), ("q2_lost_Qwen3-0.6B-Base.json", "lost")]:
    if not os.path.exists(os.path.join(R, f)):
        continue
    d = J(f)
    for g in os.listdir(R):  # merge late-layer supplementary runs
        if g.startswith(f.replace(".json", "_")):
            d.update({k: v for k, v in J(g).items() if ":L" in k})
    for loc, st in [("both", "-"), ("numA", "--")]:
        xs = [L for L in range(0, 23) if f"{loc}:L{L}:synth:true" in d]
        ax[1].plot(xs, [100 * d[f"{loc}:L{L}:synth:true"][0] for L in xs], "o" + st,
                   label=f"{nm}: {'both operand sentences' if loc == 'both' else 'operand digits only'}")
ax[1].set_title("Word problems: where to write")

# panel 3: model size (same depth 28)
for f, nm in [("q4_mid_Qwen3-0.6B-Base.json", "Qwen3-0.6B"), ("q4_mid_Qwen3-1.7B-Base.json", "Qwen3-1.7B")]:
    d = J(f)
    xs = [L for L in range(0, 28, 2) if f"L{L}:slot:true" in d]
    l, = ax[2].plot(xs, [100 * d[f"L{L}:slot:true"][0] for L in xs], "o-", label=f"{nm} slot")
    xs2 = [L for L in xs if f"L{L}:slot+suffix:true" in d]
    ax[2].plot(xs2, [100 * d[f"L{L}:slot+suffix:true"][0] for L in xs2], "s--", c=l.get_color(), label=f"{nm} slot+suffix")
ax[2].set_title("(a*b)+c, model size at equal depth")
for a in ax:
    a.set_xlabel("layer the value is written into (of 28)"); a.set_ylabel("exact accuracy %"); a.set_ylim(-3, 103)
    a.legend(fontsize=7.5); a.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "latent_write2_curves.png"), dpi=130)
print("saved")
