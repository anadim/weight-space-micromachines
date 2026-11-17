import json, sys, time, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
# Minerva 4-shot prompt (the standard Minerva MATH prompt, as in lm-eval-harness minerva_math)
SHOTS = r"""Problem:
Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain

$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
Final Answer: The final answer is $-\frac{2}{3}$. I hope it is correct.

"""
if __name__ == "__main__":
    model = sys.argv[1]; outp = sys.argv[2]; bs = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    tok = AutoTokenizer.from_pretrained(model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda().eval()
    prompts = [SHOTS + "Problem:\n" + ex["problem"] + "\n\nSolution:\n" for ex in ds]
    order = sorted(range(len(ds)), key=lambda i: len(tok(prompts[i]).input_ids))
    res = {}
    t0 = time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b+bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None, top_k=None,
                           stop_strings=["\nProblem:"], tokenizer=tok, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            new = g[j, enc.input_ids.shape[1]:]
            ntok = int((new != tok.pad_token_id).sum())
            raw = tok.decode(new, skip_special_tokens=True)
            stopped = "\nProblem:" in raw
            txt = raw.split("\nProblem:")[0]
            ex = ds[i]
            res[i] = dict(i=i, unique_id=ex["unique_id"], subject=ex["subject"], level=ex["level"], problem=ex["problem"],
                          gold=ex["answer"], solution=txt, ntok=ntok, hit_limit=(ntok >= 1024 and not stopped))
        print(f"{b+len(idx)}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    json.dump([res[i] for i in range(len(ds))], open(outp, "w"), indent=1)
    print("done")

