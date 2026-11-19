import sys, runpy, torch, fc
@torch.no_grad()
def call(self, qs, rs, bs=16, tmpl=fc.JUDGE):
    out = []
    for i in range(0, len(qs), bs):
        msgs = [self.tok.apply_chat_template([{"role": "user", "content": tmpl.format(q=q[:3000], r=(r if r.strip() else "(empty reply)")[:1500])}],
                tokenize=False, add_generation_prompt=True) for q, r in zip(qs[i:i + bs], rs[i:i + bs])]
        enc = self.tok(msgs, return_tensors="pt", padding=True).to("cuda")
        lg = self.m(**enc, logits_to_keep=1).logits[:, -1].float()   # same last-position logits, no full-vocab tensor
        out += (lg[:, self.y] - lg[:, self.n]).tolist()
    return out
fc.Judge.__call__ = call
sys.argv = ["s6_judge.py"] + sys.argv[1:]
runpy.run_path("s6_judge.py", run_name="__main__")
