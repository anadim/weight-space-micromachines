"""[28] GSM8K: patch the COPIES (rl28/gsm8k/fe, rl28/gsm8k/eng) so the same stack runs on another Qwen3 host.
Host chosen by env HOST_MODEL (default Qwen/Qwen3-0.6B-Base -> every patched file behaves exactly as the original).
Only host plumbing changes: model name, hidden/FF sizes, which veto file, dim asserts on host-fitted vectors, SW_DMARG in
the RL engine copy. No circuit/core code is touched. usage: python3 patch17.py  (idempotent)
Release note: kept for reference. The patched files already ship in the overlays trees/stk_r2q_17 and trees/arm17; TREES
below are the original working copies (fe/ = a copy of final_eval with stk_r2q_17, eng/ = a copy of engine/rel/v5)."""
import os, re, sys, glob, shutil, py_compile
HERE = os.path.dirname(os.path.abspath(__file__))
LIT = "'Qwen/Qwen3-0.6B-Base'"
ENV = "__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base')"
TREES = [f'{HERE}/fe/dig_san', f'{HERE}/fe/wiring', f'{HERE}/fe/stk_r2q_17', f'{HERE}/eng/dig_san', f'{HERE}/eng/stk', f'{HERE}/eng/wiring']
n_lit = 0
for t in TREES:
    for p in glob.glob(f'{t}/*.py'):
        s = open(p).read()
        if LIT in s and ENV not in s and '_SHAPES' not in s:
            k = s.count(LIT); s2 = s.replace(LIT, ENV)
            open(p, 'w').write(s2); n_lit += k
            try: py_compile.compile(p, doraise=True)
            except Exception as e: print('COMPILE FAIL', p, e); open(p, 'w').write(s)
print('model literals replaced:', n_lit)

FQ_OLD = "H, KVH, HD, D, FF, NL, EPS, THETA = 16, 8, 128, 1024, 3072, 28, 1e-6, 1e6\n"
FQ_NEW = ("import os as _os0\n"
          "HOST = _os0.environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base')   # [28] host choice; shapes below per host\n"
          "_SHAPES = {'Qwen/Qwen3-0.6B-Base': (1024, 3072), 'Qwen/Qwen3-1.7B-Base': (2048, 6144)}\n"
          "H, KVH, HD, NL, EPS, THETA = 16, 8, 128, 28, 1e-6, 1e6\n"
          "D, FF = _SHAPES[HOST]\n")
LD_OLD = "def load_hf_params(name=" + ENV + ", device='cuda', dtype=torch.float32):\n    from transformers import AutoModelForCausalLM\n    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)\n"
LD_NEW = ("def load_hf_params(name=None, device='cuda', dtype=torch.float32):\n    from transformers import AutoModelForCausalLM\n"
          "    name = name or HOST\n    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)\n"
          "    c = m.config\n    assert (c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim,\n"
          "            c.tie_word_embeddings) == (D, FF, NL, H, KVH, HD, True), ('host shape mismatch', name, c)\n")
for t in ('fe', 'eng'):
    p = f'{HERE}/{t}/wiring/fastqwen.py'; s = open(p).read()
    if '_SHAPES' not in s:
        assert s.count(FQ_OLD) == 1 and s.count(LD_OLD) == 1, p
        s = s.replace(FQ_OLD, FQ_NEW).replace(LD_OLD, LD_NEW); open(p, 'w').write(s); print('patched', p)
    # veto: per-host file (1.7B: repo/results/gsm_bigger/veto_q17.pt, same solve_veto.py procedure on 1.7B-Base)
    p = f'{HERE}/{t}/wiring/implant.py'; s = open(p).read()
    a = "VETO = os.path.join(HERE, 'wsmrepo', 'veto_qwen3.pt')\n"
    if 'veto_q17' not in s:
        assert s.count(a) == 1, p
        s = s.replace(a, "VETO = os.path.join(HERE, 'wsmrepo', 'veto_q17.pt' if os.environ.get('HOST_MODEL', '').endswith('1.7B-Base') else 'veto_qwen3.pt')\n")
        open(p, 'w').write(s); print('patched', p)
    shutil.copy(f'{HERE}/../../repo/results/gsm_bigger/veto_q17.pt', f'{HERE}/{t}/wiring/wsmrepo/veto_q17.pt')

# dim asserts on host-fitted vectors (verifier probe, stop row): a 0.6B file on the 1.7B host fails loudly
for p in (f'{HERE}/fe/stk_r2q_17/comp_serial.py', f'{HERE}/eng/stk/comp_serial.py'):
    s = open(p).read()
    a = "        pr = torch.load(probe); self.v_w = pr['w'].to(dev).float(); self.v_b = float(pr['b']); self.v_L = int(pr.get('layer', 18))\n"
    if 'probe dim' not in s:
        assert s.count(a) == 1, p
        s = s.replace(a, a + "        assert self.v_w.numel() == Q.D, ('probe dim', self.v_w.numel(), Q.D)\n")
    a = "        self._stoprow = sr['r']; self.t_stopdir = sr['d'].float()\n"
    if a in s and 'stoprow dim' not in s:
        s = s.replace(a, a + "        assert sr['r'].numel() == Q.D, ('stoprow dim', sr['r'].numel(), Q.D)\n")
    open(p, 'w').write(s); py_compile.compile(p, doraise=True)
sys.argv = ['patch_dmarg.py', f'{HERE}/eng/stk']
exec(open(f'{HERE}/fe/patch_dmarg.py').read())
print('ok')
