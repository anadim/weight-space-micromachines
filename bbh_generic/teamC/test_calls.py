"""Machine-alone tests of the call-format tools: '<tool_call>NAME ARGS</tool_call>' is the context; the machine's writes
are fed back until </tool_response>; compare with callfmt_C references."""
import os, sys, random, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.environ.get('WSM_REPO', os.path.join(HERE, '..', '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import teamc_tools as TT, callfmt_C as CF
from micromachines import SPECS, first
N, seed = int(sys.argv[1]), int(sys.argv[2])
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', torch_dtype=torch.float32).cuda().eval()
MS = ('expr_call', 'bool_call', 'calc_call')
TT.prep_call_tokens(model, tok); A = TT.attach_gen(model, tok, MS); E = model.get_input_embeddings()
def drive(name, ctx, maxn=20):
    M = A.nets[name]; I = A.all_ids[name]; outs = SPECS[name]['out']; ot = [first(I[s]) for s in outs]
    ids = tok(ctx, add_special_tokens=False)['input_ids']; out = []
    with torch.no_grad():
        e = M.prime_sym(M.symbols(E(torch.tensor([ids], device='cuda'))))[:, -1]
        for _ in range(maxn):
            if e.max() < 0.5: break
            j = int(e.argmax()); out.append(ot[j])
            if outs[j] == 'OE': break
            e = M.step_sym(M.symbols(E(torch.tensor([out[-1]], device='cuda'))))[:, 0]
    return tok.decode(out)
r = random.Random(seed); C = {'ok': 0, 'bad': 0}
cases = [CF.random_call(r) for _ in range(N)] + CF.EDGE_CASES + ['calc $4.50 * 3', 'calc (2 + 3) * 1.5', 'calc x * 2', 'calc 2.5 * 4 apples']
for args in cases:
    ctx = 'Some text before. ' + CF.CS + args + CF.CE
    got = {m: drive(m, ctx) for m in MS}
    fired = {m: g for m, g in got.items() if g}
    want = CF.OS + CF.run_ref(args) + CF.OE
    name = args.split(' ')[0]
    okk = (list(fired) == [name + '_call'] and fired[name + '_call'] == want) if name in ('expr', 'bool', 'calc') else not fired
    C['ok' if okk else 'bad'] += 1
    if not okk or args in CF.EDGE_CASES: print('OK ' if okk else 'BAD', repr(args), fired, 'want', repr(want))
print(C)
