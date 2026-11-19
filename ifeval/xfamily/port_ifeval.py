"""Port the FROZEN Team-D IFEval implants (frozen_orig/, md5 = ifeval/heldout/frozen_MD5.txt) to another chat host.
Every edit is a mechanical replacement of a Qwen-specific constant by a host-config value; no rule logic, word list,
detector, threshold or table definition changes. python port_ifeval.py SRC_DIR DST_DIR
Edits:
  common.py  MODEL / EOS_IDS -> host config H (env IFE_MODEL); vocab_strings: V and the special-token set from the tokenizer
  tables.py  'i >= 151643' (Qwen special-token boundary) -> 'i in SPEC' (the tokenizer's special/added ids; for Qwen this is
             exactly ids >= 151643); IMSTART/IMEND ids from H
  circuit.py role ids (<|im_start|>, assistant, user, system, '\\n', ' '), EOS ids from H; answer-start rule: Qwen
             '</think>' '\\n\\n'  ->  ChatML '<|im_start|>' + assistant-role pieces + '\\n' for hosts without a think block
"""
import sys, os, re, shutil
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)

COMMON = r'''import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- host config (xfamily port). ans: 'think' = Qwen3 thinking-off template ('</think>' '\n\n' starts the answer);
# 'chatml' = '<|im_start|>' + assistant role pieces + '\n' starts the answer.
HOSTS = {
    'Qwen/Qwen3-0.6B': dict(eos=[151645, 151643], im_start=151644, im_end=151645, asst=[77091], user=872, system=8948,
                            nl=198, space=220, ans='think', think_end=151668, nn=271, V=151936),
    'HuggingFaceTB/SmolLM2-1.7B-Instruct': dict(eos=[2, 0], im_start=1, im_end=2, asst=[520, 9531], user=4093, system=9690,
                                                nl=198, space=216, ans='chatml', V=49152),
}
MODEL = os.environ.get('IFE_MODEL', 'HuggingFaceTB/SmolLM2-1.7B-Instruct')
H = HOSTS[MODEL]
EOS_IDS = H['eos']          # [end-of-turn, end-of-text]


def special_ids(tok):
    return set(tok.all_special_ids) | set(tok.get_added_vocab().values())


def load(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda().eval()
    for p in model.parameters(): p.requires_grad_(False)
    assert model.lm_head.weight.shape[0] == H['V'], (model.lm_head.weight.shape, H['V'])
    return tok, model


def chat(tok, prompt, msgs=None):
    msgs = (msgs or []) + [{'role': 'user', 'content': prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def vocab_strings(tok, path='vocab_strs.json'):
    if os.path.exists(path): return json.load(open(path))
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    bd = {v: k for k, v in bytes_to_unicode().items()}
    V = H['V']; toks = tok.convert_ids_to_tokens(list(range(len(tok)))); SP = special_ids(tok)
    s = []
    for i in range(V):
        t = toks[i] if i < len(toks) else None
        if t is None: s.append(''); continue
        if i in SP: s.append(t); continue                      # special tokens
        s.append(bytes([bd[c] for c in t]).decode('utf-8', errors='replace'))
    json.dump(s, open(path, 'w')); return s


def token_sets(strs, device='cuda'):
    m = lambda f: torch.tensor([f(s) for s in strs], device=device)
    return {
        'comma': m(lambda s: ',' in s),
        'upper': m(lambda s: s != s.lower()),
        'lower': m(lambda s: s != s.upper()),
        'eos': m(lambda s: False) | torch.isin(torch.arange(len(strs), device=device), torch.tensor(EOS_IDS, device=device)),
        'nn': m(lambda s: '\n\n' in s),
        'star3': m(lambda s: '***' in s),
    }
'''
open(f'{dst}/common.py', 'w').write(COMMON)

# ---------------- tables.py
t = open(f'{src}/tables.py').read(); n0 = t
t = t.replace("def build(tok, strs, device='cuda', cache='tables.pt'):",
              "def build(tok, strs, device='cuda', cache='tables.pt'):\n    from common import special_ids, H as _H\n    SPEC = special_ids(tok) | set(range(len(tok), len(strs)))")
reps = [
    ("if i >= 151643: continue", "if i in SPEC: continue"),
    ("if i < 151643 else 0.", "if i not in SPEC else 0."),
    ("enumerate(strs[:151643])", "((i, s) for i, s in enumerate(strs) if i not in SPEC)"),
    ("range(151643)", "(i for i in range(len(strs)) if i not in SPEC)"),
    ("tabs['IMSTART'][151644] = 1; tabs['IMEND'][151645] = 1", "tabs['IMSTART'][_H['im_start']] = 1; tabs['IMEND'][_H['im_end']] = 1"),
]
for a, b in reps:
    assert a in t, a; t = t.replace(a, b)
assert '1516' not in t, [l for l in t.splitlines() if '1516' in l]
open(f'{dst}/tables.py', 'w').write(t)

# ---------------- circuit.py
c = open(f'{src}/circuit.py').read()
creps = [
    ("EOS = 151645", "from common import H\nEOS = H['im_end']"),
    ("self.EOSMASK[[151645, 151643]] = 1", "self.EOSMASK[H['eos']] = 1"),
    ("is_role_tok = (self.prevtok == 151644).float()", "is_role_tok = (self.prevtok == H['im_start']).float()"),
    ("(ids == 77091).float()", "(ids == H['asst'][0]).float()"),
    ("# 77091 = 'assistant'", "# first piece of the assistant role"),
    ("(self.win[:, -3] == 151644) & (ids == 198) & (self.asst < 0.5)", "(self.win[:, -3] == H['im_start']) & (ids == H['nl']) & (self.asst < 0.5)"),
    ("ans_start = ((self.win[:, -2] == 151668) & (ids == 271) & (self.asst > 0.5)).float()   # '</think>' '\\n\\n'",
     "if H['ans'] == 'think':\n            ans_start = ((self.win[:, -2] == H['think_end']) & (ids == H['nn']) & (self.asst > 0.5)).float()   # '</think>' '\\n\\n'\n"
     "        else:                                    # ChatML: '<|im_start|>' + assistant role pieces + '\\n'\n"
     "            na = len(H['asst']); okk = (self.win[:, -(na + 2)] == H['im_start']) & (ids == H['nl']) & (self.asst > 0.5)\n"
     "            for k_, a_ in enumerate(H['asst']): okk = okk & (self.win[:, -(na + 1) + k_] == a_)\n"
     "            ans_start = okk.float()"),
    ("((self.prevtok == 151644) & (ids == 8948))", "((self.prevtok == H['im_start']) & (ids == H['system']))"),
    ("((self.prevtok == 151644) & (ids == 872))", "((self.prevtok == H['im_start']) & (ids == H['user']))"),
    ("endsep = torch.where(g['pn'] > 0.5, torch.full_like(nn, 220), nn)", "endsep = torch.where(g['pn'] > 0.5, torch.full_like(nn, H['space']), nn)"),
    ("hsep = torch.where(g['pn'] > 0.5, torch.full_like(nn, 220), nn)", "hsep = torch.where(g['pn'] > 0.5, torch.full_like(nn, H['space']), nn)"),
]
for a, b in creps:
    assert a in c, a; c = c.replace(a, b)
left = re.findall(r'\b(1516\d\d|77091|8948|872|220|198|271)\b', c)
assert not left, left
open(f'{dst}/circuit.py', 'w').write(c)
shutil.copy(f'{src}/run.py', f'{dst}/run.py')
print('ported', src, '->', dst)
