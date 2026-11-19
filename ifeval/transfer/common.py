import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("WSM_MODEL", "Qwen/Qwen3-0.6B")
EOS_IDS = [151645, 151643]          # <|im_end|>, <|endoftext|>


def load(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda().eval()
    for p in model.parameters(): p.requires_grad_(False)
    return tok, model


def chat(tok, prompt, msgs=None):
    msgs = (msgs or []) + [{'role': 'user', 'content': prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def vocab_strings(tok, path='vocab_strs.json'):
    if os.path.exists(path): return json.load(open(path))
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    bd = {v: k for k, v in bytes_to_unicode().items()}
    V = 151936; toks = tok.convert_ids_to_tokens(list(range(len(tok))))
    s = []
    for i in range(V):
        t = toks[i] if i < len(toks) else None
        if t is None: s.append(''); continue
        if i >= 151643: s.append(t); continue                      # special tokens
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
