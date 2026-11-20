"""Mixed-task env for train_library.py --task mixed: chat format (Qwen3 template, thinking off), shared system prompt
(= the engine's shared prefix), per-type programmatic rewards.
  gsm   : last 'answer is X' number == gold (float, 1e-6 rel)
  bbh   : last 'answer is X' == gold under the team harness norms (envs.normA/B/C), case-insensitive extraction
  ifeval: lm_eval IFEval checker, prompt-level strict
  combo : base part correct AND the format constraint passes (lm_eval checker on its one instruction)
"""
import os, sys, re, json
HERE = os.path.dirname(os.path.abspath(__file__))
# The answer-format line lives in the SYSTEM prompt: in the user turn ("finish with a line of the form 'The answer is X.'")
# it is a genuine end-phrase instruction and the IFEval circuit latched its 'end' rule on 100 % of prompts (if_latch_probe.py).
SYS = "You are a careful assistant. For math and reasoning questions, think step by step; your last sentence states the result as: The answer is X."
FIN = "\nThink step by step, then finish with a line of the form 'The answer is X.'"


def user_text(row):
    return row['user'].replace(FIN, '')
_U = None


def _checker():
    global _U
    if _U is None:
        from lm_eval.tasks.ifeval import utils as U
        import langdetect; langdetect.DetectorFactory.seed = 0
        _U = U
    return _U


def prefix_and_qfn(tok):
    sys_txt = tok.apply_chat_template([{'role': 'system', 'content': SYS}], tokenize=False)
    pre = tok(sys_txt, add_special_tokens=False)['input_ids']

    def q_ids(row):
        full = tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_text(row)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert full.startswith(sys_txt)
        return tok(full[len(sys_txt):], add_special_tokens=False)['input_ids']
    return pre, q_ids


def load(split):
    suf = os.environ.get('MIX_SUFFIX', '')                  # e.g. _gb = gsm + bbh only
    rows = [json.loads(l) for l in open(os.path.join(HERE, f'{split}{suf}.jsonl'))]
    types = os.environ.get('MIX_TYPES', '')                 # e.g. bbh (debug / sweeps)
    return [r for r in rows if not types or r['type'] in types.split(',')]


def _ans(text):
    m = re.findall(r'answer is[:\s]*(.*?)(?:\.\s*$|\.\s*\n|\n|$|")', text, flags=re.I | re.M)
    return m[-1].strip().strip('*').strip() if m else ''


def _num(x):
    x = x.replace(',', '').replace('$', '')
    m = re.search(r'-?\d+(?:\.\d+)?', x)
    return float(m.group()) if m else None


def base_ok(text, row):
    a = _ans(text)
    if row.get('fam') is None:                            # gsm
        v = _num(a); g = float(row['gold'])
        return float(v is not None and abs(v - g) <= 1e-6 * max(1.0, abs(g)))
    sys.path.insert(0, os.environ.get('RLENVS', os.path.dirname(os.path.abspath(__file__))))
    import envs as EV
    fam, task, gold = row['fam'], row['task'], row['gold']
    if fam == 'A': return float(EV.normA(a) == EV.normA(gold))
    if fam == 'B': return float(EV.normB(a, task) == EV.normB(gold, task))
    return float(EV.normC(a, task) == EV.normC(gold, task))


def if_ok(text, row):
    U = _checker()
    import langdetect; langdetect.DetectorFactory.seed = 0
    doc = {'key': row.get('key', 0), 'instruction_id_list': row['instruction_id_list'], 'prompt': user_text(row), 'kwargs': row['kwargs']}
    try:
        return float(U.process_results(doc, [text])['prompt_level_strict_acc'])
    except Exception:
        return 0.0


def score(text, row):
    """-> (reward, base_part, format_part) ; parts are None when not applicable"""
    text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    t = row['type']
    if t in ('gsm', 'bbh'):
        b = base_ok(text, row); return b, b, None
    if t == 'ifeval':
        f = if_ok(text, row); return f, None, f
    b = base_ok(text, row); f = if_ok(text, row)
    return b * f, b, f


import re as _re
_ANN = _re.compile(r'((?:[\d.$,()]+\s*[-+*/x\u00d7]\s*)*[\d.$,()]+)?\s*=\s*\$?<<([^=>]*)=([^>]*)>>')
_OP = _re.compile(r'^\s*(-?[\d.]+)\s*([-+*/])\s*(-?[\d.]+)\s*$')
def calc_env_line(ans):
    """GSM8K gold 'sold 48/2 = <<48/2=24>>24 clips' -> 'sold <calc>48 / 2</calc> 24 clips' when the annotation is ONE
    binary op on plain numbers; otherwise the annotation is just removed (plain text kept)."""
    def sub(m):
        m2 = _OP.match(m.group(2))
        if not m2: return (m.group(1) or '') + ' = '
        return f'<calc>{m2.group(1)} {m2.group(2)} {m2.group(3)}</calc> '
    out = _ANN.sub(sub, ans)
    out = _re.sub(r'<calc>([^<]*)</calc> (\S+)', lambda m: f'<calc>{m.group(1)}</calc> {m.group(2)}', out)
    return _re.sub(r'<<[^>]*>>', '', out)
