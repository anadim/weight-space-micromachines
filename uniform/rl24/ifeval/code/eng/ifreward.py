"""IFEval reward / scorer = lm_eval 0.4.13 IFEval checkers, prompt-level strict (the seeded scorer of transfer/reduce_seeded.py:
langdetect seed 0, random.seed(key) per prompt, <think> blocks removed, stop tokens removed). lm_eval lives in $PYL
(pip --target, no deps): appended to sys.path (never shadows the venv) and, if the package __init__s pull optional
dependencies, lm_eval / lm_eval.tasks are registered as bare packages (score_reduce.py), so only the checker files load."""
import importlib, importlib.util, os, random, re, sys, types
_U = None


def _load():
    global _U
    if _U is not None: return _U
    pyl = os.environ.get('PYL')
    if pyl and pyl not in sys.path: sys.path.append(pyl)
    try:
        importlib.import_module('lm_eval.tasks.ifeval.utils')
    except Exception:
        for k in [k for k in sys.modules if k == 'lm_eval' or k.startswith('lm_eval.')]: del sys.modules[k]
        root = os.path.dirname(importlib.util.find_spec('lm_eval').origin)
        for name, path in [('lm_eval', root), ('lm_eval.tasks', os.path.join(root, 'tasks'))]:
            m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
        importlib.import_module('lm_eval.tasks.ifeval.utils')
    import lm_eval.tasks.ifeval.utils as U
    import langdetect; langdetect.DetectorFactory.seed = 0
    _U = U
    return U


def clean(text, fmt='chat'):
    text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    if fmt == 'plain' and '\nInstruction:' in text: text = text.split('\nInstruction:')[0]
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def score(doc, text, fmt='chat'):
    """-> (prompt_strict 0/1, [inst strict], [inst loose], prompt_loose 0/1)"""
    U = _load()
    import langdetect; langdetect.DetectorFactory.seed = 0
    random.seed(int(doc['key']))
    d = {k: doc[k] for k in ('key', 'prompt', 'instruction_id_list', 'kwargs')}
    try:
        o = U.process_results(d, [clean(text, fmt)])
        return (float(o['prompt_level_strict_acc']), [bool(x) for x in o['inst_level_strict_acc']],
                [bool(x) for x in o['inst_level_loose_acc']], float(o['prompt_level_loose_acc']))
    except Exception:
        n = len(doc['instruction_id_list']); return 0.0, [False] * n, [False] * n, 0.0


def degenerate(text, fmt='chat'):
    """task [31](b) candidate early-zero rule (measured only, see degen.py; NOT used by the running arms):
    'noletter' if the cleaned output has no letter (str.isalpha, any script), 'repeat' if it is one unit repeated
    (whitespace-collapsed text = a period-p string, p <= 32, >= 4 periods; or >= 4 words, all the same word), else ''."""
    t = clean(text, fmt)
    if not any(c.isalpha() for c in t): return 'noletter'
    w = t.split()
    if len(w) >= 4 and len(set(w)) == 1: return 'repeat'
    s = ' '.join(w); n = len(s)
    for p in range(1, 33):
        if n < 4 * p: break
        if s[p:] == s[:n - p]: return 'repeat'
    return ''


def _job(args):
    doc, text, fmt = args
    return score(doc, text, fmt)


class Pool:
    """process pool (fork) for batch scoring; falls back to serial."""
    def __init__(self, n=16):
        import multiprocessing as mp
        _load()
        self.p = mp.get_context('fork').Pool(n) if n > 1 else None

    def map(self, docs, texts, fmt='chat'):
        args = list(zip(docs, texts, [fmt] * len(docs)))
        return self.p.map(_job, args, chunksize=4) if self.p is not None else [_job(x) for x in args]
