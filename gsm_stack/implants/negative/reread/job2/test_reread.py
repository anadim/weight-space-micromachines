"""CPU test of the forced re-read writer queue (job/stk/serial_stack.py: SerialEngine.rr_ids / rr_fill, lifted with ast).

For every dev/dev2 question (gen/out base rows or STK items) and both variants (full, ask), and both newline forms:
  1. the queue = tokens(RR_PHRASE) [or tokens('\\n' + RR_PHRASE)] + a contiguous slice of the row's 'Question: q\\nAnswer:' ids
     (exact copy; 'ask' may prepend one ' ' token if the ask does not start with a space token);
  2. full: decode(queue) == phrase + ' ' + q.strip() + '\\n'  (text-exact);  ask: decode ends with the question's tail + '\\n';
  3. len(queue) <= QL, and rr_fill writes the rowid-indexed buffers (eos padded) with the right lengths.
usage: python3 test_reread.py QUESTIONS_JSONL_GZ...   (needs torch + a Qwen tokenizer: transformers or tokenizers)
"""
import ast, gzip, json, os, sys
import torch

SRC = next(p for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), d, 'stk', 'serial_stack.py') for d in ('job', '.')] if os.path.exists(p))


class Tok:
    def __init__(self):
        try:
            from transformers import AutoTokenizer
            self.t = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base'); self.hf = True
        except Exception:
            from tokenizers import Tokenizer
            import glob
            f = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*/tokenizer.json'))[0]
            self.t = Tokenizer.from_file(f); self.hf = False

    def __call__(self, s, add_special_tokens=False):
        return {'input_ids': self.t(s, add_special_tokens=False)['input_ids'] if self.hf else self.t.encode(s, add_special_tokens=False).ids}

    def decode(self, ids):
        return self.t.decode(ids)


def lift(rr):
    tree = ast.parse(open(SRC).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SerialEngine')
    meth = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('rr_ids', 'rr_fill')]
    assert len(meth) == 2
    QL = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'QL')
    ph = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('RR_PHRASE', 'WAIT')]
    ns = {'torch': torch, 'RR': rr}
    holder = ast.ClassDef(name='Eng', bases=[], keywords=[], body=meth, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=ph + [QL, holder], type_ignores=[])), SRC, 'exec'), ns)
    return ns


def main(files):
    tok = Tok(); T_ = lambda s: tok(s)['input_ids']
    qs = []
    for f in files:
        qs += [json.loads(l)['q'] for l in gzip.open(f, 'rt')]
    nfail = 0
    for rr in ('full', 'ask', 'qa'):
        ns = lift(rr); QL = ns['QL']; ph = ns['RR_PHRASE']
        e = ns['Eng'](); e.tok = tok; e.eos = 151643; e.dev = 'cpu'
        e.rr_p = [T_(ph), T_('\n' + ph)]; e.rr_w = [T_(ns['WAIT']), T_('\n' + ns['WAIT'])]; e.rr_nl = T_('\n'); e.rr_hd = T_('Question:'); e.rr_tl = T_('Answer:')
        Bp = 129; e.rr_q = [torch.full((Bp, QL), e.eos, dtype=torch.long) for _ in range(2)]; e.rr_l = [torch.zeros(Bp, dtype=torch.long) for _ in range(2)]
        lens, ex = [], []
        for q in qs:
            qi = T_(f"Question: {q}\nAnswer:")
            out = e.rr_ids(qi)
            for k, ids in enumerate(out):
                if rr == 'qa':
                    ok = ids == e.rr_w[k] + qi and len(ids) <= QL and tok.decode(ids) == ('\n' if k else '') + ns['WAIT'] + f"Question: {q}\nAnswer:"
                    nfail += not ok; lens.append(len(ids)); continue
                p = e.rr_p[k]; body = ids[len(p):]
                ok = ids[:len(p)] == p and len(ids) <= QL
                core = body[1:] if (rr == 'ask' and tok.decode(body[:1]) == ' ' and body[:1] == T_(' ')) else body
                s = ''.join(map(str, qi))
                ok &= any(qi[i:i + len(core)] == core for i in range(len(qi) - len(core) + 1)) or core[-1:] == e.rr_nl
                txt = tok.decode(ids)
                if rr == 'full': ok &= txt == ('\n' if k else '') + ph + ' ' + q.strip() + '\n'
                else: ok &= txt.endswith(q.strip()[-12:] + '\n') and txt.startswith(('\n' if k else '') + ph + ' ')
                nfail += not ok
                if not ok and nfail < 5: print('FAIL', rr, k, repr(txt[:300]))
            lens.append(len(out[1]))
            if rr == 'ask' and len(ex) < 6: ex.append(tok.decode(out[0]))
        # rr_fill (G=1 and G=2)
        for G in (1, 2):
            sub = [T_(f"Question: {q}\nAnswer:") for q in qs[:128 // G]]
            e.rr_fill(sub, G)
            for p, qi in enumerate(sub):
                for g in range(G):
                    for k, ids in enumerate(e.rr_ids(qi)):
                        j = p * G + g
                        nfail += not (e.rr_l[k][j] == len(ids) and e.rr_q[k][j, :len(ids)].tolist() == ids and (e.rr_q[k][j, len(ids):] == e.eos).all())
        print(rr, 'n', len(qs), 'queue len max', max(lens), 'mean %.1f' % (sum(lens) / len(lens)), 'QL', QL)
        for x in ex: print('  ask:', repr(x))
    print('PASS' if nfail == 0 else f'FAIL {nfail}')
    return nfail


if __name__ == '__main__':
    sys.exit(1 if main(sys.argv[1:]) else 0)
