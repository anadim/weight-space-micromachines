"""Tool-alone exactness tests: the compiled circuits (no host model) against the Python references in callfmt.py.

Token ids come from the real Qwen3 tokenizer; each id is mapped to the tool's input symbol by its TEXT table (exactly what
the reader computes from the embeddings, which is checked separately when attaching). The reply is generated token by
token by the circuit itself (argmax of its outputs, fed back), then compared with the reference.

TOOL_DEVICE=cpu python test_tools.py [n_random]
"""
import sys, random, time, torch, pickle, os
import register  # noqa: F401
from micromachines.specs import SPECS
from micromachines.machine import Micromachine
from transformers import AutoTokenizer
from callfmt import CS, CE, OS, OE, track_ref, order_ref, ORDER_OPS

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
dev = os.environ.get('TOOL_DEVICE', 'cpu')


class ToolAlone:
    def __init__(self, name):
        sp = SPECS[name]; self.name = name
        self.m = Micromachine(name).to(dev)
        self.syms, self.out = sp['syms'], sp['out']
        self.id2sym = {}
        for s, t in sp['text'].items():
            ids = tok(t, add_special_tokens=False)['input_ids']; assert len(ids) == 1, (s, t, ids)
            if s in self.syms: self.id2sym[ids[0]] = self.syms.index(s)
        self.out_ids = [tok(sp['text'][o], add_special_tokens=False)['input_ids'][0] for o in self.out]

    def flags(self, ids):
        x = torch.zeros(len(ids), len(self.syms))
        for t, i in enumerate(ids):
            if i in self.id2sym: x[t, self.id2sym[i]] = 1
        return x.to(dev)

    @torch.no_grad()
    def run(self, text, max_out=12):
        """text ends with </tool_call>. -> (reply string or None if the tool stayed silent at </tool_call>, raw outputs)"""
        ids = tok(text, add_special_tokens=False)['input_ids']
        e = self.m.prime_sym(self.flags(ids)[None])[0]
        last = e[-1]
        silent_before = float(e[:-1].abs().sum())               # the tool must write nothing before </tool_call>
        if last.abs().sum() == 0: return None, silent_before
        reply = []
        for _ in range(max_out):
            assert torch.all((last == 0) | (last == 1)) and last.sum() == 1, last
            nid = self.out_ids[int(last.argmax())]
            reply.append(nid)
            if nid == tok.convert_tokens_to_ids(OE): break
            last = self.m.step_sym(self.flags([nid]))[0, 0]
        s = tok.decode(reply)
        assert s.startswith(OS) and s.endswith(OE), s
        return s[len(OS):-len(OE)], silent_before


def rand_track(r):
    n = r.randint(1, 9); st = ''.join(r.choice('0123456789') for _ in range(n)) if r.random() < .5 else ''.join(str(i) for i in range(1, n + 1))
    ops = []
    for _ in range(r.choice([0, 1, 2, 3, 5, 7, 9, 12, 20])):
        t = r.random(); i = r.randint(1, n); j = r.randint(1, n)
        if t < .6: ops.append(f'{i}{j}')
        elif t < .8: ops.append(f'{i}={r.randint(0, 9)}')
        else: ops.append(f'{i}<{j}')
    q = r.choice(['', f'?{r.randint(1, n)}', f'@{r.randint(0, 9)}', f'?{r.randint(1, 9)}'])
    return f'track {st}:' + ','.join(ops) + q


def rand_order(r):
    n = r.randint(2, 7); p = list(range(1, n + 1)); r.shuffle(p)
    cons = []
    for _ in range(r.randint(0, 12)):
        op = r.choice(ORDER_OPS); a = r.randint(1, n); b = r.randint(1, n)
        pa = p.index(a) + 1
        if r.random() < .8:                                        # mostly true constraints, so solutions exist
            if op in ('<', '>', '+', '~'):
                b = r.choice([x for x in range(1, n + 1) if x != a]); pb = p.index(b) + 1
                op = {True: '<', False: '>'}[pa < pb] if op in '<>' else ('+' if pb == pa + 1 else ('~' if abs(pa - pb) == 1 else ('<' if pa < pb else '>')))
            elif op == '=': b = pa
            elif op == '=-': b = n + 1 - pa
            elif op == '!=': b = r.choice([x for x in range(1, n + 1) if x != pa] or [1])
            else: b = r.choice([x for x in range(1, n + 1) if x != n + 1 - pa] or [1])
        cons.append(f'{a}{op}{b}')
    q = r.choice(['', f'?{r.randint(1, n)}', f'?-{r.randint(1, n)}', f'@{r.randint(1, n)}', f'?{r.randint(1, 7)}'])
    return f'order {n}:' + ','.join(cons) + q


MALFORMED = {
    'track': ['track 123', 'track :12', 'track 123:12,', 'track 123:12,,23', 'track 123:14', 'track 123:11', 'track 123:4=1', 'track 123:1<1',
              'track 1234567890:12', 'track 123:12?4', 'track 123:12?', 'track 123:@7', 'track 112:@1', 'track 123: 12', 'track123:12',
              'track 123:12 ?1', 'track 123:1=12', 'track 123:12x', 'order 3:1<2', 'track 12a:12', 'track 123;12'],
    'order': ['order 3', 'order 3:', 'order 8:1<2', 'order 1:', 'order 3:1<1', 'order 3:1<4', 'order 3:1=4?1', 'order 3:1<2,', 'order 3:1<2?4',
              'order 3:1<2?1', 'order 3:1<2,2<1', 'order 3:12', 'order 3: 1<2', 'order 3:1<2?', 'order 3:1<<2', 'track 3:1<2', 'order 3:0<1',
              'order 3:1<9', 'order 3:1 < 2', 'order 35:1<2'],
}


def check(tool, calls, ref, label):
    bad = 0; t0 = time.time(); nonempty = 0
    for c in calls:
        want = ref(c) if c.startswith(tool.name.split('_')[0] + ' ') else None     # a call naming another tool: silent
        for prefix in ('', 'Some text before. 12, 3: ', f'{CS}order 3:1<2{CE}{OS}1{OE} and {CS}track 12:12{CE}{OS}21{OE} then '):
            got, leak = tool.run(prefix + CS + c + CE)
            ok = (got == want or (want == '' and got is None)) and (leak == 0 or CE in prefix)   # silent = fail-safe; earlier calls in the prefix are answered
            if not ok:
                bad += 1
                if bad <= 10: print('  FAIL', label, repr(prefix[:20]), c, 'want', repr(want), 'got', repr(got), 'leak', leak)
        nonempty += bool(want)
    print(f'{label}: {len(calls)} calls x 3 contexts, {bad} failures ({nonempty} with a non-empty reference reply) {time.time() - t0:.0f}s', flush=True)
    return bad


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    r = random.Random(12345)
    tot = 0
    for name, gen, ref in (('track_gen', rand_track, track_ref), ('order_gen', rand_order, order_ref)):
        t0 = time.time(); tool = ToolAlone(name)
        print(f'{name}: built in {time.time() - t0:.0f}s; stack1 units {tool.m.stack1.n_units} nnz {tool.m.stack1.nnz}, stack2 units {tool.m.stack2.n_units}', flush=True)
        key = name.split('_')[0]
        tot += check(tool, MALFORMED[key], ref, name + ' malformed')
        tot += check(tool, [gen(r) for _ in range(N)], ref, name + ' random')
        # silence: ordinary text with the tool's own symbols but no call
        for txt in ['order 3:1<2?1 is what I would write', 'track 123:12?1</tool_response>', 'Options: (A) 1 (B) 2. ' * 5]:
            ids = tok(txt, add_special_tokens=False)['input_ids']; e = tool.m.prime_sym(tool.flags(ids)[None])[0]
            if e.abs().sum() != 0: print('  NOT SILENT on', repr(txt)); tot += 1
    print('TOTAL FAILURES', tot)
