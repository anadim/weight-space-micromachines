"""MM12 'eyes': give Qwen sight of pasted digit images.

An image in the prompt is <|img|> + 196 digit tokens (MNIST, 2x2 average-pooled to 14x14, gray level quantized to 0..9,
row-major, no separators: Qwen tokenizes every digit on its own) + <|/img|>.
The model asks about image k with <|see|>k<|/see|>; the machine answers <|see_out|>D<|/see_out|> with the digit class.

Inside the machine (tensors only, no branching on data):
  reader (exact, linear, like every machine)  -> digit value per token + image/call flags
  at every image-end token: a small CNN on the 196 preceding values -> class one-hot (steep softmax), gated by a format check
  image ordinal = cumulative count of image ends; slot memory M[k] = class of the k-th image (a cumulative sum)
  at a call: the index digit k selects M[k] (a dot product) -> one-hot over the output tokens -> the usual write D.
The CNN is the one TRAINED part (on MNIST train; questions for dev/test use MNIST test images it never saw).
python eyes.py cnn          train the CNN  -> runs/eye_cnn.pt
python eyes.py data N       questions      -> data/eyes_train.jsonl (N), data/eyes_dev.jsonl (300), data/eyes_test.jsonl (500)"""
import sys, json, random, torch, torch.nn as nn, torch.nn.functional as Fn
from calcfmt import calc_ref, CS, CE, OS, OE

IMG = ('<|img|>', '<|/img|>')
EY = ('<|see|>', '<|/see|>', '<|see_out|>', '<|/see_out|>')
SYMS = list('0123456789') + ['IS', 'IE', 'CS', 'CE', 'OS', 'OE']
TEXT = {**{c: c for c in '0123456789'}, 'IS': IMG[0], 'IE': IMG[1], 'CS': EY[0], 'CE': EY[1], 'OS': EY[2], 'OE': EY[3]}
OUT = list('0123456789') + ['OS', 'OE']
NPIX, SIDE, NSLOT = 196, 14, 10
SI = {s: i for i, s in enumerate(SYMS)}


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)
        self.f1, self.f2 = nn.Linear(64 * 7 * 7, 128), nn.Linear(128, 10)

    def forward(self, x):                    # x (N, 196) gray levels 0..9
        x = (x / 9.0).view(-1, 1, SIDE, SIDE)
        x = Fn.max_pool2d(Fn.relu(self.c2(Fn.relu(self.c1(x)))), 2)
        return self.f2(Fn.relu(self.f1(x.flatten(1))))


def mnist():
    from datasets import load_dataset
    import numpy as np
    out = {}
    for split in ('train', 'test'):
        d = load_dataset('ylecun/mnist', split=split)
        X = torch.tensor(np.stack([np.array(im, dtype=np.float32) for im in d['image']]))
        X = Fn.avg_pool2d(X[:, None], 2)[:, 0].flatten(1)                        # 14x14
        out[split] = (torch.round(X / 255 * 9).long(), torch.tensor(d['label']))
    return out


class EyeMachine(nn.Module):
    def __init__(self, reader_w, reader_b, out_dirs, cnn, gain=1e4, beta=60.0, chunk=4096):
        super().__init__()
        self.register_buffer('reader_w', reader_w); self.register_buffer('reader_b', reader_b); self.register_buffer('out_dirs', out_dirs)
        self.cnn, self.gain, self.beta, self.chunk = cnn, gain, beta, chunk
        self.K = NPIX + 2

    def symbols(self, emb):
        return torch.clamp(Fn.linear(emb.float(), self.reader_w, self.reader_b), 0, 1)

    def _classes(self, win):                 # win (N, K, n_sym), offset 0 = newest -> (N,10) class one-hot, gated
        val = sum(d * win[..., d] for d in range(1, 10))                           # (N, K)
        isdig = win[..., :10].sum(-1)
        pix = val[:, 1:NPIX + 1].flip(1)                                           # the 196 tokens before the end, in order
        ok = win[:, 0, SI['IE']] * win[:, NPIX + 1, SI['IS']] * isdig[:, 1:NPIX + 1].prod(-1)
        outs = []
        for s in range(0, pix.shape[0], self.chunk):                               # fixed chunks (memory), not data-dependent
            outs.append(torch.softmax(self.beta * self.cnn(pix[s:s + self.chunk]).float(), -1))
        return torch.cat(outs) * ok[:, None], ok

    def _emit(self, last4, M):              # last4 (B,4,n_sym) offset 0 = newest; M (B,NSLOT,10) -> (B, len(OUT))
        s0, s1, s2, s3 = last4[:, 0], last4[:, 1], last4[:, 2], last4[:, 3]
        e = last4.new_zeros(last4.shape[0], len(OUT))
        e[:, OUT.index('OS')] = s0[:, SI['CE']] * s2[:, SI['CS']] * s1[:, 1:10].sum(-1)              # CS k CE -> open the box
        gate = s0[:, SI['OS']] * s1[:, SI['CE']] * s3[:, SI['CS']]                                   # CS k CE OS -> the class
        ans = torch.einsum('bk,bkc->bc', s2[:, 1:NSLOT], M[:, 1:NSLOT]) * gate[:, None]              # index 1..9
        e[:, :10] = ans
        found = ans.sum(-1)
        e[:, OUT.index('OE')] = gate * (1 - found) + s1[:, SI['OS']] * s2[:, SI['CE']] * s0[:, :10].sum(-1)
        return e

    def _windows(self, sym):                 # (B,T,n) -> (B,T,K,n), offset 0 = current
        pad = Fn.pad(sym, (0, 0, self.K - 1, 0))
        return pad.unfold(1, self.K, 1).permute(0, 1, 3, 2).flip(2)

    @torch.no_grad()
    def prime(self, emb, full=False, ids=None):
        sym = self.symbols(emb); B, T, n = sym.shape
        win = self._windows(sym)
        cls, ok = self._classes(win.reshape(B * T, self.K, n)); cls, ok = cls.view(B, T, 10), ok.view(B, T)
        ordn = torch.cumsum(ok, 1) * ok                                             # k for an image end, else 0
        slot = Fn.one_hot(ordn.round().long().clamp(0, NSLOT - 1), NSLOT).float() * ok[..., None]
        Mt = torch.cumsum(slot[..., :, None] * cls[..., None, :], 1)               # (B,T,NSLOT,10)
        last4 = win[:, :, :4]
        e = self._emit(last4.reshape(B * T, 4, n), Mt.reshape(B * T, NSLOT, 10)).view(B, T, -1)
        self.state = (sym[:, -self.K:] if T >= self.K else Fn.pad(sym, (0, 0, self.K - T, 0)), Mt[:, -1], torch.cumsum(ok, 1)[:, -1])
        return (e @ self.out_dirs) * self.gain

    @torch.no_grad()
    def step(self, emb, ids=None):
        s = self.symbols(emb); buf, M, cnt = self.state
        buf = torch.cat([buf[:, 1:], s[:, None]], 1)
        win = buf.flip(1)                                                          # offset 0 = newest
        cls, ok = self._classes(win)
        M = M + Fn.one_hot((cnt + ok).round().long().clamp(0, NSLOT - 1), NSLOT).float()[..., None] * (cls * ok[:, None])[:, None, :]
        cnt = cnt + ok
        self.state = (buf, M, cnt)
        e = self._emit(win[:, :4], M)
        # (B,1,d) like every other machine: a (B,d) write broadcasts against the host's (B,1,d) state into (B,B,d) and every
        # sample reads the LAST sample's write (bug in the first eyes run: batch-1 tests cannot see it)
        return ((e @ self.out_dirs) * self.gain)[:, None, :], e


# ---------------------------------------------------------------- questions
def img_text(px): return IMG[0] + ''.join(str(int(v)) for v in px) + IMG[1]


EXPR = [('a+b', 2), ('a-b', 2), ('a*b', 2), ('a+b+c', 3), ('a+b*c', 3), ('a*b+c', 3), ('(a+b)*c', 3), ('a*b-c', 3), ('a+b+c+d', 4), ('a*b+c*d', 4)]
WORD = {'a': 'image 1', 'b': 'image 2', 'c': 'image 3', 'd': 'image 4'}


def steps(expr, v):
    """Evaluation order with standard precedence -> list of (call, result), final value. Operands are digits/values."""
    import ast
    calls = []
    def ev(n):
        if isinstance(n, ast.Name): return str(v[n.id])
        l, r = ev(n.left), ev(n.right); op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*'}[type(n.op)]
        e = f'{l}{op}{r}'; o = calc_ref(e); calls.append((e, o)); return o
    return calls, ev(ast.parse(expr, mode='eval').body)


def question(r, X, Y):
    expr, n = r.choice(EXPR)
    idx = [r.randrange(len(X)) for _ in range(n)]
    v = {k: int(Y[i]) for k, i in zip('abcd', idx)}
    imgs = '\n'.join(f'Image {j + 1}: {img_text(X[i])}' for j, i in enumerate(idx))
    words = expr
    for k in 'abcd': words = words.replace(k, WORD[k])
    words = words.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ')
    q = f'{imgs}\nEach image shows one handwritten digit. What is {words}?'
    calls, val = steps(expr, v)
    seen = ' '.join(f'Image {j + 1} shows {EY[0]}{j + 1}{EY[1]}{EY[2]}{v[k]}{EY[3]} {v[k]}.' for j, k in enumerate('abcd'[:n]))
    math = ' '.join(f'{e} = {CS}{e}{CE}{OS}{o}{OE} {o}.' for e, o in calls)
    return {'task': f'eyes_{n}img', 'prompt': f"Q: {q}\nA: Let's think step by step.", 'completion': f' {seen}\n{math} So the answer is {val}.',
            'gold': str(val), 'question': q, 'labels': [int(Y[i]) for i in idx]}


if __name__ == '__main__':
    if sys.argv[1] == 'cnn':
        D = mnist(); (X, Y), (Xt, Yt) = D['train'], D['test']
        torch.save({'Xtr': X.to(torch.uint8), 'Ytr': Y, 'Xte': Xt.to(torch.uint8), 'Yte': Yt}, 'data/mnist14q.pt')
        torch.manual_seed(0); net = CNN().cuda(); opt = torch.optim.Adam(net.parameters(), 1e-3)
        Xc, Yc = X.float().cuda(), Y.cuda()
        for ep in range(8):
            perm = torch.randperm(len(Xc), device='cuda')
            for b in range(0, len(Xc), 128):
                i = perm[b:b + 128]; x = Xc[i]
                x = torch.roll(x.view(-1, SIDE, SIDE), shifts=(random.randint(-1, 1), random.randint(-1, 1)), dims=(1, 2)).reshape(-1, NPIX)   # small shifts
                loss = Fn.cross_entropy(net(x), Yc[i]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): acc = (net(Xt.float().cuda()).argmax(-1).cpu() == Yt).float().mean().item()
            print(f'epoch {ep} test acc {100*acc:.2f}', flush=True)
        torch.save(net.state_dict(), 'runs/eye_cnn.pt'); print('CNN params', sum(p.numel() for p in net.parameters()))
    else:
        N = int(sys.argv[2]); M = torch.load('data/mnist14q.pt')
        r = random.Random(5)
        tr = [question(r, M['Xtr'], M['Ytr']) for _ in range(N)]
        r = random.Random(6); te = [question(r, M['Xte'], M['Yte']) for _ in range(800)]
        for f, rows in (('data/eyes_train.jsonl', tr), ('data/eyes_dev.jsonl', te[:300]), ('data/eyes_test.jsonl', te[300:])):
            with open(f, 'w') as o:
                for x in rows: o.write(json.dumps(x) + '\n')
        print(tr[0]['completion']); print(len(tr), 'train')
