"""Offline identity of the shared-weights export (export.py --shared 1) against the full export (same build, same GPU).

python check_shared.py FULL.pt SHARED.pt          (cwd = ifeval/heldout, PYTHONPATH as for export.py)
1. tensors: SHARED = FULL minus {voc.E, force.F} plus force.fix_tok / fix_rows / gain / chunk and voc.shape; shared ones bitwise equal
2. the host's lm_head.weight (tied with embed_tokens) == FULL's voc.E, bitwise
3. force rows: Executor._force(widx) == FULL force.F[widx] bitwise (fp32 bit patterns) for all V+1 rows: in order (blocks of 24),
   and a random permutation cut into random batch sizes (1..24 and 1..64). For the record also the unpadded variant (the per-row
   mean taken over a block of B rows): mismatching rows per B.
4. readout + vocab op: Executor._vocab(vin, h) of both (top, blocked, alt, blocked mask) on real host final states, B = 24 and 13
"""
import random, sys, time, torch
sys.setrecursionlimit(100000)
from common import load
from execu import Executor

full_p, sh_p = sys.argv[1], sys.argv[2]
FULL, SH = torch.load(full_p), torch.load(sh_p)
tok, model = load()
E = model.lm_head.weight
bits = lambda x: x.contiguous().view(torch.int32)
ok_all = True


def say(name, ok, extra=''):
    global ok_all
    ok_all &= bool(ok); print(f'CHECK {name}: {"OK" if ok else "FAIL"} {extra}', flush=True)


# 1. tensors
new = {'force.fix_tok', 'force.fix_rows', 'force.gain', 'force.chunk', 'voc.shape'}
gone = {'voc.E', 'force.F'}
say('keys', set(SH) == (set(FULL) - gone) | new, f'full {len(FULL)} shared {len(SH)}')
say('shared tensors bitwise', all(FULL[k].dtype == SH[k].dtype and torch.equal(FULL[k], SH[k]) for k in SH if k not in new))
for nm, D in [('full', FULL), ('shared', SH)]:
    print(f'SIZE {nm}: tensors {len(D)} numbers {sum(v.numel() for v in D.values())} bytes {sum(v.numel() * v.element_size() for v in D.values())}')

# 2. host matrix
say('tied lm_head/embed_tokens', E.data_ptr() == model.model.embed_tokens.weight.data_ptr(), str(tuple(E.shape)))
say('host lm_head == stored voc.E', FULL['voc.E'].dtype == E.dtype and torch.equal(FULL['voc.E'].cuda(), E))

# 3. force rows
exF = Executor(FULL, 'cuda'); exS = Executor(SH, 'cuda', host_E=E)
F = exF.sd['force.F']; V = E.shape[0]
say('F zero row', bool((bits(F[V]) == 0).all()))
t0 = time.time(); bad = 0
for s in range(0, V, 24):
    w = torch.arange(s, min(s + 24, V), device='cuda')
    bad += int((bits(exS._force(w)) != bits(F[w])).any(1).sum())
say('force rows in order (blocks of 24)', bad == 0, f'mismatching rows {bad}/{V} ({time.time() - t0:.1f}s)')
for hi in (24, 64):
    rng = random.Random(hi); g = torch.Generator().manual_seed(hi)
    perm = torch.randperm(V + 1, generator=g).cuda(); s = 0; bad = 0; nb = 0
    while s <= V:
        b = rng.randint(1, hi); w = perm[s:s + b]; s += b; nb += 1
        bad += int((bits(exS._force(w)) != bits(F[w])).any(1).sum())
    say(f'force rows random batches 1..{hi}', bad == 0, f'mismatching rows {bad}/{V + 1} in {nb} batches')
# unpadded variant (for the record): the per-row mean over a block of B rows
gs = exS.sd['voc.g']; mism = {}
for B in (1, 2, 4, 8, 13, 16, 24, 32, 64):
    bad = 0
    for s in range(0, V - V % B, B):
        w = torch.arange(s, s + B, device='cuda')
        d = E[w].float() / gs; d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True))
        fi = exS.fix_idx[w]; d = torch.where((fi >= 0)[:, None], exS.fix_rows[fi.clamp(min=0)], d) * exS.gain
        bad += int((bits(d) != bits(F[w])).any(1).sum())
    mism[B] = bad
print('UNPADDED variant, mismatching rows by block size B:', mism, flush=True)
w = torch.randint(0, V + 1, (24,), device='cuda'); torch.cuda.synchronize(); t0 = time.time()
for _ in range(200): exS._force(w)
torch.cuda.synchronize(); print(f'TIME _force B=24: {(time.time() - t0) / 200 * 1e3:.3f} ms/call', flush=True)

# 4. readout + vocab op on real host final states (input of the final norm)
H = []
hk = model.model.norm.register_forward_pre_hook(lambda m, x: H.append(x[0].reshape(-1, x[0].shape[-1]).clone()))
for p in ['Write a short poem about the sea, all in lowercase.', 'List three facts about Mars. Use exactly 3 bullet points.',
          'Explain recursion to a child, without using any commas.', 'Repeat the request first: what is the capital of France?']:
    ids = tok(tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + p * 4,
              return_tensors='pt').input_ids.cuda()
    with torch.no_grad(): model.model(input_ids=ids)
hk.remove(); Hs = torch.cat(H); print('readout states', tuple(Hs.shape), Hs.dtype)
g = torch.Generator().manual_seed(0); bad = 0; n = 0
for B in (24, 13):
    for s in range(0, Hs.shape[0] - B, B):
        h = Hs[s:s + B]
        vin = torch.zeros(B, 51, dtype=torch.float64)
        vin[:, 0:7] = torch.randint(0, 2, (B, 7), generator=g).double(); vin[:, 7:43] = torch.randint(0, 2147483647, (B, 36), generator=g).double()
        vin[:, 43:49] = torch.randint(0, 2147483647, (B, 6), generator=g).double(); vin[:, 49] = torch.randint(0, 2147483647, (B,), generator=g).double()
        vin[:, 50] = torch.randint(0, 2, (B,), generator=g).double(); vin = vin.cuda()
        a, b = exF._vocab(vin, h), exS._vocab(vin, h)
        n += B; bad += int(((a[0] != b[0]) | (a[1] != b[1]) | (a[2] != b[2]) | (exF._last_blk != exS._last_blk).any(1)).sum())
say('readout/vocab op', bad == 0, f'mismatching rows {bad}/{n}')
print('ALL_OK' if ok_all else 'SOME_FAIL', flush=True)
