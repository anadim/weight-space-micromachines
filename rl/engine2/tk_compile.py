"""Stack compiler for the tools megakernel: extract a compiled ReLUNet (COO buffers), interval-bound analysis
(exactness of the fp32/fp64 reference, int32/int64 choice), symbol tables, ELL/CSR packing for the kernel."""
import numpy as np, torch, torch.nn.functional as Fn

F32_EXACT, F64_EXACT, I32 = 2.0 ** 24, 2.0 ** 53, 2.0 ** 31 - 1


def extract(net):
    """ReLUNet (or teamc F64 wrapper) -> list of layers (idx, rows, cols, vals, b, shape); last = output layer."""
    net = getattr(net, 'net', net)
    out = []
    for l in range(net.L + 1):
        idx = getattr(net, f'i{l}').cpu().numpy().astype(np.int64)
        wi = getattr(net, f'wi{l}').cpu().numpy(); wv = getattr(net, f'wv{l}').double().cpu().numpy()
        b = getattr(net, f'b{l}').double().cpu().numpy()
        out.append(dict(idx=idx, rows=wi[0].astype(np.int64), cols=wi[1].astype(np.int64), vals=wv, b=b,
                        shape=tuple(getattr(net, f'shape{l}'))))
    return out, int(net.n_units)


def bounds(layers, n_in, in_lo, in_hi):
    """Interval propagation. Returns dict: wire lo/hi (n_in + n_units), output lo/hi, per-layer max |partial sum| bound
    (|b| + sum |w| max|x|: every partial sum in ANY summation order is bounded by it), max |pre-activation|."""
    lo = [np.asarray(in_lo, np.float64)]; hi = [np.asarray(in_hi, np.float64)]
    wlo = np.concatenate(lo); whi = np.concatenate(hi)
    psum, pre_abs = [], []
    nL = len(layers)
    for li, Ly in enumerate(layers):
        xl, xh = wlo[Ly['idx'][Ly['cols']]], whi[Ly['idx'][Ly['cols']]]
        w = Ly['vals']; nu = Ly['shape'][0]
        plo = Ly['b'] + np.bincount(Ly['rows'], np.where(w > 0, w * xl, w * xh), minlength=nu)
        phi = Ly['b'] + np.bincount(Ly['rows'], np.where(w > 0, w * xh, w * xl), minlength=nu)
        ps = np.abs(Ly['b']) + np.bincount(Ly['rows'], np.abs(w) * np.maximum(np.abs(xl), np.abs(xh)), minlength=nu)
        psum.append(float(ps.max()) if nu else 0.0); pre_abs.append(float(np.maximum(np.abs(plo), np.abs(phi)).max()) if nu else 0.0)
        if li < nL - 1:
            wlo = np.concatenate([wlo, np.maximum(plo, 0)]); whi = np.concatenate([whi, np.maximum(phi, 0)])
        else:
            olo, ohi = plo, phi
    return dict(lo=wlo, hi=whi, out_lo=olo, out_hi=ohi, psum=psum, pre_abs=pre_abs, max_psum=max(psum),
                n_dead=int(sum(1 for i in range(n_in, len(whi)) if whi[i] <= 0)),
                n_lin=int(sum(1 for i in range(n_in, len(wlo)) if wlo[i] > 0)))


@torch.no_grad()
def symtab(M, E, chunk=16384):
    """(V, NS) symbol flags per vocabulary row = M.symbols(E[v]) (same op as the reference, row-wise); must be 0/1."""
    out = []
    for c in range(0, E.shape[0], chunk):
        out.append(M.symbols(E[c:c + chunk]))
    s = torch.cat(out)
    ok = bool(((s == 0) | (s == 1)).all())
    return s.to(torch.uint8), ok


def pack_stack(layers, n_in, n_win, NS, wire_base):
    """CSR per unit for the kernel. Input references: window bit i < n_win -> code = -(1 + (o << 16 | s)) with
    o = i // NS (offset back in time), s = i % NS; other wires -> scratch row (wire - n_win) + wire_base.
    Returns per-layer (unit_ptr offsets into ent arrays, n_units, dst scratch row of first unit), ent_code int32,
    ent_w int64 (weights are integers), bias int64."""
    ptr, codes, ws, bias, lay = [0], [], [], [], []
    p = n_in
    for li, Ly in enumerate(layers):
        nu = Ly['shape'][0]
        order = np.lexsort((Ly['cols'], Ly['rows']))
        r, c, v = Ly['rows'][order], Ly['cols'][order], Ly['vals'][order]
        wire = Ly['idx'][c]
        code = np.where(wire < n_win, -(1 + (((wire // NS) << 16) | (wire % NS))), wire - n_win + wire_base)
        cnt = np.bincount(r, minlength=nu)
        start = len(ptr) - 1
        ptr.extend((ptr[-1] + np.cumsum(cnt)).tolist())
        codes.append(code.astype(np.int64)); ws.append(v); bias.append(Ly['b'])
        is_out = li == len(layers) - 1
        lay.append((start, nu, (p - n_win + wire_base) if not is_out else -1))
        if not is_out: p += nu
    W = np.concatenate(ws)
    assert np.all(W == np.round(W)) and np.all(np.concatenate(bias) == np.round(np.concatenate(bias)))
    return dict(ptr=np.array(ptr, np.int64), code=np.concatenate(codes).astype(np.int32), w=W.astype(np.int64),
                bias=np.concatenate(bias).astype(np.int64), layers=lay, n_scr=p - n_win)
