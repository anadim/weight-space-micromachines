"""Tools megakernel host side: NVRTC compile (CUDA C++ -> cubin, no nvcc), driver-API launch on torch's current stream
(CUDA-graph capturable), state buffers, packing of every machine's stacks."""
import os, ctypes, numpy as np, torch
import tk_compile as TC
HERE = os.path.dirname(os.path.abspath(__file__))
def _nv_dir():                     # the pip 'nvidia' namespace package (cuda_nvrtc, cuda_runtime headers) of the torch wheels
    if os.environ.get('NVIDIA_PKG_DIR'): return os.environ['NVIDIA_PKG_DIR']
    try:
        import importlib.util
        return os.path.dirname(list(importlib.util.find_spec('nvidia.cuda_nvrtc').submodule_search_locations)[0])
    except Exception:
        return '/usr/local/lib/python3.12/dist-packages/nvidia'
_NV = _nv_dir()
_nvrtc = ctypes.CDLL(f'{_NV}/cuda_nvrtc/lib/libnvrtc.so.12')
_cuda = ctypes.CDLL('libcuda.so.1')
V = ctypes.c_void_p


def _chk(r, what):
    if r != 0: raise RuntimeError(f'{what} failed: {r}')


def compile_cubin(src, arch, name='tk.cu', opts=()):
    prog = V()
    _chk(_nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), name.encode(), 0, None, None), 'nvrtcCreateProgram')
    o = [f'--gpu-architecture=sm_{arch}', '-std=c++17', '-default-device', f'-I{_NV}/cuda_runtime/include', '--use_fast_math=false'] + list(opts)
    o = [x for x in o if x != '--use_fast_math=false']
    arr = (ctypes.c_char_p * len(o))(*[x.encode() for x in o])
    r = _nvrtc.nvrtcCompileProgram(prog, len(o), arr)
    n = ctypes.c_size_t(); _nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value); _nvrtc.nvrtcGetProgramLog(prog, log)
    if r != 0: raise RuntimeError('NVRTC: ' + log.value.decode())
    _nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n)); buf = ctypes.create_string_buffer(n.value)
    _chk(_nvrtc.nvrtcGetCUBIN(prog, buf), 'nvrtcGetCUBIN')
    return buf.raw, log.value.decode()


class Kernel:
    def __init__(self, cubin, fname):
        torch.zeros(1, device='cuda'); torch.cuda.synchronize()
        self.mod = V(); _chk(_cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin), 'cuModuleLoadData')
        self.fn = V(); _chk(_cuda.cuModuleGetFunction(ctypes.byref(self.fn), self.mod, fname.encode()), 'cuModuleGetFunction')
        nr = ctypes.c_int(); _cuda.cuFuncGetAttribute(ctypes.byref(nr), 4, self.fn); self.regs = nr.value   # NUM_REGS
        nl = ctypes.c_int(); _cuda.cuFuncGetAttribute(ctypes.byref(nl), 3, self.fn); self.local = nl.value  # LOCAL_SIZE_BYTES

    def __call__(self, grid, block, args, shmem=0):
        cargs = [a if isinstance(a, ctypes._SimpleCData) else V(a) for a in args]
        ptrs = (V * len(cargs))(*[ctypes.cast(ctypes.byref(a), V) for a in cargs])
        st = torch.cuda.current_stream().cuda_stream
        _chk(_cuda.cuLaunchKernel(self.fn, grid, 1, 1, block, 1, 1, shmem, V(st), ptrs, None), 'cuLaunchKernel')


class CStack(ctypes.Structure):
    _fields_ = [('ptr', V), ('code', V), ('w', V), ('b', V), ('lay', V), ('scr', V), ('nlay', ctypes.c_int),
                ('i64m', ctypes.c_int), ('thresh', ctypes.c_float), ('pad', ctypes.c_int)]


class CTool(ctypes.Structure):
    _fields_ = [('K', ctypes.c_int), ('W', ctypes.c_int), ('NS', ctypes.c_int), ('dS', ctypes.c_int), ('key', ctypes.c_int),
                ('nout', ctypes.c_int), ('outoff', ctypes.c_int), ('pad', ctypes.c_int),
                ('symtab', V), ('symring', V), ('Sring', V), ('keyring', V), ('hs', V), ('hw', V), ('s1', CStack), ('s2', CStack)]


class ToolsKernel:
    """machines: list of (name, Micromachine, embedding matrix the reader reads). Row stride Bp = B_max + 32
    (row Bp-1 is the dummy row of dead lanes)."""
    def __init__(self, machines, B_max, dev='cuda'):
        self.names = [n for n, _, _ in machines]; self.dev = dev
        self.Bp = Bp = (B_max + 32 + 31) // 32 * 32
        self.B_max = B_max
        cap = torch.cuda.get_device_capability(); arch = cap[0] * 10 + cap[1]
        src = open(os.path.join(HERE, 'tk_kernel.cu')).read()
        cubin, self.build_log = compile_cubin(src, arch)
        self.k = Kernel(cubin, 'tk_step')
        self.keep = []; self.tools = []; self.outoff = []; off = 0
        self.flag = torch.zeros(1, dtype=torch.int32, device=dev)
        self.info = {}
        for name, M, E in machines:
            tab, ok = TC.symtab(M, E); assert ok, f'{name}: symbol table not 0/1'
            L1, _ = TC.extract(M.stack1); L2, _ = TC.extract(M.stack2)
            nwin = M.K * M.NS; dS = L1[-1]['shape'][0]; n_out = L2[-1]['shape'][0]
            f64 = hasattr(M.stack1, 'net')
            p1 = TC.pack_stack(L1, nwin, nwin, M.NS, 0)
            p2 = TC.pack_stack(L2, nwin + dS, nwin, M.NS, 0)
            t = CTool(K=M.K, W=M.W, NS=M.NS, dS=dS, key=M.key, nout=n_out, outoff=off)
            T = lambda x, dt: self._keep(torch.as_tensor(x).to(dev, dt).contiguous())
            t.symtab = T(tab, torch.uint8).data_ptr()
            self.symtab = getattr(self, 'symtab', {}); self.symtab[name] = self.keep[-1]
            t.symring = self._keep(torch.zeros(M.K * M.NS * Bp, dtype=torch.uint8, device=dev)).data_ptr()
            t.Sring = self._keep(torch.zeros(M.W * dS * Bp, dtype=torch.float32, device=dev)).data_ptr()
            t.keyring = self._keep(torch.zeros(M.W * Bp, dtype=torch.uint8, device=dev)).data_ptr()
            t.hs = self._keep(torch.full((Bp,), M.K - 1, dtype=torch.int32, device=dev)).data_ptr()
            t.hw = self._keep(torch.full((Bp,), M.W - 1, dtype=torch.int32, device=dev)).data_ptr()
            st = {}
            for nm, p, is1 in (('s1', p1, True), ('s2', p2, False)):
                big = max(np.abs(p['w']).max(), np.abs(p['bias']).max()) >= 2 ** 31
                i64m = int((is1 and f64) or big)
                dt = torch.int64 if i64m else torch.int32
                cs = CStack(nlay=len(p['layers']), i64m=i64m, thresh=float((2 ** 53 if (is1 and f64) else 2 ** 24) * 0.99))
                cs.ptr = T(p['ptr'], torch.int64).data_ptr(); cs.code = T(p['code'], torch.int32).data_ptr()
                cs.w = T(p['w'], dt).data_ptr(); cs.b = T(p['bias'], dt).data_ptr()
                cs.lay = T(np.array(p['layers'], np.int64).ravel(), torch.int32).data_ptr()
                cs.scr = self._keep(torch.zeros(max(1, p['n_scr']) * Bp, dtype=dt, device=dev)).data_ptr()
                setattr(t, nm, cs); st[nm] = dict(i64=i64m, nnz=int(len(p['w'])), layers=len(p['layers']), n_scr=p['n_scr'])
            self.info[name] = dict(st, n_out=n_out, dS=dS)
            self.tools.append(t); self.outoff.append((off, n_out)); off += n_out
        self.nout = off
        arr = (CTool * len(self.tools))(*self.tools)
        raw = bytes(arr)
        self.desc = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev)
        self.out_e = torch.zeros(Bp, self.nout, device=dev)
        self.state = [self.keep[i] for i in range(len(self.keep))]   # (incl. constant tables; gather() picks state ones)
        self._state_ids = []
        for t in self.tools:
            for ptr in (t.symring, t.Sring, t.keyring, t.hs, t.hw):
                self._state_ids.append(next(x for x in self.keep if x.data_ptr() == ptr))

    def _keep(self, x):
        self.keep.append(x); return x

    def step(self, tok, B):
        """tok: int64 (>= B,) newest token per row (rows 0..B-1). Returns out_e (Bp, nout) view; rows < B valid."""
        if tok.shape[0] < self.Bp:
            tb = getattr(self, '_tok', None)
            if tb is None: self._tok = tb = torch.zeros(self.Bp, dtype=torch.int64, device=self.dev)
            tb[:tok.shape[0]].copy_(tok); tok = tb
        nchunk = (B + 31) // 32
        self.k(len(self.tools) * nchunk, 256,
               [self.desc.data_ptr(), ctypes.c_int(len(self.tools)), tok.data_ptr(), ctypes.c_int(B), ctypes.c_int(self.Bp),
                self.out_e.data_ptr(), ctypes.c_int(self.nout), ctypes.c_int(nchunk), self.flag.data_ptr()])
        return self.out_e

    def views(self, B):
        return [self.out_e[:B, o:o + n] for o, n in self.outoff]

    @torch.no_grad()
    def reset_rows(self, rows):
        for t, (name) in zip(self.tools, self.names):
            pass
        for x in self._state_ids:
            v = x.view(-1, self.Bp)
            if x.dtype == torch.int32:          # heads
                continue
            v[:, rows] = 0
        for t in self.tools:
            hs = next(x for x in self.keep if x.data_ptr() == t.hs); hw = next(x for x in self.keep if x.data_ptr() == t.hw)
            hs[rows] = t.K - 1; hw[rows] = t.W - 1

    @torch.no_grad()
    def gather(self, idx, n):
        """rows 0..n-1 <- rows idx (compaction / refill)."""
        for x in self._state_ids:
            v = x.view(-1, self.Bp); v[:, :n] = v[:, idx]

    @torch.no_grad()
    def copy_rows(self, src, dst):
        for x in self._state_ids:
            v = x.view(-1, self.Bp); v[:, dst] = v[:, src]


class CStack2(ctypes.Structure):
    _fields_ = [('ptr', V), ('cons', V), ('w', V), ('b', V), ('lay', V), ('acc', V), ('PN', V), ('stamp', V),
                ('nlay', ctypes.c_int), ('nU', ctypes.c_int), ('nsrc_in', ctypes.c_int), ('i64m', ctypes.c_int),
                ('thresh', ctypes.c_float), ('pad', ctypes.c_int)]


class CTool2(ctypes.Structure):
    _fields_ = [('K', ctypes.c_int), ('W', ctypes.c_int), ('NS', ctypes.c_int), ('dS', ctypes.c_int), ('key', ctypes.c_int),
                ('nout', ctypes.c_int), ('outoff', ctypes.c_int), ('maxsym', ctypes.c_int),
                ('symlist', V), ('keytab', V), ('tokring', V), ('Sring', V), ('keyring', V), ('hs', V), ('hw', V),
                ('s1', CStack2), ('s2', CStack2)]


def pack_push(layers, n_in):
    """CSC push lists. Units get global ids in layer order (hidden then output); sources = [inputs (n_in)] [units]."""
    u0 = 0; lay, cons, srcs, ws, bias = [], [], [], [], []
    for Ly in layers:
        nu = Ly['shape'][0]
        cons.append(u0 + Ly['rows']); srcs.append(Ly['idx'][Ly['cols']]); ws.append(Ly['vals']); bias.append(Ly['b'])
        lay.append((u0, nu)); u0 += nu
    cons, srcs, ws, bias = map(np.concatenate, (cons, srcs, ws, bias))
    nsrc = n_in + u0
    assert srcs.max() < n_in + u0 - layers[-1]['shape'][0]           # outputs are never sources
    order = np.lexsort((cons, srcs))
    cons, srcs, ws = cons[order], srcs[order], ws[order]
    ptr = np.zeros(nsrc + 1, np.int64); np.add.at(ptr, srcs + 1, 1); ptr = np.cumsum(ptr)
    assert np.all(ws == np.round(ws)) and np.all(bias == np.round(bias))
    # every consumer is in a strictly later layer than a unit source (push order = layer order)
    lay_of = np.concatenate([np.full(n, i) for i, (_, n) in enumerate(lay)])
    us = srcs >= n_in
    assert np.all(lay_of[cons[us]] > lay_of[srcs[us] - n_in])
    return dict(ptr=ptr, cons=cons.astype(np.int32), w=ws, bias=bias, lay=lay, nU=u0, n_in=n_in)


class ToolsKernel2(ToolsKernel):
    """Event-driven (push) version; same interface as ToolsKernel."""
    def __init__(self, machines, B_max, dev='cuda'):
        self.names = [n for n, _, _ in machines]; self.dev = dev
        self.Bp = Bp = (B_max + 32 + 31) // 32 * 32; self.B_max = B_max
        self.nch = Bp // 32
        cap = torch.cuda.get_device_capability(); arch = cap[0] * 10 + cap[1]
        cubin, self.build_log = compile_cubin(open(os.path.join(HERE, 'tk2_kernel.cu')).read(), arch, opts=['-Xptxas=-v'])
        self.k = Kernel(cubin, 'tk2_step')
        self.keep = []; self.tools = []; self.outoff = []; off = 0; self.info = {}; self._state = []; self._heads = []
        self.flag = torch.zeros(1, dtype=torch.int32, device=dev)
        self.stepc = torch.zeros(1, dtype=torch.int32, device=dev)
        self.symtabs = {}
        for name, M, E in machines:
            tab, ok = TC.symtab(M, E); assert ok, f'{name}: symbol table not 0/1'
            self.symtabs[name] = tab
            L1, _ = TC.extract(M.stack1); L2, _ = TC.extract(M.stack2)
            nwin = M.K * M.NS; dS = L1[-1]['shape'][0]; n_out = L2[-1]['shape'][0]
            f64 = hasattr(M.stack1, 'net')
            cnt = tab.sum(1); maxsym = int(cnt.max())
            sl = torch.full((tab.shape[0], maxsym), -1, dtype=torch.int16, device=dev)
            nz = tab.nonzero()
            rank = torch.zeros_like(nz[:, 0])
            if len(nz):
                first = torch.cat([torch.tensor([True], device=dev), nz[1:, 0] != nz[:-1, 0]])
                idx = torch.arange(len(nz), device=dev); start = torch.cummax(torch.where(first, idx, 0), 0).values
                rank = idx - start
            sl[nz[:, 0], rank] = nz[:, 1].to(torch.int16)
            t = CTool2(K=M.K, W=M.W, NS=M.NS, dS=dS, key=M.key, nout=n_out, outoff=off, maxsym=maxsym)
            K_ = lambda x: self._keep(x)
            t.symlist = K_(sl.contiguous()).data_ptr(); t.keytab = K_(tab[:, M.key].contiguous()).data_ptr()
            st_ = [K_(torch.zeros(M.K * Bp, dtype=torch.int32, device=dev)), K_(torch.zeros(M.W * dS * Bp, device=dev)),
                   K_(torch.zeros(M.W * Bp, dtype=torch.uint8, device=dev))]
            hs = K_(torch.full((Bp,), M.K - 1, dtype=torch.int32, device=dev)); hw = K_(torch.full((Bp,), M.W - 1, dtype=torch.int32, device=dev))
            t.tokring, t.Sring, t.keyring = [x.data_ptr() for x in st_]; t.hs, t.hw = hs.data_ptr(), hw.data_ptr()
            self._state += st_; self._heads.append((hs, M.K, hw, M.W))
            inf = {}
            for nm, Ls, n_in, is1 in (('s1', L1, nwin, True), ('s2', L2, nwin + dS, False)):
                p = pack_push(Ls, n_in)
                big = max(np.abs(p['w']).max(), np.abs(p['bias']).max()) >= 2 ** 31
                i64m = int((is1 and f64) or big); dt = torch.int64 if i64m else torch.int32
                cs = CStack2(nlay=len(p['lay']), nU=p['nU'], nsrc_in=n_in, i64m=i64m,
                             thresh=float((2 ** 53 if (is1 and f64) else 2 ** 24) * 0.99))
                T = lambda x, d: K_(torch.as_tensor(x).to(dev, d).contiguous())
                cs.ptr = T(p['ptr'], torch.int64).data_ptr(); cs.cons = T(p['cons'], torch.int32).data_ptr()
                cs.w = T(p['w'], dt).data_ptr(); cs.b = T(p['bias'], dt).data_ptr()
                cs.lay = T(np.array(p['lay'], np.int64).ravel(), torch.int32).data_ptr()
                cs.acc = K_(torch.zeros(self.nch * p['nU'] * 32, dtype=dt, device=dev)).data_ptr()
                cs.PN = K_(torch.zeros(self.nch * p['nU'] * 64, device=dev)).data_ptr()
                cs.stamp = K_(torch.zeros(self.nch * p['nU'], dtype=torch.int32, device=dev)).data_ptr()
                setattr(t, nm, cs)
                inf[nm] = dict(i64=i64m, nnz=int(len(p['w'])), layers=len(p['lay']), nU=p['nU'],
                               bias_pos=int((p['bias'][:p['nU'] - (n_out if not is1 else dS)] > 0).sum()))
            self.info[name] = dict(inf, n_out=n_out, dS=dS, maxsym=maxsym)
            self.tools.append(t); self.outoff.append((off, n_out)); off += n_out
        self.nout = off
        arr = (CTool2 * len(self.tools))(*self.tools)
        self.desc = torch.frombuffer(bytearray(bytes(arr)), dtype=torch.uint8).to(dev)
        self.out_e = torch.zeros(Bp, self.nout, device=dev)
        self._state_ids = self._state + [h for hs, _, hw, _ in self._heads for h in (hs, hw)]

    def step(self, tok, B):
        if tok.shape[0] < self.Bp:
            tb = getattr(self, '_tok', None)
            if tb is None: self._tok = tb = torch.zeros(self.Bp, dtype=torch.int64, device=self.dev)
            tb[:tok.shape[0]].copy_(tok); tok = tb
        self.stepc.add_(1)
        nchunk = (B + 31) // 32
        self.k(len(self.tools) * nchunk, 256,
               [self.desc.data_ptr(), ctypes.c_int(len(self.tools)), tok.data_ptr(), ctypes.c_int(B), ctypes.c_int(self.Bp),
                self.out_e.data_ptr(), ctypes.c_int(self.nout), ctypes.c_int(nchunk), self.flag.data_ptr(), self.stepc.data_ptr()])
        return self.out_e

    @torch.no_grad()
    def reset_rows(self, rows):
        for x in self._state: x.view(-1, self.Bp)[:, rows] = 0
        for hs, K, hw, W in self._heads: hs[rows] = K - 1; hw[rows] = W - 1


class CStack3(ctypes.Structure):
    _fields_ = [('ptr', V), ('cons', V), ('w', V), ('pre0', V), ('z0', V), ('P0', V), ('N0', V), ('lay', V),
                ('acc', V), ('P', V), ('N', V), ('stamp', V),
                ('nlay', ctypes.c_int), ('nU', ctypes.c_int), ('nsrc_in', ctypes.c_int), ('i64m', ctypes.c_int),
                ('thresh', ctypes.c_longlong)]


class CTool3(ctypes.Structure):
    _fields_ = CTool2._fields_[:-2] + [('skip', V), ('s1', CStack3), ('s2', CStack3)]


def baseline(layers, n_in, x_in):
    """exact int64 evaluation of a stack on input x_in; returns per-unit pre0, z0 (relu, outputs: pre), P0, N0."""
    x = np.asarray(x_in, np.int64).copy()
    pre_all, z_all, P_all, N_all = [], [], [], []
    for li, Ly in enumerate(layers):
        nu = Ly['shape'][0]; src = Ly['idx'][Ly['cols']]
        w = Ly['vals'].astype(np.int64); xv = x[src]
        assert np.all(np.abs(Ly['vals']) * np.abs(xv.astype(np.float64)) < 2 ** 62)
        prod = w * xv
        pre = Ly['b'].astype(np.int64).copy(); np.add.at(pre, Ly['rows'], prod)
        P = np.zeros(nu, np.int64); np.add.at(P, Ly['rows'], np.where(prod > 0, prod, 0))
        N = np.zeros(nu, np.int64); np.add.at(N, Ly['rows'], np.where(prod < 0, -prod, 0))
        z = np.maximum(pre, 0) if li < len(layers) - 1 else pre
        pre_all.append(pre); z_all.append(z); P_all.append(P); N_all.append(N)
        if li < len(layers) - 1: x = np.concatenate([x, z])
    return [np.concatenate(a) for a in (pre_all, z_all, P_all, N_all)]


class ToolsKernel3(ToolsKernel):
    """Differential event-driven version (tk3_kernel.cu); same interface."""
    def __init__(self, machines, B_max, dev='cuda', scratch=0):
        self.names = [n for n, _, _ in machines]; self.dev = dev
        Bp0 = (B_max + 31) // 32 * 32; S = (scratch + 31) // 32 * 32
        self.scr0, self.n_scr = Bp0, S                      # scratch rows [Bp0, Bp0+S) (priming), dummy row = Bp-1
        self.Bp = Bp = Bp0 + S + 32; self.B_max = B_max; self.nch = Bp // 32
        cap = torch.cuda.get_device_capability(); arch = cap[0] * 10 + cap[1]
        cubin, self.build_log = compile_cubin(open(os.path.join(HERE, 'tk3_kernel.cu')).read(), arch, opts=['-Xptxas=-v'])
        self.k = Kernel(cubin, 'tk3_step')
        self.keep = []; self.tools = []; self.outoff = []; off = 0; self.info = {}; self._state = []; self._heads = []
        self.flag = torch.zeros(1, dtype=torch.int32, device=dev)
        self.stepc = torch.zeros(1, dtype=torch.int32, device=dev)
        self.symtabs = {}; self._tokrings = []
        K_ = self._keep
        for name, M, E in machines:
            tab, ok = TC.symtab(M, E); assert ok, f'{name}: symbol table not 0/1'
            self.symtabs[name] = tab
            Vv = tab.shape[0]
            always = (tab.min(0).values == 1)                                   # 1 on every vocabulary row
            L1, _ = TC.extract(M.stack1); L2, _ = TC.extract(M.stack2)
            nwin = M.K * M.NS; dS = L1[-1]['shape'][0]; n_out = L2[-1]['shape'][0]
            f64 = hasattr(M.stack1, 'net')
            extra = tab.clone(); extra[:, always] = 0                           # +1 deltas of real tokens
            na = int(always.sum())
            maxsym = max(int(extra.sum(1).max()), na) + 1
            sl = torch.full((Vv + 1, maxsym), -1, dtype=torch.int16, device=dev)
            nz = extra.nonzero()
            if len(nz):
                idx = torch.arange(len(nz), device=dev)
                first = torch.cat([torch.tensor([True], device=dev), nz[1:, 0] != nz[:-1, 0]])
                rank = idx - torch.cummax(torch.where(first, idx, 0), 0).values
                sl[nz[:, 0], rank] = nz[:, 1].to(torch.int16)
            al = always.nonzero()[:, 0]
            sl[Vv, :na] = (-al - 2).to(torch.int16)                            # empty slot: -1 on the always-on bits
            keytab = torch.cat([tab[:, M.key], torch.zeros(1, dtype=torch.uint8, device=dev)])
            t = CTool3(K=M.K, W=M.W, NS=M.NS, dS=dS, key=M.key, nout=n_out, outoff=off, maxsym=maxsym)
            t.symlist = K_(sl.contiguous()).data_ptr(); t.keytab = K_(keytab.contiguous()).data_ptr()
            self._symbuf = getattr(self, '_symbuf', []); self._symbuf.append((self.keep[-2], self.keep[-1], M, always.clone(), maxsym))
            tr = K_(torch.full((M.K * Bp,), Vv, dtype=torch.int32, device=dev)); self._tokrings.append((tr, Vv))
            st_ = [tr, K_(torch.zeros(M.W * dS * Bp, device=dev)), K_(torch.zeros(M.W * Bp, dtype=torch.uint8, device=dev))]
            hs = K_(torch.full((Bp,), M.K - 1, dtype=torch.int32, device=dev)); hw = K_(torch.full((Bp,), M.W - 1, dtype=torch.int32, device=dev))
            t.tokring, t.Sring, t.keyring = [x.data_ptr() for x in st_]; t.hs, t.hw = hs.data_ptr(), hw.data_ptr()
            self.skip = getattr(self, 'skip', {}); self.skip[name] = K_(torch.zeros(Bp, dtype=torch.uint8, device=dev))
            t.skip = self.skip[name].data_ptr()        # per-row token-filter skip (caller fills before step)
            self._state += st_; self._heads.append((hs, M.K, hw, M.W))
            win0 = np.zeros(nwin, np.int64)
            for o in range(M.K): win0[o * M.NS + al.cpu().numpy()] = 1
            inf = {}
            for nm, Ls, n_in, is1 in (('s1', L1, nwin, True), ('s2', L2, nwin + dS, False)):
                p = pack_push(Ls, n_in)
                x_in = win0 if is1 else np.concatenate([win0, np.zeros(dS, np.int64)])
                pre0, z0, P0, N0 = baseline(Ls, n_in, x_in)
                th = 2 ** 53 if (is1 and f64) else 2 ** 24
                assert P0.max() < th and N0.max() < th and np.abs(pre0).max() < th, f'{name}.{nm}: baseline not exact'
                big = max(np.abs(p['w']).max(), np.abs(pre0).max(), np.abs(z0).max()) >= 2 ** 31
                i64m = int((is1 and f64) or big); dt = torch.int64 if i64m else torch.int32
                cs = CStack3(nlay=len(p['lay']), nU=p['nU'], nsrc_in=n_in, i64m=i64m, thresh=th)
                T = lambda x, d: K_(torch.as_tensor(x).to(dev, d).contiguous())
                cs.ptr = T(p['ptr'], torch.int64).data_ptr(); cs.cons = T(p['cons'], torch.int32).data_ptr()
                cs.w = T(p['w'], dt).data_ptr(); cs.pre0 = T(pre0, dt).data_ptr(); cs.z0 = T(z0, dt).data_ptr()
                cs.P0 = T(P0, torch.int64).data_ptr(); cs.N0 = T(N0, torch.int64).data_ptr()
                cs.lay = T(np.array(p['lay'], np.int64).ravel(), torch.int32).data_ptr()
                cs.acc = K_(torch.zeros(1, dtype=dt, device=dev)).data_ptr()
                cs.P = K_(torch.zeros(self.nch * p['nU'] * 32, dtype=torch.int64, device=dev)).data_ptr()
                cs.N = K_(torch.zeros(self.nch * p['nU'] * 32, dtype=torch.int64, device=dev)).data_ptr()
                cs.stamp = K_(torch.zeros(self.nch * p['nU'], dtype=torch.int32, device=dev)).data_ptr()
                setattr(t, nm, cs)
                inf[nm] = dict(i64=i64m, nnz=int(len(p['w'])), layers=len(p['lay']), nU=p['nU'], base_active=int((z0[:p['nU'] - Ls[-1]['shape'][0]] != 0).sum()))
            self.info[name] = dict(inf, n_out=n_out, dS=dS, maxsym=maxsym, always=na)
            self.tools.append(t); self.outoff.append((off, n_out)); off += n_out
        self.nout = off
        arr = (CTool3 * len(self.tools))(*self.tools)
        self.desc = torch.frombuffer(bytearray(bytes(arr)), dtype=torch.uint8).to(dev)
        self.out_e = torch.zeros(Bp, self.nout, device=dev)
        self._state_ids = self._state + [h for hs, _, hw, _ in self._heads for h in (hs, hw)]

    def step(self, tok, B, r0=0):
        if tok.shape[0] < self.Bp:
            tb = getattr(self, '_tok', None)
            if tb is None: self._tok = tb = torch.zeros(self.Bp, dtype=torch.int64, device=self.dev)
            tb[:tok.shape[0]].copy_(tok); tok = tb
        self.stepc.add_(1)
        nchunk = (B + 31) // 32
        self.k(len(self.tools) * nchunk, 512,
               [self.desc.data_ptr(), ctypes.c_int(len(self.tools)), tok.data_ptr(), ctypes.c_int(B), ctypes.c_int(self.Bp),
                self.out_e.data_ptr(), ctypes.c_int(self.nout), ctypes.c_int(nchunk), self.flag.data_ptr(), self.stepc.data_ptr(),
                ctypes.c_int(r0)])
        return self.out_e

    @torch.no_grad()
    def update_symtab(self, i, E):
        """machine i reads a live embedding (calc: host E after a weight update): rebuild its symbol lists in place.
        Must stay 0/1 with the same always-on set (else the baseline would change): asserted."""
        sl, kt, M, always, maxsym = self._symbuf[i]
        tab, ok = TC.symtab(M, E); assert ok, 'symbol table not 0/1 after the update'
        if torch.equal(tab, self.symtabs[self.names[i]]): return False
        assert torch.equal(tab.min(0).values == 1, always), 'always-on symbol set changed'
        extra = tab.clone(); extra[:, always] = 0
        assert int(extra.sum(1).max()) < maxsym
        Vv = tab.shape[0]; dev = tab.device
        sl[:Vv] = -1
        nz = extra.nonzero()
        if len(nz):
            idx = torch.arange(len(nz), device=dev)
            first = torch.cat([torch.tensor([True], device=dev), nz[1:, 0] != nz[:-1, 0]])
            rank = idx - torch.cummax(torch.where(first, idx, 0), 0).values
            sl[nz[:, 0], rank] = nz[:, 1].to(torch.int16)
        kt[:Vv] = tab[:, M.key]
        self.symtabs[self.names[i]] = tab
        return True

    @torch.no_grad()
    def reset_rows(self, rows):
        for x in self._state: x.view(-1, self.Bp)[:, rows] = 0
        for tr, Vv in self._tokrings: tr.view(-1, self.Bp)[:, rows] = Vv
        for hs, K, hw, W in self._heads: hs[rows] = K - 1; hw[rows] = W - 1


# ============================================================ FLOAT mode (trainable implants)
class CStackF(ctypes.Structure):
    _fields_ = [('ptr', V), ('cons', V), ('w', V), ('pre0', V), ('z0', V), ('lay', V), ('acc', V), ('stamp', V),
                ('nlay', ctypes.c_int), ('nU', ctypes.c_int), ('nsrc_in', ctypes.c_int), ('f64', ctypes.c_int)]


class CToolF(ctypes.Structure):
    _fields_ = [('K', ctypes.c_int), ('W', ctypes.c_int), ('NS', ctypes.c_int), ('dS', ctypes.c_int), ('key', ctypes.c_int),
                ('nout', ctypes.c_int), ('outoff', ctypes.c_int), ('maxsym', ctypes.c_int),
                ('symidx', V), ('symval', V), ('keytab', V), ('tokring', V), ('Sring', V), ('keyring', V), ('hs', V), ('hw', V),
                ('skip', V), ('s1', CStackF), ('s2', CStackF)]


def _net_coo(net):
    """per layer (rows, cols->wire idx, nu) of a ReLUNet (or F64 wrapper); the VALUES are read live at refresh()."""
    net = getattr(net, 'net', net)
    out = []
    for l in range(net.L + 1):
        idx = getattr(net, f'i{l}'); wi = getattr(net, f'wi{l}')
        out.append(dict(rows=wi[0].cpu().numpy().astype(np.int64), src=idx[wi[1]].cpu().numpy().astype(np.int64),
                        nu=int(getattr(net, f'shape{l}')[0])))
    return net, out


class ToolsKernelF(ToolsKernel):
    """Float-mode tools kernel for TRAINABLE implants. Structure (sparsity pattern, symbol count caps) is fixed at
    build; refresh() re-reads the live parameters: reader_w/reader_b (-> soft reader table over the vocabulary),
    every ReLUNet wv{l}/b{l} (-> push-list weights via a fixed permutation), and recomputes the baseline. Call refresh()
    after every optimizer step (outside CUDA graphs; buffers keep their addresses, so captured graphs stay valid).
    machines: list of (name, Micromachine, E) where E is the embedding the reader reads (live tensor)."""
    def __init__(self, machines, B_max, dev='cuda', scratch=0, sym_cap=64):
        self.machines = machines; self.names = [n for n, _, _ in machines]; self.dev = dev
        Bp0 = (B_max + 31) // 32 * 32; S = (scratch + 31) // 32 * 32
        self.scr0, self.n_scr = Bp0, S
        self.Bp = Bp = Bp0 + S + 32; self.B_max = B_max; self.nch = Bp // 32
        cap = torch.cuda.get_device_capability(); arch = cap[0] * 10 + cap[1]
        cubin, self.build_log = compile_cubin(open(os.path.join(HERE, 'tkf_kernel.cu')).read(), arch)
        self.k = Kernel(cubin, 'tkf_step')
        self.keep = []; self.tools = []; self.outoff = []; off = 0; self.info = {}; self._state = []; self._heads = []
        self.stepc = torch.zeros(1, dtype=torch.int32, device=dev); self.flag = torch.zeros(1, dtype=torch.int32, device=dev)
        self.skip = {}; self._tokrings = []; self._ref = []
        K_ = self._keep
        for name, M, E in machines:
            NS = M.NS; Vv = E.shape[0]; cap_ = min(NS, sym_cap)
            net1, L1 = _net_coo(M.stack1); net2, L2 = _net_coo(M.stack2)
            nwin = M.K * NS; dS = L1[-1]['nu']; n_out = L2[-1]['nu']
            t = CToolF(K=M.K, W=M.W, NS=NS, dS=dS, key=M.key, nout=n_out, outoff=off, maxsym=cap_)
            symidx = K_(torch.full((Vv + 1, cap_), -1, dtype=torch.int16, device=dev)); symval = K_(torch.zeros(Vv + 1, cap_, device=dev))
            keytab = K_(torch.zeros(Vv + 1, device=dev))
            t.symidx, t.symval, t.keytab = symidx.data_ptr(), symval.data_ptr(), keytab.data_ptr()
            tr = K_(torch.full((M.K * Bp,), Vv, dtype=torch.int32, device=dev)); self._tokrings.append((tr, Vv))
            st_ = [tr, K_(torch.zeros(M.W * dS * Bp, device=dev)), K_(torch.zeros(M.W * Bp, device=dev))]
            hs = K_(torch.full((Bp,), M.K - 1, dtype=torch.int32, device=dev)); hw = K_(torch.full((Bp,), M.W - 1, dtype=torch.int32, device=dev))
            t.tokring, t.Sring, t.keyring = [x.data_ptr() for x in st_]; t.hs, t.hw = hs.data_ptr(), hw.data_ptr()
            self.skip[name] = K_(torch.zeros(Bp, dtype=torch.uint8, device=dev)); t.skip = self.skip[name].data_ptr()
            self._state += st_; self._heads.append((hs, M.K, hw, M.W))
            ref = dict(name=name, M=M, E=E, symidx=symidx, symval=symval, keytab=keytab, cap=cap_, stacks=[])
            for nm, net, Ls, n_in, is1 in (('s1', net1, L1, nwin, True), ('s2', net2, L2, nwin + dS, False)):
                f64 = int(is1 and hasattr(M.stack1, 'net'))
                dt = torch.float64 if f64 else torch.float32
                u0 = 0; cons, srcs, lay = [], [], []
                for Ly in Ls:
                    cons.append(u0 + Ly['rows']); srcs.append(Ly['src']); lay.append((u0, Ly['nu'])); u0 += Ly['nu']
                cons = np.concatenate(cons); srcs = np.concatenate(srcs)
                order = np.lexsort((cons, srcs))                  # COO (layer-major, buffer order) -> CSC push order
                ptr = np.zeros(n_in + u0 + 1, np.int64); np.add.at(ptr, srcs + 1, 1); ptr = np.cumsum(ptr)
                cs = CStackF(nlay=len(lay), nU=u0, nsrc_in=n_in, f64=f64)
                T = lambda x, d: K_(torch.as_tensor(x).to(dev, d).contiguous())
                cs.ptr = T(ptr, torch.int64).data_ptr(); cs.cons = T(cons[order], torch.int32).data_ptr()
                wbuf = K_(torch.zeros(len(order), dtype=dt, device=dev)); cs.w = wbuf.data_ptr()
                pre0 = K_(torch.zeros(u0, dtype=dt, device=dev)); z0 = K_(torch.zeros(u0, dtype=dt, device=dev))
                cs.pre0, cs.z0 = pre0.data_ptr(), z0.data_ptr()
                cs.lay = T(np.array(lay, np.int64).ravel(), torch.int32).data_ptr()
                cs.acc = K_(torch.zeros(self.nch * u0 * 32, dtype=dt, device=dev)).data_ptr()
                cs.stamp = K_(torch.zeros(self.nch * u0, dtype=torch.int32, device=dev)).data_ptr()
                setattr(t, nm, cs)
                ref['stacks'].append(dict(net=net, perm=torch.as_tensor(order, device=dev), w=wbuf, pre0=pre0, z0=z0, n_in=n_in, dt=dt,
                                          nlay=len(lay)))
            self._ref.append(ref)
            self.tools.append(t); self.outoff.append((off, n_out)); off += n_out
            self.info[name] = dict(n_out=n_out, dS=dS, sym_cap=cap_)
        self.nout = off
        arr = (CToolF * len(self.tools))(*self.tools)
        self.desc = torch.frombuffer(bytearray(bytes(arr)), dtype=torch.uint8).to(dev)
        self.out_e = torch.zeros(Bp, self.nout, device=dev)
        self._state_ids = self._state + [h for hs, _, hw, _ in self._heads for h in (hs, hw)]
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """re-read every live implant parameter (reader, stack weights/biases) into the kernel's buffers."""
        for ref in self._ref:
            M = ref['M']; E = ref['E']
            tab = torch.clamp(torch.nn.functional.linear(E.float(), M.reader_w.float(), M.reader_b.float()), 0, 1)   # (V, NS)
            nzc = (tab != 0).sum(1); assert int(nzc.max()) <= ref['cap'], f"{ref['name']}: reader nonzeros {int(nzc.max())} > cap"
            ref['symidx'][:-1] = -1; ref['symval'][:-1] = 0
            nz = tab.nonzero()
            if len(nz):
                idx = torch.arange(len(nz), device=self.dev)
                first = torch.cat([torch.tensor([True], device=self.dev), nz[1:, 0] != nz[:-1, 0]])
                rank = idx - torch.cummax(torch.where(first, idx, 0), 0).values
                ref['symidx'][nz[:, 0], rank] = nz[:, 1].to(torch.int16); ref['symval'][nz[:, 0], rank] = tab[nz[:, 0], nz[:, 1]]
            ref['keytab'][:-1] = tab[:, M.key]
            for sd in ref['stacks']:
                net = sd['net']; dt = sd['dt']
                wv = torch.cat([getattr(net, f'wv{l}').detach().to(dt) for l in range(sd['nlay'])])
                sd['w'].copy_(wv[sd['perm']])
                # baseline = the stack on an all-zero input (zero window, C = 0): pre-activations of every unit
                n_in = sd['n_in']; wires = torch.zeros(1, n_in + net.n_units, dtype=dt, device=self.dev)
                pres = []; p = n_in
                for l in range(net.L + 1):
                    lin = _lin_fresh(net, l, wires[:, getattr(net, f'i{l}')], dt)      # live wv/b (no cached CSR)
                    pres.append(lin[0])
                    if l < net.L:
                        wires[:, p:p + lin.shape[-1]] = torch.relu(lin); p += lin.shape[-1]
                pre0 = torch.cat(pres).to(dt)
                sd['pre0'].copy_(pre0)
                z0 = pre0.clone(); nh = net.n_units; z0[:nh] = torch.relu(pre0[:nh]); sd['z0'].copy_(z0)

    def step(self, tok, B, r0=0):
        if tok.shape[0] < self.Bp:
            tb = getattr(self, '_tok', None)
            if tb is None: self._tok = tb = torch.zeros(self.Bp, dtype=torch.int64, device=self.dev)
            tb[:tok.shape[0]].copy_(tok); tok = tb
        self.stepc.add_(1)
        nchunk = (B + 31) // 32
        self.k(len(self.tools) * nchunk, 512,
               [self.desc.data_ptr(), ctypes.c_int(len(self.tools)), tok.data_ptr(), ctypes.c_int(B), ctypes.c_int(self.Bp),
                self.out_e.data_ptr(), ctypes.c_int(self.nout), ctypes.c_int(nchunk), self.stepc.data_ptr(), ctypes.c_int(r0)])
        return self.out_e

    @torch.no_grad()
    def reset_rows(self, rows):
        for x in self._state: x.view(-1, self.Bp)[:, rows] = 0
        for tr, Vv in self._tokrings: tr.view(-1, self.Bp)[:, rows] = Vv
        for hs, K, hw, W in self._heads: hs[rows] = K - 1; hw[rows] = W - 1

    def update_symtab(self, i, E):          # engine2.refresh() hook: the float kernel re-reads everything
        self.refresh(); return True


def _lin_fresh(net, l, x, dt):
    """ReLUNet.lin from the LIVE wv/b (ReLUNet caches its CSR in net._csr on first use: stale after training)."""
    W = torch.sparse_coo_tensor(getattr(net, f'wi{l}'), getattr(net, f'wv{l}').detach().to(dt), getattr(net, f'shape{l}'))
    return (torch.sparse.mm(W, x.to(dt).T).T + getattr(net, f'b{l}').detach().to(dt))
