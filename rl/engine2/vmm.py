"""Fixed-address device buffers whose PHYSICAL memory can be released and re-mapped (CUDA virtual memory management via
the driver API). Used for the decode KV cache: CUDA graphs keep its virtual addresses, while training gets the
physical memory between rollouts (the KV contents are not needed across an update: every wave re-prefills)."""
import ctypes, torch
_cu = ctypes.CDLL('libcuda.so.1')
U64 = ctypes.c_ulonglong


class _Loc(ctypes.Structure): _fields_ = [('type', ctypes.c_int), ('id', ctypes.c_int)]
class _Flags(ctypes.Structure): _fields_ = [('compressionType', ctypes.c_ubyte), ('gpuDirectRDMACapable', ctypes.c_ubyte),
                                            ('usage', ctypes.c_ushort), ('reserved', ctypes.c_ubyte * 4)]
class _Prop(ctypes.Structure): _fields_ = [('type', ctypes.c_int), ('requestedHandleTypes', ctypes.c_int), ('location', _Loc),
                                           ('win32HandleMetaData', ctypes.c_void_p), ('allocFlags', _Flags)]
class _Access(ctypes.Structure): _fields_ = [('location', _Loc), ('flags', ctypes.c_int)]


def _chk(r, w):
    if r != 0: raise RuntimeError(f'{w}: CUresult {r}')


class _CAI:
    def __init__(self, ptr, n): self.__cuda_array_interface__ = dict(shape=(n,), typestr='|u1', data=(ptr, False), version=3)


class VMMBuffer:
    def __init__(self, nbytes, device=0):
        torch.zeros(1, device='cuda'); torch.cuda.synchronize()
        self.prop = _Prop(type=1, requestedHandleTypes=0, location=_Loc(1, device))       # PINNED, DEVICE
        g = ctypes.c_size_t(); _chk(_cu.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(self.prop), 0), 'granularity')
        self.size = (nbytes + g.value - 1) // g.value * g.value
        self.ptr = U64(); _chk(_cu.cuMemAddressReserve(ctypes.byref(self.ptr), ctypes.c_size_t(self.size), ctypes.c_size_t(0), U64(0), U64(0)), 'reserve')
        self.handle = None; self.device = device
        self.map()
        self.tensor = torch.as_tensor(_CAI(self.ptr.value, self.size), device='cuda')       # uint8, zero-copy

    def map(self):
        assert self.handle is None
        h = U64(); _chk(_cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(self.size), ctypes.byref(self.prop), U64(0)), 'create')
        _chk(_cu.cuMemMap(self.ptr, ctypes.c_size_t(self.size), ctypes.c_size_t(0), h, U64(0)), 'map')
        acc = _Access(location=_Loc(1, self.device), flags=3)
        _chk(_cu.cuMemSetAccess(self.ptr, ctypes.c_size_t(self.size), ctypes.byref(acc), ctypes.c_size_t(1)), 'access')
        self.handle = h

    def unmap(self):
        if self.handle is None: return
        torch.cuda.synchronize()
        _chk(_cu.cuMemUnmap(self.ptr, ctypes.c_size_t(self.size)), 'unmap')
        _chk(_cu.cuMemRelease(self.handle), 'release'); self.handle = None


def kv_arena(eng):
    """move eng.Kc / eng.Vc into one VMM buffer (same shapes/dtypes, fixed addresses). Call before any graph capture."""
    shp = eng.Kc[0].shape; dt = eng.Kc[0].dtype; n = eng.Kc[0].numel() * eng.Kc[0].element_size(); L = len(eng.Kc)
    eng.Kc = eng.Vc = None; torch.cuda.empty_cache()
    buf = VMMBuffer(2 * L * n)
    flat = buf.tensor
    views = [flat[i * n:(i + 1) * n].view(dt).view(shp) for i in range(2 * L)]
    eng.Kc = views[:L]; eng.Vc = views[L:]
    torch.cuda.empty_cache()
    return buf
