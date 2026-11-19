// Tools megakernel: one launch = one decode step of every attached micromachine (calc_s4 + BBH tools) for all rows.
// CTA = (tool, chunk of 32 rows); lane = row; warps loop over the units of a layer; __syncthreads between layers.
// Exact integer arithmetic (all weights/biases/wires are integers; int32 or int64 per stack from the bound analysis).
// State layout is row-minor ([...][Bp]) so every wire access of a warp is one coalesced row vector.
typedef long long i64;
struct Stack {
  const i64* ptr;      // unit -> first entry (global over the stack's layers), n_units_total + 1
  const int* code;     // entry -> input: <0 window bit -(1 + (o<<16 | s)); >=0 scratch row
  const void* w;       // entry weights (int32 or int64)
  const void* b;       // unit biases (int32 or int64)
  const int* lay;      // per layer: first unit, n units, dst scratch row (-1 = output layer)
  void* scr;           // scratch wires [n_scr][Bp] (int32 or int64)
  int nlay, i64m;      // i64m: int64 wires (expr stack 1, fp64 reference); else int32
  float thresh;        // exactness guard: the reference is exact iff every |partial sum| < thresh (2^24 fp32, 2^53 fp64)
  int pad;
};
struct Tool {
  int K, W, NS, dS, key, nout, outoff, pad;
  const unsigned char* symtab;   // [V][NS]
  unsigned char* symring;        // [K][NS][Bp]
  float* Sring;                  // [W][dS][Bp]  (fp32, as the reference stores S)
  unsigned char* keyring;        // [W][Bp]
  int* hs; int* hw;              // per-row ring heads
  Stack s1, s2;
};

template <typename T>
__device__ __forceinline__ T in_val(const Tool& t, const Stack& st, int code, int h, int r, int Bp) {
  if (code < 0) {
    int c = -code - 1; int o = c >> 16, s = c & 0xffff;
    int slot = h - o; slot += (slot < 0) ? t.K : 0;
    return (T)t.symring[((size_t)slot * t.NS + s) * Bp + r];   // own rows only, written before a __syncthreads (no .nc)
  }
  return ((const T*)st.scr)[(size_t)code * Bp + r];
}

template <typename T>
__device__ void run_stack(const Tool& t, const Stack& st, int h, int r, bool live, int Bp, float* outS, float* oute,
                          int nw, int warp, int* flag, int ti) {
  const T* W = (const T*)st.w; const T* Bi = (const T*)st.b; T* scr = (T*)st.scr;
  for (int l = 0; l < st.nlay; l++) {
    int u0 = st.lay[3 * l], nu = st.lay[3 * l + 1], dst = st.lay[3 * l + 2];
    for (int u = warp; u < nu; u += nw) {
      i64 e0 = st.ptr[u0 + u], e1 = st.ptr[u0 + u + 1];
      T acc = Bi[u0 + u];
      // reference = (sum of products, any order) + bias: every partial sum lies in [N, P] (sums of the negative /
      // positive products); exact iff P, |N| and |result| < 2^24 (2^53 fp64). fp32 P/N: conservative via thresh margin.
      float P = 0.f, N = 0.f;
      #pragma unroll 4
      for (i64 e = e0; e < e1; e++) {
        T wv = W[e]; T x = in_val<T>(t, st, st.code[e], h, r, Bp);
        T pr = wv * x; acc += pr; float f = (float)pr;
        P += f > 0.f ? f : 0.f; N -= f < 0.f ? f : 0.f;
      }
      if (live && !(P < st.thresh && N < st.thresh && fabsf((float)acc) < st.thresh)) atomicOr(flag, 1 << ti);
      if (live) {
        if (dst >= 0) scr[(size_t)(dst + u) * Bp + r] = acc > 0 ? acc : (T)0;
        else if (outS) outS[(size_t)u * Bp + r] = (float)(double)acc;
        else oute[u] = (float)(double)acc;
      }
    }
    __syncthreads();
  }
}

extern "C" __global__ void __launch_bounds__(256) tk_step(const Tool* tools, int ntools, const i64* tok, int B, int Bp,
                                                         float* out_e, int nout_total, int nchunk, int* flag) {
  const int ti = blockIdx.x / nchunk, ch = blockIdx.x % nchunk;
  const Tool& t = tools[ti];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int r = ch * 32 + lane;
  const bool live = r < B;
  const int rr = live ? r : Bp - 1;                       // dead lanes compute on the dummy row Bp-1 (never read)
  int h = t.hs[rr] + 1; if (h >= t.K) h = 0;
  int hw = t.hw[rr] + 1; if (hw >= t.W) hw = 0;
  i64 v = tok[rr];
  // A: reader (table lookup) -> sym ring slot h, key ring slot hw
  for (int s = warp; s < t.NS; s += nw) t.symring[((size_t)h * t.NS + s) * Bp + rr] = t.symtab[v * t.NS + s];
  if (warp == 0) t.keyring[(size_t)hw * Bp + rr] = t.symtab[v * t.NS + t.key];
  __syncthreads();
  // B: stack 1 -> S ring slot hw
  float* Sslot = t.Sring + (size_t)hw * t.dS * Bp;
  if (t.s1.i64m) run_stack<i64>(t, t.s1, h, rr, true, Bp, Sslot, nullptr, nw, warp, flag, ti);
  else run_stack<int>(t, t.s1, h, rr, true, Bp, Sslot, nullptr, nw, warp, flag, ti);
  // C: hop = most recent key slot within W (else newest), copy S into stack-2 scratch rows [0, dS)
  int sel = hw;
  for (int d = 0; d < t.W; d++) {
    int sl = hw - d; sl += (sl < 0) ? t.W : 0;
    if (t.keyring[(size_t)sl * Bp + rr]) { sel = sl; break; }
  }
  for (int j = warp; j < t.dS; j += nw) {
    float c = t.Sring[((size_t)sel * t.dS + j) * Bp + rr];
    if (t.s2.i64m) ((i64*)t.s2.scr)[(size_t)j * Bp + rr] = (i64)c;
    else ((int*)t.s2.scr)[(size_t)j * Bp + rr] = (int)c;
  }
  __syncthreads();
  // D: stack 2 -> e (row-major out_e[r][outoff + j])
  float* oe = out_e + (size_t)rr * nout_total + t.outoff;
  if (t.s2.i64m) run_stack<i64>(t, t.s2, h, rr, live, Bp, nullptr, oe, nw, warp, flag, ti);
  else run_stack<int>(t, t.s2, h, rr, live, Bp, nullptr, oe, nw, warp, flag, ti);
  if (threadIdx.x < 32 && live) { t.hs[rr] = h; t.hw[rr] = hw; }
}
