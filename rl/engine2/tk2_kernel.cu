// Tools megakernel v2 (event-driven / push). One launch = one decode step of every micromachine for all rows.
// CTA = (tool, chunk of 32 rows), lane = row. Wires are ~1% nonzero on real text, so instead of gathering every fan-in
// entry (17M per row) each nonzero wire PUSHES w*x into its consumers' accumulators (68k per row). Accumulators are
// chunk-local [unit][32] arrays; a per-(chunk,unit) stamp marks units that received a push this step (others hold 0),
// so only touched units are read+reset. All integer adds are exact in any order; the guard tracks, per unit, the sums
// of positive / negative products (P, N) -- the fp32 (fp64) reference (sum of products in any order, then + bias) is
// exact iff P, N and |result| stay below 2^24 (2^53); P/N in fp32 with a 1% margin = conservative.
typedef long long i64;
struct Stack {
  const i64* ptr;        // source -> first push entry; sources = [window bits (K*NS)] [C (dS, stack 2)] [units]
  const int* cons;       // entry -> consumer unit id
  const void* w;         // entry weight (int32 / int64)
  const void* b;         // unit bias
  const int* lay;        // per layer: first unit, n units
  void* acc;             // [nchunk][nU][32] (int32 / int64)
  float* PN;             // [nchunk][nU][2][32]
  int* stamp;            // [nchunk][nU]
  int nlay, nU, nsrc_in, i64m;   // nsrc_in = number of non-unit sources (window + C)
  float thresh; int pad;
};
struct Tool {
  int K, W, NS, dS, key, nout, outoff, maxsym;
  const short* symlist;          // [V][maxsym] symbol ids of each token (-1 padded)
  const unsigned char* keytab;   // [V] key flag
  int* tokring;                  // [K][Bp] token ids
  float* Sring;                  // [W][dS][Bp]
  unsigned char* keyring;        // [W][Bp]
  int* hs; int* hw;              // per-row ring heads
  Stack s1, s2;
};

__device__ __forceinline__ void red_add(int* a, int v) { atomicAdd(a, v); }
__device__ __forceinline__ void red_add(i64* a, i64 v) { atomicAdd((unsigned long long*)a, (unsigned long long)v); }

template <typename T>
__device__ __forceinline__ void push_src(const Stack& st, i64 src, T x, int chunk, int lane, int step) {
  const T* W = (const T*)st.w;
  T* acc = (T*)st.acc + (size_t)chunk * st.nU * 32;
  float* PN = st.PN + (size_t)chunk * st.nU * 64;
  int* stamp = st.stamp + (size_t)chunk * st.nU;
  i64 e0 = st.ptr[src], e1 = st.ptr[src + 1];
  for (i64 e = e0; e < e1; e++) {
    int c = st.cons[e]; T p = W[e] * x;
    if (x != 0) {
      red_add(acc + (size_t)c * 32 + lane, p);
      float f = (float)p;
      if (f > 0.f) atomicAdd(PN + (size_t)c * 64 + lane, f); else atomicAdd(PN + (size_t)c * 64 + 32 + lane, -f);
      stamp[c] = step;
    }
  }
}

// one stack: optional extra input sources already pushed by the caller; walk layers; out-layer values -> out(u, v)
template <typename T, typename OUT>
__device__ void run_stack(const Stack& st, int chunk, int lane, int warp, int nw, int step, bool live, int* flag, int ti, OUT out) {
  T* acc = (T*)st.acc + (size_t)chunk * st.nU * 32;
  float* PN = st.PN + (size_t)chunk * st.nU * 64;
  const int* stamp = st.stamp + (size_t)chunk * st.nU;
  const T* Bi = (const T*)st.b;
  for (int l = 0; l < st.nlay; l++) {
    __threadfence_block(); __syncthreads();
    int u0 = st.lay[2 * l], nu = st.lay[2 * l + 1];
    bool is_out = l == st.nlay - 1;
    for (int u = u0 + warp; u < u0 + nu; u += nw) {
      T v = Bi[u];
      if (__ldcg(stamp + u) == step) {
        T a = __ldcg(acc + (size_t)u * 32 + lane); acc[(size_t)u * 32 + lane] = 0;
        float P = __ldcg(PN + (size_t)u * 64 + lane), N = __ldcg(PN + (size_t)u * 64 + 32 + lane);
        PN[(size_t)u * 64 + lane] = 0.f; PN[(size_t)u * 64 + 32 + lane] = 0.f;
        v += a;
        if (live && !(P < st.thresh && N < st.thresh && fabsf((float)v) < st.thresh)) atomicOr(flag, 1 << ti);
      } else if (live && !(fabsf((float)v) < st.thresh)) atomicOr(flag, 1 << ti);
      if (is_out) { out(u - u0, v); continue; }
      if (v < 0 || !live) v = 0;
      if (__any_sync(0xffffffffu, v != 0)) push_src<T>(st, (i64)st.nsrc_in + u, v, chunk, lane, step);
    }
  }
  __threadfence_block(); __syncthreads();
}

extern "C" __global__ void __launch_bounds__(256) tk2_step(const Tool* tools, int ntools, const i64* tok, int B, int Bp,
                                                           float* out_e, int nout_total, int nchunk, int* flag, const int* stepp) {
  const int ti = blockIdx.x / nchunk, ch = blockIdx.x % nchunk;
  const Tool& t = tools[ti];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int r = ch * 32 + lane;
  const bool live = r < B;
  const int rr = live ? r : Bp - 1;
  const int step = *stepp;
  int h = t.hs[rr] + 1; if (h >= t.K) h = 0;
  int hw = t.hw[rr] + 1; if (hw >= t.W) hw = 0;
  const i64 v = tok[rr];
  if (warp == 0) {
    t.tokring[(size_t)h * Bp + rr] = (int)v;
    t.keyring[(size_t)hw * Bp + rr] = t.keytab[v];
  }
  __threadfence_block(); __syncthreads();
  // window pushes (stack 1): each lane pushes its own row's nonzero window bits (offset o, symbol s) -> source o*NS+s
  const bool i1 = t.s1.i64m;
  for (int o = warp; o < t.K; o += nw) {
    int sl = h - o; sl += (sl < 0) ? t.K : 0;
    int tk = t.tokring[(size_t)sl * Bp + rr];
    for (int j = 0; j < t.maxsym && live; j++) {
      int s = t.symlist[(size_t)tk * t.maxsym + j];
      if (s < 0) break;
      if (i1) push_src<i64>(t.s1, (i64)o * t.NS + s, (i64)1, ch, lane, step);
      else push_src<int>(t.s1, (i64)o * t.NS + s, 1, ch, lane, step);
    }
  }
  float* Sslot = t.Sring + (size_t)hw * t.dS * Bp;
  auto outS = [&](int u, auto val) { if (live) Sslot[(size_t)u * Bp + rr] = (float)(double)val; };
  if (i1) run_stack<i64>(t.s1, ch, lane, warp, nw, step, live, flag, ti, outS);
  else run_stack<int>(t.s1, ch, lane, warp, nw, step, live, flag, ti, outS);
  // hop: most recent key slot within W (else newest)
  int sel = hw;
  for (int d = 0; d < t.W; d++) {
    int sl = hw - d; sl += (sl < 0) ? t.W : 0;
    if (t.keyring[(size_t)sl * Bp + rr]) { sel = sl; break; }
  }
  // stack 2 inputs: window bits (sources o*NS+s) and C (sources K*NS + j)
  const bool i2 = t.s2.i64m;
  for (int o = warp; o < t.K; o += nw) {
    int sl = h - o; sl += (sl < 0) ? t.K : 0;
    int tk = t.tokring[(size_t)sl * Bp + rr];
    for (int j = 0; j < t.maxsym && live; j++) {
      int s = t.symlist[(size_t)tk * t.maxsym + j];
      if (s < 0) break;
      if (i2) push_src<i64>(t.s2, (i64)o * t.NS + s, (i64)1, ch, lane, step);
      else push_src<int>(t.s2, (i64)o * t.NS + s, 1, ch, lane, step);
    }
  }
  const i64 cbase = (i64)t.K * t.NS;
  for (int j = warp; j < t.dS; j += nw) {
    float c = live ? t.Sring[((size_t)sel * t.dS + j) * Bp + rr] : 0.f;
    if (__any_sync(0xffffffffu, c != 0.f)) {
      if (i2) push_src<i64>(t.s2, cbase + j, (i64)c, ch, lane, step);
      else push_src<int>(t.s2, cbase + j, (int)c, ch, lane, step);
    }
  }
  float* oe = out_e + (size_t)rr * nout_total + t.outoff;
  auto oute = [&](int u, auto val) { if (live) oe[u] = (float)(double)val; };
  if (i2) run_stack<i64>(t.s2, ch, lane, warp, nw, step, live, flag, ti, oute);
  else run_stack<int>(t.s2, ch, lane, warp, nw, step, live, flag, ti, oute);
  if (threadIdx.x < 32 && live) { t.hs[rr] = h; t.hw[rr] = hw; }
}
