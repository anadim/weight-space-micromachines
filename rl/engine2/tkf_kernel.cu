// Tools megakernel, FLOAT mode (trainable implants): same structure as tk3 (CTA = tool x 32-row chunk, lane = row,
// event-driven pushes relative to a baseline, touched-unit lists), but weights / reader values / wires are fp32,
// accumulation with fp32 atomics (order-dependent rounding: numerically, not bitwise, equal to the PyTorch stack),
// reader values are any value in [0,1] (soft reader), and the hop is the reference softmax(BIG*key - GAMMA*dist) @ S.
// All weight arrays are refreshed from the live parameters before each rollout (ToolsKernelF.refresh()).
typedef long long i64;
#define INFINITY __int_as_float(0x7f800000)
struct StackF {
  const i64* ptr; const int* cons; const void* w;    // push lists over sources [window K*NS][C dS][units]
  const void* pre0; const void* z0;                  // baseline pre-activation / value per unit (zero window, C = 0)
  const int* lay;                                    // per layer: first unit, n units
  void* acc;                                         // [nchunk][nU][32] deltas (T)
  int* stamp;                                        // [nchunk][nU]
  int nlay, nU, nsrc_in, f64;                        // f64: double wires (expr stack 1, whose reference runs in fp64)
};
struct ToolF {
  int K, W, NS, dS, key, nout, outoff, maxsym;
  const short* symidx; const float* symval;          // [V+1][maxsym] nonzero reader outputs of each token (-1 end)
  const float* keytab;                               // [V+1] reader value of the key symbol
  int* tokring; float* Sring; float* keyring; int* hs; int* hw;
  const unsigned char* skip;
  StackF s1, s2;
};
#define TILE 4096
#define BIGF 1e6f
#define GAMMAF 300.f

template <typename T>
__device__ __forceinline__ void push_range(const StackF& st, i64 e0, i64 e1, T x, int chunk, int lane, int step) {
  if (x == (T)0) return;
  size_t base = (size_t)chunk * st.nU;
  T* acc = (T*)st.acc + base * 32 + lane; int* stamp = st.stamp + base; const T* W = (const T*)st.w;
  i64 e = e0;
  for (; e + 4 <= e1; e += 4) {
    int c[4]; T w[4];
    #pragma unroll
    for (int q = 0; q < 4; q++) { c[q] = st.cons[e + q]; w[q] = W[e + q]; }
    #pragma unroll
    for (int q = 0; q < 4; q++) { atomicAdd(acc + (size_t)c[q] * 32, w[q] * x); stamp[c[q]] = step; }
  }
  for (; e < e1; e++) { int c = st.cons[e]; atomicAdd(acc + (size_t)c * 32, W[e] * x); stamp[c] = step; }
}

__device__ __forceinline__ void push_src(const StackF& st, i64 src, double x, int chunk, int lane, int step) {
  if (x == 0.0) return;
  if (st.f64) push_range<double>(st, st.ptr[src], st.ptr[src + 1], x, chunk, lane, step);
  else push_range<float>(st, st.ptr[src], st.ptr[src + 1], (float)x, chunk, lane, step);
}

__device__ __forceinline__ void push_window(const ToolF& t, const StackF& st, int h, int rr, int Bp, int chunk, int lane,
                                            int warp, int nw, int step, bool live) {
  if (!live) return;
  for (int o = warp; o < t.K; o += nw) {
    int sl = h - o; sl += (sl < 0) ? t.K : 0;
    int tk = t.tokring[(size_t)sl * Bp + rr];
    for (int j = 0; j < t.maxsym; j++) {
      int s = t.symidx[(size_t)tk * t.maxsym + j];
      if (s < 0) break;
      push_src(st, (i64)o * t.NS + s, t.symval[(size_t)tk * t.maxsym + j], chunk, lane, step);
    }
  }
}

template <typename T>
__device__ void run_stack_t(const StackF& st, int chunk, int lane, int warp, int nw, int step, bool live,
                            float* out, size_t ostride, int* slist, int* scount) {
  size_t base = (size_t)chunk * st.nU;
  T* acc = (T*)st.acc + base * 32; const int* stamp = st.stamp + base;
  const T* pre0 = (const T*)st.pre0; const T* z0 = (const T*)st.z0;
  for (int l = 0; l < st.nlay; l++) {
    int u0 = st.lay[2 * l], nu = st.lay[2 * l + 1];
    bool is_out = l == st.nlay - 1;
    for (int t0 = 0; t0 < nu; t0 += TILE) {
      int tn = min(TILE, nu - t0);
      __syncthreads();
      if (is_out) {
        for (int k = warp; k < tn; k += nw) {
          int u = u0 + t0 + k; T v = pre0[u];
          if (__ldcg(stamp + u) == step) { size_t a = (size_t)u * 32 + lane; v += __ldcg(acc + a); acc[a] = (T)0; }
          if (live) out[(size_t)(t0 + k) * ostride] = (float)v;
        }
        continue;
      }
      if (threadIdx.x == 0) *scount = 0;
      __syncthreads();
      for (int g = warp * 32; g < tn; g += nw * 32) {
        int k = g + lane;
        bool touched = k < tn && __ldcg(stamp + u0 + t0 + k) == step;
        unsigned m = __ballot_sync(0xffffffffu, touched);
        int bi = 0;
        if (lane == 0 && m) bi = atomicAdd(scount, __popc(m));
        bi = __shfl_sync(0xffffffffu, bi, 0);
        if (touched) slist[bi + __popc(m & ((1u << lane) - 1))] = u0 + t0 + k;
      }
      __syncthreads();
      int nt = *scount;
      for (int i = warp; i < nt; i += nw) {
        int u = slist[i]; size_t a = (size_t)u * 32 + lane;
        T d = __ldcg(acc + a); acc[a] = (T)0;
        T pv = pre0[u] + d; T x = pv > (T)0 ? pv : (T)0;
        T dx = live ? x - z0[u] : (T)0;
        if (__any_sync(0xffffffffu, dx != (T)0)) push_range<T>(st, st.ptr[st.nsrc_in + u], st.ptr[st.nsrc_in + u + 1], dx, chunk, lane, step);
      }
    }
  }
  __syncthreads();
}

__device__ void run_stack(const StackF& st, int chunk, int lane, int warp, int nw, int step, bool live,
                          float* out, size_t ostride, int* slist, int* scount) {
  if (st.f64) run_stack_t<double>(st, chunk, lane, warp, nw, step, live, out, ostride, slist, scount);
  else run_stack_t<float>(st, chunk, lane, warp, nw, step, live, out, ostride, slist, scount);
}

extern "C" __global__ void __launch_bounds__(512) tkf_step(const ToolF* tools, int ntools, const i64* tok, int B, int Bp,
                                                           float* out_e, int nout_total, int nchunk, const int* stepp, int r0) {
  __shared__ int slist[TILE]; __shared__ int scount;
  const int ti = blockIdx.x / nchunk, cl = blockIdx.x % nchunk, ch = (r0 >> 5) + cl;
  const ToolF& t = tools[ti];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int r = r0 + cl * 32 + lane;
  const bool live0 = cl * 32 + lane < B;
  const int rr = live0 ? r : Bp - 1;
  const bool sk = live0 && t.skip[rr];
  const bool live = live0 && !sk;
  const int step = *stepp;
  int h = t.hs[rr] + 1; if (h >= t.K) h = 0;
  int hw = t.hw[rr] + 1; if (hw >= t.W) hw = 0;
  if (warp == 0 && live) {
    i64 v = tok[rr];
    t.tokring[(size_t)h * Bp + rr] = (int)v;
    t.keyring[(size_t)hw * Bp + rr] = t.keytab[v];
  }
  __syncthreads();
  push_window(t, t.s1, h, rr, Bp, ch, lane, warp, nw, step, live);
  float* Sslot = t.Sring + (size_t)hw * t.dS * Bp + rr;
  run_stack(t.s1, ch, lane, warp, nw, step, live, Sslot, Bp, slist, &scount);
  // hop = softmax over the W slots of BIG*key - GAMMA*dist (dist 0 = newest), weights per lane
  float mx = -INFINITY;
  for (int d = 0; d < t.W; d++) { int sl = hw - d; sl += (sl < 0) ? t.W : 0;
    mx = fmaxf(mx, BIGF * t.keyring[(size_t)sl * Bp + rr] - GAMMAF * d); }
  float den = 0.f;
  for (int d = 0; d < t.W; d++) { int sl = hw - d; sl += (sl < 0) ? t.W : 0;
    den += __expf(BIGF * t.keyring[(size_t)sl * Bp + rr] - GAMMAF * d - mx); }
  push_window(t, t.s2, h, rr, Bp, ch, lane, warp, nw, step, live);
  const i64 cbase = (i64)t.K * t.NS;
  for (int j = warp; j < t.dS; j += nw) {
    float c = 0.f;
    if (live) for (int d = 0; d < t.W; d++) {
      int sl = hw - d; sl += (sl < 0) ? t.W : 0;
      float wgt = __expf(BIGF * t.keyring[(size_t)sl * Bp + rr] - GAMMAF * d - mx);
      if (wgt != 0.f) c += wgt * t.Sring[((size_t)sl * t.dS + j) * Bp + rr];
    }
    c = c / den;
    if (__any_sync(0xffffffffu, c != 0.f)) push_src(t.s2, cbase + j, c, ch, lane, step);
  }
  float* oe = out_e + (size_t)rr * nout_total + t.outoff;
  run_stack(t.s2, ch, lane, warp, nw, step, live, oe, 1, slist, &scount);
  if (sk) for (int j = threadIdx.x >> 5; j < t.nout; j += blockDim.x >> 5) oe[j] = 0.f;
  if (threadIdx.x < 32 && live) { t.hs[rr] = h; t.hw[rr] = hw; }
}
