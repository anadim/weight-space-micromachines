// Tools megakernel v3: differential (baseline-relative) event-driven evaluation.
// Baseline z0 = every stack evaluated on the always-on window (symbols that are 1 for every token, e.g. ANY) and C = 0
// (precomputed exactly on the host). At a step only DEVIATIONS from the baseline are pushed: window bits beyond the
// always-on set (+1; the empty-slot sentinel pushes -1 for the always-on bits), C values (full, stack 2), and hidden
// units whose value differs from z0. Touched units (stamp == step) are recomputed: pre = pre0 + dacc, x = relu(pre),
// dx = x - z0 pushed if any lane is nonzero. Untouched units keep the baseline (verified exact at build time).
// Exactness guard (the fp32/fp64 reference sums products in any order, then adds the bias): P = sum of positive products,
// N = |sum of negative products|; both exact int64 (baseline + deltas; inputs other than C are >= 0, so P/N are linear
// in them; C products are classified by their own sign). Reference exact iff P, N, |pre| < 2^24 (2^53 for fp64).
typedef long long i64;
struct Stack {
  const i64* ptr; const int* cons; const void* w;   // push lists over sources [window K*NS][C dS][units]
  const void* pre0; const void* z0;                  // baseline pre-activation / value per unit
  const i64* P0; const i64* N0;                      // baseline positive / negative product sums
  const int* lay;                                    // per layer: first unit, n units
  void* acc; i64* P; i64* N;                         // [nchunk][nU][32] deltas
  int* stamp;                                        // [nchunk][nU]
  int nlay, nU, nsrc_in, i64m;
  i64 thresh;
};
struct Tool {
  int K, W, NS, dS, key, nout, outoff, maxsym;
  const short* symlist;   // [V+1][maxsym]: s >= 0: +1 on symbol s; s <= -2: -1 on symbol (-s-2); -1 end. Row V = empty slot
  const unsigned char* keytab;
  int* tokring; float* Sring; unsigned char* keyring; int* hs; int* hw;
  const unsigned char* skip;     // [Bp] per-row skip flag (token filter): state untouched, outputs 0
  const i64* tok2;               // [Bp] per-tool token override (remapped feed ids), or null = the shared tokens
  Stack s1, s2;
};

__device__ __forceinline__ void red_add(int* a, int v) { atomicAdd(a, v); }
__device__ __forceinline__ void red_add(i64* a, i64 v) { atomicAdd((unsigned long long*)a, (unsigned long long)v); }

// push x (per lane; 0 = nothing) from source src; signed_in: source may be negative (C) -> P/N by product sign
template <typename T>
__device__ __forceinline__ void push_range(const Stack& st, i64 e0, i64 e1, T x, int chunk, int lane, int step, bool signed_in) {
  // pre = pre0 + dP - dN: every product goes to P (positive part) or N (|negative part|); no separate accumulator
  if (x == 0) return;
  const T* W = (const T*)st.w;
  size_t base = (size_t)chunk * st.nU;
  i64* P = st.P + base * 32 + lane; i64* N = st.N + base * 32 + lane; int* stamp = st.stamp + base;
  i64 e = e0;
  for (; e + 4 <= e1; e += 4) {
    int c[4]; T w[4];
    #pragma unroll
    for (int q = 0; q < 4; q++) { c[q] = st.cons[e + q]; w[q] = W[e + q]; }
    #pragma unroll
    for (int q = 0; q < 4; q++) {
      i64 p = (i64)w[q] * (i64)x; bool pos = signed_in ? (p > 0) : (w[q] > 0);
      red_add((pos ? P : N) + (size_t)c[q] * 32, pos ? p : -p);
      stamp[c[q]] = step;
    }
  }
  for (; e < e1; e++) {
    int c = st.cons[e]; T w = W[e];
    i64 p = (i64)w * (i64)x; bool pos = signed_in ? (p > 0) : (w > 0);
    red_add((pos ? P : N) + (size_t)c * 32, pos ? p : -p);
    stamp[c] = step;
  }
}

template <typename T>
__device__ __forceinline__ void push_src(const Stack& st, i64 src, T x, int chunk, int lane, int step, bool signed_in) {
  if (x == 0) return;
  push_range<T>(st, st.ptr[src], st.ptr[src + 1], x, chunk, lane, step, signed_in);
}

template <typename T>
__device__ __forceinline__ void push_window(const Tool& t, const Stack& st, int h, int rr, int Bp, int chunk, int lane,
                                            int warp, int nw, int step, bool live) {
  if (!live) return;
  for (int o = warp; o < t.K; o += nw) {
    int sl = h - o; sl += (sl < 0) ? t.K : 0;
    int tk = t.tokring[(size_t)sl * Bp + rr];
    for (int j = 0; j < t.maxsym; j++) {
      int s = t.symlist[(size_t)tk * t.maxsym + j];
      if (s == -1) break;
      if (s >= 0) push_src<T>(st, (i64)o * t.NS + s, (T)1, chunk, lane, step, false);
      else push_src<T>(st, (i64)o * t.NS + (-s - 2), (T)-1, chunk, lane, step, false);
    }
  }
}

#define TILE 4096
#ifndef NARROW
#define NARROW 0
#endif
template <typename T>
__device__ void run_stack(const Stack& st, int chunk, int lane, int warp, int nw, int step, bool live, int* flag, int ti,
                          float* out, size_t ostride, int* slist, int* scount) {
  size_t base = (size_t)chunk * st.nU;
  i64* P = st.P + base * 32; i64* N = st.N + base * 32; const int* stamp = st.stamp + base;
  const T* pre0 = (const T*)st.pre0; const T* z0 = (const T*)st.z0;
  for (int l = 0; l < st.nlay; l++) {
    int u0 = st.lay[2 * l], nu = st.lay[2 * l + 1];
    bool is_out = l == st.nlay - 1;
    for (int t0 = 0; t0 < nu; t0 += TILE) {
      int tn = min(TILE, nu - t0);
      __syncthreads();                                   // pushes of earlier layers / previous tile complete
      if (is_out) {                                      // outputs: every unit (baseline or recomputed)
        for (int k = warp; k < tn; k += nw) {
          int u = u0 + t0 + k; T v = pre0[u];
          if (__ldcg(stamp + u) == step) {
            size_t a = (size_t)u * 32 + lane;
            i64 dp = __ldcg(P + a), dn = __ldcg(N + a); P[a] = 0; N[a] = 0;
            i64 pv = (i64)v + dp - dn, pp = st.P0[u] + dp, nn = st.N0[u] + dn;
            if (live && !(pp < st.thresh && nn < st.thresh && pv < st.thresh && -pv < st.thresh)) atomicOr(flag, 1 << ti);
            v = (T)pv;
          }
          if (live) out[(size_t)(t0 + k) * ostride] = (float)(double)v;
        }
        continue;
      }
      if (nu <= NARROW) {                                // narrow layer: read every unit directly (untouched = zeros),
        for (int u = u0 + warp; u < u0 + nu; u += nw) {  // no stamp scan, one barrier per layer
          size_t a = (size_t)u * 32 + lane;
          i64 e0 = st.ptr[st.nsrc_in + u], e1 = st.ptr[st.nsrc_in + u + 1];
          T p0 = pre0[u], zz = z0[u]; i64 P0u = st.P0[u], N0u = st.N0[u];
          i64 dp = __ldcg(P + a), dn = __ldcg(N + a);
          if (dp | dn) { P[a] = 0; N[a] = 0; }
          i64 pv = (i64)p0 + dp - dn, pp = P0u + dp, nn = N0u + dn;
          if (live && !(pp < st.thresh && nn < st.thresh && pv < st.thresh && -pv < st.thresh)) atomicOr(flag, 1 << ti);
          T x = pv > 0 ? (T)pv : (T)0;
          T dx = live ? x - zz : (T)0;
          if (__any_sync(0xffffffffu, dx != 0)) push_range<T>(st, e0, e1, dx, chunk, lane, step, false);
        }
        continue;
      }
      if (threadIdx.x == 0) *scount = 0;
      __syncthreads();
      for (int g = warp * 32; g < tn; g += nw * 32) {   // coalesced stamp scan, lanes = units
        int k = g + lane;
        bool touched = k < tn && __ldcg(stamp + u0 + t0 + k) == step;
        unsigned m = __ballot_sync(0xffffffffu, touched);
        int base_i = 0;
        if (lane == 0 && m) base_i = atomicAdd(scount, __popc(m));
        base_i = __shfl_sync(0xffffffffu, base_i, 0);
        if (touched) slist[base_i + __popc(m & ((1u << lane) - 1))] = u0 + t0 + k;
      }
      __syncthreads();
      int nt = *scount;
      for (int i = warp; i < nt; i += nw) {             // touched units spread over all warps, lanes = rows
        int u = slist[i]; size_t a = (size_t)u * 32 + lane;
        i64 dp = __ldcg(P + a), dn = __ldcg(N + a); P[a] = 0; N[a] = 0;
        i64 pv = (i64)pre0[u] + dp - dn, pp = st.P0[u] + dp, nn = st.N0[u] + dn;
        if (live && !(pp < st.thresh && nn < st.thresh && pv < st.thresh && -pv < st.thresh)) atomicOr(flag, 1 << ti);
        T x = pv > 0 ? (T)pv : (T)0;
        T dx = live ? x - z0[u] : (T)0;
        if (__any_sync(0xffffffffu, dx != 0)) push_src<T>(st, (i64)st.nsrc_in + u, dx, chunk, lane, step, false);
      }
    }
  }
  __syncthreads();
}

extern "C" __global__ void __launch_bounds__(512) tk3_step(const Tool* tools, int ntools, const i64* tok, int B, int Bp,
                                                           float* out_e, int nout_total, int nchunk, int* flag, const int* stepp, int r0) {
  __shared__ int slist[TILE]; __shared__ int scount;
  const int ti = blockIdx.x / nchunk, cl = blockIdx.x % nchunk, ch = (r0 >> 5) + cl;   // r0: first row (multiple of 32)
  const Tool& t = tools[ti];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int r = r0 + cl * 32 + lane;
  const bool live0 = cl * 32 + lane < B;
  const int rr = live0 ? r : Bp - 1;
  const bool sk = live0 && t.skip[rr];
  const bool live = live0 && !sk;                    // a skipped row behaves as a dead lane (no state change)
  const int step = *stepp;
  int h = t.hs[rr] + 1; if (h >= t.K) h = 0;
  int hw = t.hw[rr] + 1; if (hw >= t.W) hw = 0;
  if (warp == 0 && live) {
    i64 v = t.tok2 ? t.tok2[rr] : tok[rr];
    t.tokring[(size_t)h * Bp + rr] = (int)v;
    t.keyring[(size_t)hw * Bp + rr] = t.keytab[v];
  }
  __syncthreads();
  if (t.s1.i64m) push_window<i64>(t, t.s1, h, rr, Bp, ch, lane, warp, nw, step, live);
  else push_window<int>(t, t.s1, h, rr, Bp, ch, lane, warp, nw, step, live);
  float* Sslot = t.Sring + (size_t)hw * t.dS * Bp + rr;
  if (t.s1.i64m) run_stack<i64>(t.s1, ch, lane, warp, nw, step, live, flag, ti, Sslot, Bp, slist, &scount);
  else run_stack<int>(t.s1, ch, lane, warp, nw, step, live, flag, ti, Sslot, Bp, slist, &scount);
  int sel = hw;
  for (int d = 0; d < t.W; d++) {
    int sl = hw - d; sl += (sl < 0) ? t.W : 0;
    if (t.keyring[(size_t)sl * Bp + rr]) { sel = sl; break; }
  }
  if (t.s2.i64m) push_window<i64>(t, t.s2, h, rr, Bp, ch, lane, warp, nw, step, live);
  else push_window<int>(t, t.s2, h, rr, Bp, ch, lane, warp, nw, step, live);
  const i64 cbase = (i64)t.K * t.NS;
  for (int j = warp; j < t.dS; j += nw) {
    float c = live ? t.Sring[((size_t)sel * t.dS + j) * Bp + rr] : 0.f;
    if (__any_sync(0xffffffffu, c != 0.f)) {
      if (t.s2.i64m) push_src<i64>(t.s2, cbase + j, (i64)c, ch, lane, step, true);
      else push_src<int>(t.s2, cbase + j, (int)c, ch, lane, step, true);
    }
  }
  float* oe = out_e + (size_t)rr * nout_total + t.outoff;
  if (t.s2.i64m) run_stack<i64>(t.s2, ch, lane, warp, nw, step, live, flag, ti, oe, 1, slist, &scount);
  else run_stack<int>(t.s2, ch, lane, warp, nw, step, live, flag, ti, oe, 1, slist, &scount);
  if (sk) for (int j = threadIdx.x >> 5; j < t.nout; j += blockDim.x >> 5) oe[j] = 0.f;
  if (threadIdx.x < 32 && live) { t.hs[rr] = h; t.hw[rr] = hw; }
}
