// Fused log-prob head: one CTA per token row over the bf16 logits (cuBLAS h @ E^T output, as F.linear gives).
// fwd: z = float(logit)/T; lse = max + log sum exp(z - max); lp = z[t] - lse      (online, one pass, fp32)
// bwd: dlogit = bf16( g * (onehot(t) - exp(z - lse)) / T )   written in place over the logits
#include <cuda_bf16.h>
#define INFINITY __int_as_float(0x7f800000)
extern "C" __global__ void __launch_bounds__(512) lp_fwd(const __nv_bfloat16* lg, const long long* tgt, float* lp, float* lse,
                                                        int V, float invT) {
  const int n = blockIdx.x; const __nv_bfloat16* row = lg + (size_t)n * V;
  float m = -INFINITY, s = 0.f;
  for (int v = threadIdx.x * 2; v < V; v += blockDim.x * 2) {
    __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(row + v);
    float a = __bfloat162float(p.x) * invT, b = __bfloat162float(p.y) * invT;
    float mn = fmaxf(m, fmaxf(a, b));
    s = s * __expf(m - mn) + __expf(a - mn) + __expf(b - mn); m = mn;
  }
  __shared__ float sm[32], ss[32];
  for (int o = 16; o; o >>= 1) {
    float m2 = __shfl_xor_sync(0xffffffffu, m, o), s2 = __shfl_xor_sync(0xffffffffu, s, o);
    float mn = fmaxf(m, m2); s = s * __expf(m - mn) + s2 * __expf(m2 - mn); m = mn;
  }
  int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  if (l == 0) { sm[w] = m; ss[w] = s; }
  __syncthreads();
  if (w == 0) {
    m = l < nw ? sm[l] : -INFINITY; s = l < nw ? ss[l] : 0.f;
    for (int o = 16; o; o >>= 1) {
      float m2 = __shfl_xor_sync(0xffffffffu, m, o), s2 = __shfl_xor_sync(0xffffffffu, s, o);
      float mn = fmaxf(m, m2); s = (mn == -INFINITY) ? 0.f : s * __expf(m - mn) + s2 * __expf(m2 - mn); m = mn;
    }
    if (l == 0) {
      float L = m + logf(s); lse[n] = L;
      lp[n] = __bfloat162float(row[tgt[n]]) * invT - L;
    }
  }
}
extern "C" __global__ void __launch_bounds__(512) lp_bwd(__nv_bfloat16* lg, const long long* tgt, const float* lse,
                                                        const float* g, int V, float invT) {
  const int n = blockIdx.x; __nv_bfloat16* row = lg + (size_t)n * V;
  const float L = lse[n], gn = g[n]; const long long t = tgt[n];
  for (int v = threadIdx.x * 2; v < V; v += blockDim.x * 2) {
    __nv_bfloat162 p = *reinterpret_cast<__nv_bfloat162*>(row + v);
    float a = __bfloat162float(p.x) * invT, b = __bfloat162float(p.y) * invT;
    float da = gn * ((v == t ? 1.f : 0.f) - __expf(a - L)) * invT;
    float db = gn * ((v + 1 == t ? 1.f : 0.f) - __expf(b - L)) * invT;
    *reinterpret_cast<__nv_bfloat162*>(row + v) = __floats2bfloat162_rn(da, db);
  }
}
