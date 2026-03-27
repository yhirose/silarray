
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------------------
// Tiled SGEMM — simdgroup_matrix 8×8 MMA with float4 vectorized loads
// ---------------------------------------------------------------------------

struct gemm_params {
  uint M, N, K, lda, ldb, trans_a, trans_b;
  uint batch_stride_a, batch_stride_b, batch_stride_c;
};

template <uint BM, uint BN, uint BK>
void sgemm_impl_(
    device const float* A,
    device const float* B,
    device float*       C,
    constant gemm_params& p,
    threadgroup float* As,
    threadgroup float* Bs,
    uint3 tgid, uint tid, uint sid, uint lane)
{
  constexpr uint N_SM = 2, N_SN = 2;
  constexpr uint TM = BM / N_SM, TN = BN / N_SN;
  constexpr uint FM = TM / 8, FN = TN / 8;
  constexpr uint THREADS = N_SM * N_SN * 32;
  constexpr uint aS = BK + 4;   // float4-aligned padding
  constexpr uint bS = BN + 4;

  // Batch offset
  A += tgid.z * p.batch_stride_a;
  B += tgid.z * p.batch_stride_b;
  C += tgid.z * p.batch_stride_c;

  uint row_tg = tgid.y;
  uint col_tg = tgid.x;

  uint wm = sid / N_SN, wn = sid % N_SN;

  simdgroup_matrix<float, 8, 8> acc[FM][FN];
  for (uint i = 0; i < FM; i++)
    for (uint j = 0; j < FN; j++)
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

  uint row0 = row_tg * BM, col0 = col_tg * BN;
  uint a_rs = p.trans_a ? 1u : p.lda;
  uint a_cs = p.trans_a ? p.lda : 1u;
  uint b_rs = p.trans_b ? 1u : p.ldb;
  uint b_cs = p.trans_b ? p.ldb : 1u;
  bool full_tile = (row0 + BM <= p.M) && (col0 + BN <= p.N);
  uint k_full = (p.K / BK) * BK;

  // -- load macros with float4 vectorization for non-transposed path --------
  #define LOAD_A_FAST                                                       \
    if (!p.trans_a) {                                                        \
      constexpr uint F4 = BK / 4;                                           \
      for (uint i = tid; i < BM * F4; i += THREADS) {                      \
        uint r = i / F4, fc = i % F4;                                       \
        auto v = *reinterpret_cast<device const float4*>(                   \
            &A[(row0 + r) * p.lda + k0 + fc * 4]);                         \
        *reinterpret_cast<threadgroup float4*>(&As[r * aS + fc * 4]) = v;  \
      }                                                                      \
    } else {                                                                 \
      for (uint i = tid; i < BM * BK; i += THREADS) {                      \
        uint r = i / BK, c = i % BK;                                        \
        As[r * aS + c] = A[(k0 + c) * p.lda + row0 + r];                   \
      }                                                                      \
    }

  #define LOAD_B_FAST                                                       \
    if (!p.trans_b) {                                                        \
      constexpr uint F4 = BN / 4;                                           \
      for (uint i = tid; i < BK * F4; i += THREADS) {                      \
        uint r = i / F4, fc = i % F4;                                       \
        auto v = *reinterpret_cast<device const float4*>(                   \
            &B[(k0 + r) * p.ldb + col0 + fc * 4]);                         \
        *reinterpret_cast<threadgroup float4*>(&Bs[r * bS + fc * 4]) = v;  \
      }                                                                      \
    } else {                                                                 \
      for (uint i = tid; i < BK * BN; i += THREADS) {                      \
        uint r = i / BN, c = i % BN;                                        \
        Bs[r * bS + c] = B[(col0 + c) * p.ldb + k0 + r];                   \
      }                                                                      \
    }

  #define LOAD_A_SAFE                                                       \
    for (uint i = tid; i < BM * BK; i += THREADS) {                        \
      uint r = i / BK, c = i % BK;                                          \
      uint gr = row0 + r, gc = k0 + c;                                      \
      As[r * aS + c] = (gr < p.M && gc < p.K)                              \
          ? A[gr * a_rs + gc * a_cs] : 0.0f;                                \
    }

  #define LOAD_B_SAFE                                                       \
    for (uint i = tid; i < BK * BN; i += THREADS) {                        \
      uint r = i / BN, c = i % BN;                                          \
      uint gr = k0 + r, gc = col0 + c;                                      \
      Bs[r * bS + c] = (gr < p.K && gc < p.N)                              \
          ? B[gr * b_rs + gc * b_cs] : 0.0f;                                \
    }

  #define MMA_BLOCK                                                         \
    for (uint kk = 0; kk < BK; kk += 8) {                                  \
      simdgroup_matrix<float, 8, 8> af[FM], bf[FN];                        \
      for (uint i = 0; i < FM; i++)                                         \
        simdgroup_load(af[i], &As[(wm*TM + i*8) * aS + kk], aS);          \
      for (uint j = 0; j < FN; j++)                                         \
        simdgroup_load(bf[j], &Bs[kk * bS + wn*TN + j*8], bS);            \
      for (uint i = 0; i < FM; i++)                                         \
        for (uint j = 0; j < FN; j++)                                       \
          simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);\
    }

  // -- main loop ------------------------------------------------------------
  bool a_full = (row0 + BM <= p.M);
  bool b_full = (col0 + BN <= p.N);

  if (a_full && b_full) {
    for (uint k0 = 0; k0 < k_full; k0 += BK) {
      LOAD_A_FAST  LOAD_B_FAST
      threadgroup_barrier(mem_flags::mem_threadgroup);
      MMA_BLOCK
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else if (b_full) {
    for (uint k0 = 0; k0 < k_full; k0 += BK) {
      LOAD_A_SAFE  LOAD_B_FAST
      threadgroup_barrier(mem_flags::mem_threadgroup);
      MMA_BLOCK
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else if (a_full) {
    for (uint k0 = 0; k0 < k_full; k0 += BK) {
      LOAD_A_FAST  LOAD_B_SAFE
      threadgroup_barrier(mem_flags::mem_threadgroup);
      MMA_BLOCK
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (uint k0 = 0; k0 < k_full; k0 += BK) {
      LOAD_A_SAFE  LOAD_B_SAFE
      threadgroup_barrier(mem_flags::mem_threadgroup);
      MMA_BLOCK
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (k_full < p.K) {
    uint k0 = k_full;
    LOAD_A_SAFE  LOAD_B_SAFE
    threadgroup_barrier(mem_flags::mem_threadgroup);
    MMA_BLOCK
  }

  #undef LOAD_A_FAST
  #undef LOAD_B_FAST
  #undef LOAD_A_SAFE
  #undef LOAD_B_SAFE
  #undef MMA_BLOCK

  // -- store ----------------------------------------------------------------
  for (uint i = 0; i < FM; i++) {
    for (uint j = 0; j < FN; j++) {
      uint r = row0 + wm * TM + i * 8;
      uint c = col0 + wn * TN + j * 8;
      if (r + 8 <= p.M && c + 8 <= p.N) {
        simdgroup_store(acc[i][j], C + r * p.N + c, p.N);
      } else if (r < p.M && c < p.N) {
        // Reuse tail of As for edge scratch (4 simdgroups × 64 floats = 256)
        threadgroup float* sc = As;
        simdgroup_store(acc[i][j], &sc[sid * 64], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
          uint er = r + e / 8, ec = c + e % 8;
          if (er < p.M && ec < p.N) C[er * p.N + ec] = sc[sid * 64 + e];
        }
      }
    }
  }
}

kernel void sgemm_32_(device const float* A [[buffer(0)]],
                      device const float* B [[buffer(1)]],
                      device float* C       [[buffer(2)]],
                      constant gemm_params& p [[buffer(3)]],
                      uint3 tgid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[32 * 20];  // 32 * (16+4)
  threadgroup float Bs[16 * 36];  // 16 * (32+4)
  sgemm_impl_<32, 32, 16>(A, B, C, p, As, Bs, tgid, tid, sid, lane);
}

kernel void sgemm_64_(device const float* A [[buffer(0)]],
                      device const float* B [[buffer(1)]],
                      device float* C       [[buffer(2)]],
                      constant gemm_params& p [[buffer(3)]],
                      uint3 tgid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[64 * 20];  // 64 * (16+4)
  threadgroup float Bs[16 * 68];  // 16 * (64+4)
  sgemm_impl_<64, 64, 16>(A, B, C, p, As, Bs, tgid, tid, sid, lane);
}

// ===========================================================================
// GEMV: y = x * B  where x is 1×K, B is K×N (row-major), output y is 1×N.
// Each thread owns 1 output column, iterates K with 4-row unrolling.
// No threadgroup memory, no barriers. x cached in GPU L1/L2.
// Adjacent threads read adjacent columns of B (coalesced within simdgroup).
// ===========================================================================

kernel void sgemv_f32_(
  device const float* x    [[buffer(0)]],
  device const float* B    [[buffer(1)]],
  device float*       y    [[buffer(2)]],
  constant uint32_t&  N    [[buffer(3)]],
  constant uint32_t&  K    [[buffer(4)]],
  uint gid [[thread_position_in_grid]])
{
  uint col = gid;
  if (col >= N) return;

  float acc = 0.0f;

  // 4-row unrolled loop
  uint k4 = K / 4;
  device const float4* x4 = reinterpret_cast<device const float4*>(x);
  for (uint i = 0; i < k4; i++) {
    float4 xv = x4[i];
    uint base = i * 4;
    acc += xv.x * B[(base    ) * N + col]
         + xv.y * B[(base + 1) * N + col]
         + xv.z * B[(base + 2) * N + col]
         + xv.w * B[(base + 3) * N + col];
  }
  for (uint k = k4 * 4; k < K; k++) {
    acc += x[k] * B[k * N + col];
  }

  y[col] = acc;
}

// GEMV variant unrolled 8 rows per iter so the compiler can pipeline
// 8 independent B loads — measurably faster than the 4-unroll baseline
// on M1 for K=4096. Caller must ensure K % 8 == 0.
// ===========================================================================

kernel void sgemv_k8_f32_(
  device const float* x    [[buffer(0)]],
  device const float* B    [[buffer(1)]],
  device float*       y    [[buffer(2)]],
  constant uint32_t&  N    [[buffer(3)]],
  constant uint32_t&  K    [[buffer(4)]],
  uint gid [[thread_position_in_grid]])
{
  uint col = gid;
  if (col >= N) return;

  device const float4* x4 = reinterpret_cast<device const float4*>(x);

  float acc = 0.0f;

  uint k8 = K / 8;
  for (uint i = 0; i < k8; i++) {
    float4 xv0 = x4[2 * i];
    float4 xv1 = x4[2 * i + 1];
    uint base = i * 8;
    acc += xv0.x * B[(base    ) * N + col]
         + xv0.y * B[(base + 1) * N + col]
         + xv0.z * B[(base + 2) * N + col]
         + xv0.w * B[(base + 3) * N + col]
         + xv1.x * B[(base + 4) * N + col]
         + xv1.y * B[(base + 5) * N + col]
         + xv1.z * B[(base + 6) * N + col]
         + xv1.w * B[(base + 7) * N + col];
  }

  y[col] = acc;
}

// ===========================================================================
// STEEL-pattern SGEMM 64×64 — faithful port of MLX's BlockLoader+BlockMMA
//   All 5 optimizations: function constants, thread_elements(), enum:short,
//   short types, template structs with scope-managed register lifetimes
// ===========================================================================

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// Compile-time integer constant type — provides stronger constant propagation
// hints to the Metal compiler than plain template parameters
template <int N> struct Int { STEEL_CONST int value = N; constexpr operator int() const { return N; } };

// Fragment type matching MLX's BaseMMAFrag
typedef float2 frag_type;

// Function constants — set via MTLFunctionConstantValues at PSO creation.
// NN-only: trans_a/trans_b function constants reserved for future use.
// reserved for future transpose-specialized PSOs.
constant bool fc_trans_a [[function_constant(0)]];
constant bool fc_trans_b [[function_constant(1)]];
constant bool fc_mn_aligned [[function_constant(2)]];
constant bool fc_k_aligned [[function_constant(3)]];

// BlockLoader — cooperative tile loading with pre-computed pointers
template <short BROWS, short BCOLS, short dst_ld, short reduction_dim, short tgp_size>
struct SteelLoader {
  STEEL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  STEEL_CONST short vec_size = n_reads;
  STEEL_CONST short TCOLS = BCOLS / n_reads;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  STEEL_CONST short n_rows = BROWS / TROWS;

  const int src_ld;
  const int tile_stride;
  const short bi;
  const short bj;
  threadgroup float* dst;
  const device float* src;

  struct alignas(16) ReadVec { float v[vec_size]; };

  METAL_FUNC SteelLoader(
      const device float* src_, int src_ld_,
      threadgroup float* dst_,
      ushort simd_group_id, ushort simd_lane_id)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        bi(short(simd_group_id * 32 + simd_lane_id) / TCOLS),
        bj(vec_size * (short(simd_group_id * 32 + simd_lane_id) % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVec*)(&dst[i * dst_ld])) =
          *((const device ReadVec*)(&src[i * src_ld]));
    }
  }

  // Branchless load_safe: all threads issue the same load instruction
  // (invalid threads read src[0]), then mask invalid values to zero.
  // This avoids SIMD divergence on the load path.
  METAL_FUNC void load_safe(short2 tile_dim) const {
    tile_dim -= short2(bj, bi);
    if (tile_dim.x <= 0 || tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS)
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++)
          dst[i * dst_ld + j] = 0.0f;
      return;
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        bool valid = (i < tile_dim.y) && (j < tile_dim.x);
        dst[i * dst_ld + j] = valid ? src[valid ? i * src_ld + j : 0] : 0.0f;
      }
    }
  }

  METAL_FUNC void next() { src += tile_stride; }
};

// BlockMMA — MLX-faithful fragment loading + serpentine tile_matmad
template <short BM, short BN, short BK, short WM, short WN,
          bool tA, bool tB, short lda_tgp, short ldb_tgp>
struct SteelMMA {
  STEEL_CONST short kFrag = 8;
  STEEL_CONST short TM = BM / (kFrag * WM);
  STEEL_CONST short TN = BN / (kFrag * WN);

  STEEL_CONST short A_str_m = tA ? 1 : lda_tgp;
  STEEL_CONST short A_str_k = tA ? lda_tgp : 1;
  STEEL_CONST short B_str_k = tB ? 1 : ldb_tgp;
  STEEL_CONST short B_str_n = tB ? ldb_tgp : 1;

  STEEL_CONST short tile_stride_a = kFrag * A_str_k;
  STEEL_CONST short tile_stride_b = kFrag * B_str_k;

  // Fragment storage (MLX frag_type layout)
  frag_type Atile[TM];
  frag_type Btile[TN];
  frag_type Ctile[TM * TN];

  short sm, sn;
  short As_off, Bs_off;

  METAL_FUNC SteelMMA(ushort sid, ushort lane) {
    short tm = kFrag * short(sid / WN);
    short tn = kFrag * short(sid % WN);

    short qid = short(lane) / 4;
    short fm = (qid & 4) + ((short(lane) / 2) % 4);
    short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;

    sm = fm; sn = fn;
    As_off = (tm + sm) * A_str_m + sn * A_str_k;
    Bs_off = sm * B_str_k + (tn + sn) * B_str_n;
    sm += tm; sn += tn;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM * TN; i++) Ctile[i] = frag_type(0);
  }

  // Fragment load from threadgroup memory
  // For NN case (str_y=1), elements are contiguous → float2 bulk read
  template <typename StrX, typename StrY>
  METAL_FUNC static constexpr void load_frag(
      thread frag_type& dst, const threadgroup float* src, StrX, StrY str_y) {
    if (int(str_y) == 1) {
      dst = *reinterpret_cast<const threadgroup frag_type*>(src);
    } else {
      dst[0] = src[0];
      dst[1] = src[int(str_y)];
    }
  }

  // MLX BaseMMAFrag::mma — frag_type in, simdgroup_matrix inside
  METAL_FUNC static constexpr void frag_mma(
      thread frag_type& D, thread frag_type& A,
      thread frag_type& B, thread frag_type& C) {
    simdgroup_matrix<float, 8, 8> A_mat, B_mat, C_mat;
    reinterpret_cast<thread frag_type&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;
    simdgroup_multiply_accumulate(C_mat, A_mat, B_mat, C_mat);
    D = reinterpret_cast<thread frag_type&>(C_mat.thread_elements());
  }

  // MLX pattern: per kk step — load 1 K-slice, immediately MMA
  METAL_FUNC void mma(const threadgroup float* As, const threadgroup float* Bs) {
    As += As_off;
    Bs += Bs_off;

    // Pre-computed fragment offsets (avoid repeated multiplications)
    constexpr short A_frag_stride = kFrag * WM * A_str_m;
    constexpr short B_frag_stride = kFrag * WN * B_str_n;

    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFrag) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        load_frag(Atile[i], &As[i * A_frag_stride], Int<A_str_m>{}, Int<A_str_k>{});
      }

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        load_frag(Btile[j], &Bs[j * B_frag_stride], Int<B_str_k>{}, Int<B_str_n>{});
      }

      simdgroup_barrier(mem_flags::mem_none);

      STEEL_PRAGMA_UNROLL
      for (short m = 0; m < TM; m++) {
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < TN; n++) {
          short n_serp = (m % 2) ? (TN - 1 - n) : n;
          frag_mma(Ctile[m * TN + n_serp], Atile[m], Btile[n_serp], Ctile[m * TN + n_serp]);
        }
      }

      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  // MLX BaseMMAFrag::store pattern
  METAL_FUNC void store_result(device float* C, int ldd) {
    C += sm * ldd + sn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        *reinterpret_cast<device float2*>(
            &C[(i * kFrag) * WM * ldd + (j * kFrag) * WN]) = Ctile[i * TN + j];
      }
    }
  }

  METAL_FUNC void store_result_safe(device float* C, int ldd, short2 dims) {
    C += sm * ldd + sn;
    dims -= short2(sn, sm);
    if (dims.x <= 0 || dims.y <= 0) return;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        short r = (i * kFrag) * WM;
        short c = (j * kFrag) * WN;
        if (r < dims.y && c < dims.x)
          C[r * ldd + c] = Ctile[i * TN + j][0];
        if (r < dims.y && c + 1 < dims.x)
          C[r * ldd + c + 1] = Ctile[i * TN + j][1];
      }
    }
  }
  // Fused store: apply bias before writing to device memory
  METAL_FUNC void store_result_bias(
      device float* C, int ldd, device const float* bias, uint bias_len,
      short col0) {
    C += sm * ldd + sn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        short c0 = col0 + sn + (j * kFrag) * WN;
        frag_type val = Ctile[i * TN + j];
        val[0] += bias[c0 % bias_len];
        val[1] += bias[(c0 + 1) % bias_len];
        *reinterpret_cast<device float2*>(
            &C[(i * kFrag) * WM * ldd + (j * kFrag) * WN]) = val;
      }
    }
  }
  // Fused store: apply bias + sigmoid before writing to device memory
  METAL_FUNC void store_result_bias_sigmoid(
      device float* C, int ldd, device const float* bias, uint bias_len,
      short col0) {
    C += sm * ldd + sn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        short c0 = col0 + sn + (j * kFrag) * WN;
        frag_type val = Ctile[i * TN + j];
        val[0] += bias[c0 % bias_len];
        val[1] += bias[(c0 + 1) % bias_len];
        val[0] = 1.0f / (1.0f + exp(-val[0]));
        val[1] = 1.0f / (1.0f + exp(-val[1]));
        *reinterpret_cast<device float2*>(
            &C[(i * kFrag) * WM * ldd + (j * kFrag) * WN]) = val;
      }
    }
  }
};

// STEEL gemm body — shared by 64x64 and 32x64 kernel variants.
// BN is always 64. WM and WN are always 2. Only BM varies.
template <short BM>
void steel_gemm_body_(
    device const float* A, device const float* B, device float* C,
    constant gemm_params& p, threadgroup float* As, threadgroup float* Bs,
    uint3 tgid, uint sid, uint lane) {
  constexpr short BN = 64, BK = 16, WM = 2, WN = 2, pad = 4;
  constexpr short tgp_size = WM * WN * 32;

  A += tgid.z * p.batch_stride_a;
  B += tgid.z * p.batch_stride_b;
  C += tgid.z * p.batch_stride_c;

  short swizzle = short(p.trans_a);
  short tiles_n = short(p.N / BN);
  short tiles_m = short(p.M / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short row0 = tid_y * BM, col0 = tid_x * BN;
  A += row0 * int(p.lda);
  B += int(col0);

  constexpr short lda_nn = BK + pad, ldb_nn = BN + pad;
  SteelLoader<BM, BK, lda_nn, true, tgp_size> loader_a(A, int(p.lda), As, sid, ushort(lane));
  SteelLoader<BK, BN, ldb_nn, false, tgp_size> loader_b(B, int(p.ldb), Bs, sid, ushort(lane));
  SteelMMA<BM, BN, BK, WM, WN, false, false, lda_nn, ldb_nn> mma_op(sid, lane);

  int k_iters = fc_k_aligned ? int(p.K >> 4) : int(p.K / BK);
  short tgp_bm = fc_mn_aligned ? BM : min(short(BM), short(p.M - row0));
  short tgp_bn = fc_mn_aligned ? BN : min(short(BN), short(p.N - col0));
  bool is_interior = fc_mn_aligned || (tgp_bm == BM && tgp_bn == BN);

  if (!fc_k_aligned) {
    short lbk = short(p.K) - short(k_iters * BK);
    size_t k_jump_a = size_t(k_iters) * BK;
    size_t k_jump_b = size_t(k_iters) * BK * p.ldb;
    loader_a.src += k_jump_a; loader_b.src += k_jump_b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_safe(short2(lbk, tgp_bm));
    loader_b.load_safe(short2(tgp_bn, lbk));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.src -= k_jump_a + BK; loader_b.src -= k_jump_b + BK * p.ldb;
    loader_a.next(); loader_b.next();
  }

  if (is_interior) {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe(); loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next(); loader_b.next();
    }
  } else {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(short2(BK, tgp_bm));
      loader_b.load_safe(short2(tgp_bn, BK));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next(); loader_b.next();
    }
  }

  C += row0 * int(p.N) + col0;
  if (is_interior) { mma_op.store_result(C, int(p.N)); }
  else { mma_op.store_result_safe(C, int(p.N),
      short2(min(short(BN), short(p.N - col0)), min(short(BM), short(p.M - row0)))); }
}

// STEEL gemm+bias body
template <short BM>
void steel_gemm_bias_body_(
    device const float* A, device const float* B, device float* C,
    constant gemm_params& p, device const float* bias, uint32_t bias_len,
    threadgroup float* As, threadgroup float* Bs,
    uint3 tgid, uint sid, uint lane) {
  constexpr short BN = 64, BK = 16, WM = 2, WN = 2, pad = 4;
  constexpr short tgp_size = WM * WN * 32;

  A += tgid.z * p.batch_stride_a; B += tgid.z * p.batch_stride_b; C += tgid.z * p.batch_stride_c;
  short swizzle = short(p.trans_a);
  short tiles_n = short((p.N + BN - 1) / BN), tiles_m = short((p.M + BM - 1) / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short row0 = tid_y * BM, col0 = tid_x * BN;
  A += row0 * int(p.lda); B += int(col0);
  constexpr short lda_nn = BK + pad, ldb_nn = BN + pad;
  SteelLoader<BM, BK, lda_nn, true, tgp_size> loader_a(A, int(p.lda), As, sid, ushort(lane));
  SteelLoader<BK, BN, ldb_nn, false, tgp_size> loader_b(B, int(p.ldb), Bs, sid, ushort(lane));
  SteelMMA<BM, BN, BK, WM, WN, false, false, lda_nn, ldb_nn> mma_op(sid, lane);

  int k_iters = fc_k_aligned ? int(p.K >> 4) : int(p.K / BK);
  short tgp_bm = fc_mn_aligned ? BM : min(short(BM), short(p.M - row0));
  short tgp_bn = fc_mn_aligned ? BN : min(short(BN), short(p.N - col0));
  bool is_interior = fc_mn_aligned || (tgp_bm == BM && tgp_bn == BN);

  if (!fc_k_aligned) {
    short lbk = short(p.K) - short(k_iters * BK);
    size_t k_jump_a = size_t(k_iters) * BK, k_jump_b = size_t(k_iters) * BK * p.ldb;
    loader_a.src += k_jump_a; loader_b.src += k_jump_b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_safe(short2(lbk, tgp_bm)); loader_b.load_safe(short2(tgp_bn, lbk));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.src -= k_jump_a + BK; loader_b.src -= k_jump_b + BK * p.ldb;
    loader_a.next(); loader_b.next();
  }
  if (is_interior) {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe(); loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs); loader_a.next(); loader_b.next();
    }
  } else {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(short2(BK, tgp_bm)); loader_b.load_safe(short2(tgp_bn, BK));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs); loader_a.next(); loader_b.next();
    }
  }
  C += row0 * int(p.N) + col0;
  if (is_interior) { mma_op.store_result_bias(C, int(p.N), bias, bias_len, col0); }
  else { mma_op.store_result_safe(C, int(p.N),
      short2(min(short(BN), short(p.N - col0)), min(short(BM), short(p.M - row0)))); }
}

// STEEL gemm+bias+sigmoid body
template <short BM>
void steel_gemm_bias_sigmoid_body_(
    device const float* A, device const float* B, device float* C,
    constant gemm_params& p, device const float* bias, uint32_t bias_len,
    threadgroup float* As, threadgroup float* Bs,
    uint3 tgid, uint sid, uint lane) {
  constexpr short BN = 64, BK = 16, WM = 2, WN = 2, pad = 4;
  constexpr short tgp_size = WM * WN * 32;

  A += tgid.z * p.batch_stride_a; B += tgid.z * p.batch_stride_b; C += tgid.z * p.batch_stride_c;
  short swizzle = short(p.trans_a);
  short tiles_n = short((p.N + BN - 1) / BN), tiles_m = short((p.M + BM - 1) / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short row0 = tid_y * BM, col0 = tid_x * BN;
  A += row0 * int(p.lda); B += int(col0);
  constexpr short lda_nn = BK + pad, ldb_nn = BN + pad;
  SteelLoader<BM, BK, lda_nn, true, tgp_size> loader_a(A, int(p.lda), As, sid, ushort(lane));
  SteelLoader<BK, BN, ldb_nn, false, tgp_size> loader_b(B, int(p.ldb), Bs, sid, ushort(lane));
  SteelMMA<BM, BN, BK, WM, WN, false, false, lda_nn, ldb_nn> mma_op(sid, lane);

  int k_iters = fc_k_aligned ? int(p.K >> 4) : int(p.K / BK);
  short tgp_bm = fc_mn_aligned ? BM : min(short(BM), short(p.M - row0));
  short tgp_bn = fc_mn_aligned ? BN : min(short(BN), short(p.N - col0));
  bool is_interior = fc_mn_aligned || (tgp_bm == BM && tgp_bn == BN);

  if (!fc_k_aligned) {
    short lbk = short(p.K) - short(k_iters * BK);
    size_t k_jump_a = size_t(k_iters) * BK, k_jump_b = size_t(k_iters) * BK * p.ldb;
    loader_a.src += k_jump_a; loader_b.src += k_jump_b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_safe(short2(lbk, tgp_bm)); loader_b.load_safe(short2(tgp_bn, lbk));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.src -= k_jump_a + BK; loader_b.src -= k_jump_b + BK * p.ldb;
    loader_a.next(); loader_b.next();
  }
  if (is_interior) {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe(); loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs); loader_a.next(); loader_b.next();
    }
  } else {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(short2(BK, tgp_bm)); loader_b.load_safe(short2(tgp_bn, BK));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs); loader_a.next(); loader_b.next();
    }
  }
  C += row0 * int(p.N) + col0;
  if (is_interior) { mma_op.store_result_bias_sigmoid(C, int(p.N), bias, bias_len, col0); }
  else { mma_op.store_result_safe(C, int(p.N),
      short2(min(short(BN), short(p.N - col0)), min(short(BM), short(p.M - row0)))); }
}

// --- STEEL 64x64 kernel entry points ---

kernel void sgemm_steel_(device const float* A [[buffer(0)]],
                          device const float* B [[buffer(1)]],
                          device float* C       [[buffer(2)]],
                          constant gemm_params& p [[buffer(3)]],
                          threadgroup float* As [[threadgroup(0)]],
                          threadgroup float* Bs [[threadgroup(1)]],
                          uint3 tgid [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sid [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  steel_gemm_body_<64>(A, B, C, p, As, Bs, tgid, sid, lane);
}

kernel void sgemm_bias_steel_(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant gemm_params& p [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    constant uint32_t& bias_len [[buffer(5)]],
    threadgroup float* As [[threadgroup(0)]],
    threadgroup float* Bs [[threadgroup(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  steel_gemm_bias_body_<64>(A, B, C, p, bias, bias_len, As, Bs, tgid, sid, lane);
}

kernel void sgemm_bias_sigmoid_steel_(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant gemm_params& p [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    constant uint32_t& bias_len [[buffer(5)]],
    threadgroup float* As [[threadgroup(0)]],
    threadgroup float* Bs [[threadgroup(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  steel_gemm_bias_sigmoid_body_<64>(A, B, C, p, bias, bias_len, As, Bs, tgid, sid, lane);
}

// --- STEEL 32x64 kernel entry points (more threadgroups for small M) ---

#define STEEL_32x64_KERNEL(name, body_fn, ...) \
kernel void name( \
    device const float* A [[buffer(0)]], \
    device const float* B [[buffer(1)]], \
    device float* C       [[buffer(2)]], \
    constant gemm_params& p [[buffer(3)]], \
    __VA_ARGS__ \
    threadgroup float* As [[threadgroup(0)]], \
    threadgroup float* Bs [[threadgroup(1)]], \
    uint3 tgid [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint sid [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]])

STEEL_32x64_KERNEL(sgemm_steel_32x64_, steel_gemm_body_<32>, ) {
  steel_gemm_body_<32>(A, B, C, p, As, Bs, tgid, sid, lane);
}

STEEL_32x64_KERNEL(sgemm_bias_steel_32x64_, steel_gemm_bias_body_<32>,
    device const float* bias [[buffer(4)]],
    constant uint32_t& bias_len [[buffer(5)]],) {
  steel_gemm_bias_body_<32>(A, B, C, p, bias, bias_len, As, Bs, tgid, sid, lane);
}

STEEL_32x64_KERNEL(sgemm_bias_sigmoid_steel_32x64_, steel_gemm_bias_sigmoid_body_<32>,
    device const float* bias [[buffer(4)]],
    constant uint32_t& bias_len [[buffer(5)]],) {
  steel_gemm_bias_sigmoid_body_<32>(A, B, C, p, bias, bias_len, As, Bs, tgid, sid, lane);
}

#undef STEEL_32x64_KERNEL
#undef STEEL_CONST
#undef STEEL_PRAGMA_UNROLL

// ---------------------------------------------------------------------------

template <typename Ope, typename T>
void arithmetic_operation_(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_length,
  constant uint32_t& B_length,
  uint gid)
{
  auto A_arr = static_cast<device const T*>(A);
  auto B_arr = static_cast<device const T*>(B);
  auto OUT_arr = reinterpret_cast<device T*>(OUT);

  auto A_index = gid % A_length;
  auto B_index = gid % B_length;

  OUT_arr[gid] = Ope()(A_arr[A_index], B_arr[B_index]);
}

#define DEFINE_BINOP(Name, Op) \
  template <typename T> struct Name { \
    T operator()(T a, T b) { return a Op b; } \
    float4 operator()(float4 a, float4 b) { return a Op b; } \
  };
DEFINE_BINOP(add_, +)
DEFINE_BINOP(sub_, -)
DEFINE_BINOP(mul_, *)
DEFINE_BINOP(div_, /)
#undef DEFINE_BINOP

struct powf_ {
  float operator()(float a, float b) { return pow(a, b); }
  float4 operator()(float4 a, float4 b) { return pow(a, b); }
};
struct powi_ { int operator()(int a, int b) {
  return round(pow(static_cast<float>(a), static_cast<float>(b)));
} };

// Fast modulo using precomputed inverse: x % len ≈ x - ((x * inv) >> 32) * len
// inv must be precomputed on CPU as uint32_t((1ULL << 32) / len + 1)
inline uint fast_mod(uint x, uint len, uint inv) {
  uint q = uint((ulong(x) * ulong(inv)) >> 32);
  uint r = x - q * len;
  return select(r, r - len, r >= len);
}

// float4 vectorized path: gid is in units of float4 (4 elements per thread)
// B_inv is precomputed (1<<32)/B_length+1 for fast modulo (passed from CPU).
template <typename Ope>
void arithmetic_operation_f4_(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_length,
  constant uint32_t& B_length,
  constant uint32_t& OUT_length,
  constant uint32_t& B_inv,
  uint gid)
{
  auto A_arr = static_cast<device const float*>(A);
  auto B_arr = static_cast<device const float*>(B);
  auto OUT_arr = reinterpret_cast<device float*>(OUT);

  Ope op;
  uint base = gid * 4;
  if (base + 4 <= OUT_length && A_length == OUT_length && B_length == OUT_length) {
    auto a4 = *reinterpret_cast<device const float4*>(A_arr + base);
    auto b4 = *reinterpret_cast<device const float4*>(B_arr + base);
    *reinterpret_cast<device float4*>(OUT_arr + base) = op(a4, b4);
  } else if (base + 4 <= OUT_length && A_length == OUT_length && B_length == 1) {
    auto a4 = *reinterpret_cast<device const float4*>(A_arr + base);
    *reinterpret_cast<device float4*>(OUT_arr + base) = op(a4, float4(B_arr[0]));
  } else if (base + 4 <= OUT_length && A_length == OUT_length && B_length > 1) {
    auto a4 = *reinterpret_cast<device const float4*>(A_arr + base);
    uint b_base = fast_mod(base, B_length, B_inv);
    float4 b4;
    if (b_base + 4 <= B_length) {
      b4 = *reinterpret_cast<device const float4*>(B_arr + b_base);
    } else {
      b4 = float4(B_arr[b_base],
                   B_arr[fast_mod(base + 1, B_length, B_inv)],
                   B_arr[fast_mod(base + 2, B_length, B_inv)],
                   B_arr[fast_mod(base + 3, B_length, B_inv)]);
    }
    *reinterpret_cast<device float4*>(OUT_arr + base) = op(a4, b4);
  } else {
    for (uint i = 0; i < 4 && base + i < OUT_length; i++) {
      OUT_arr[base + i] = op(A_arr[(base + i) % A_length], B_arr[(base + i) % B_length]);
    }
  }
}

constant uint32_t Float = 0;

kernel void add(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length, constant uint32_t& B_inv,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<add_<float>>(A, B, OUT, A_length, B_length, OUT_length, B_inv, gid);
  else arithmetic_operation_<add_<int>, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void sub(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length, constant uint32_t& B_inv,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<sub_<float>>(A, B, OUT, A_length, B_length, OUT_length, B_inv, gid);
  else arithmetic_operation_<sub_<int>, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void mul(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length, constant uint32_t& B_inv,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<mul_<float>>(A, B, OUT, A_length, B_length, OUT_length, B_inv, gid);
  else arithmetic_operation_<mul_<int>, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void div(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length, constant uint32_t& B_inv,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<div_<float>>(A, B, OUT, A_length, B_length, OUT_length, B_inv, gid);
  else arithmetic_operation_<div_<int>, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void pow(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length, constant uint32_t& B_inv,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<powf_>(A, B, OUT, A_length, B_length, OUT_length, B_inv, gid);
  else arithmetic_operation_<powi_, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void sigmoid_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& length,
  uint gid [[thread_position_in_grid]])
{
  uint base = gid * 4;
  if (base + 4 <= length) {
    auto v = *reinterpret_cast<device const float4*>(IN + base);
    auto r = 1.0f / (1.0f + exp(-v));
    *reinterpret_cast<device float4*>(OUT + base) = r;
  } else {
    for (uint i = 0; i < 4 && base + i < length; i++) {
      OUT[base + i] = 1.0f / (1.0f + exp(-IN[base + i]));
    }
  }
}

kernel void relu_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& length,
  uint gid [[thread_position_in_grid]])
{
  uint base = gid * 4;
  if (base + 4 <= length) {
    auto v = *reinterpret_cast<device const float4*>(IN + base);
    *reinterpret_cast<device float4*>(OUT + base) = max(v, float4(0.0f));
  } else {
    for (uint i = 0; i < 4 && base + i < length; i++) {
      OUT[base + i] = max(IN[base + i], 0.0f);
    }
  }
}

// Threadgroup parallel sum reduction helper.
float tg_reduce_sum(threadgroup float* shared, uint tid, uint tg_size, float val) {
  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tg_size / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return shared[0];
}

// Simdgroup + threadgroup two-level reduction for sum.
// First reduces within each simdgroup (barrier-free), then across simdgroups.
float tg_simd_reduce_sum(threadgroup float* shared, uint tid, uint tg_size, float val) {
  // Level 1: simdgroup reduction (no barrier needed)
  val = simd_sum(val);

  // Level 2: inter-simdgroup reduction via shared memory
  uint simd_id = tid / 32;
  uint lane = tid % 32;
  uint num_simds = tg_size / 32;

  if (lane == 0) shared[simd_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final reduction by first simdgroup
  if (simd_id == 0) {
    val = (lane < num_simds) ? shared[lane] : 0.0f;
    val = simd_sum(val);
    if (lane == 0) shared[0] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return shared[0];
}

kernel void sum_f32_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& length,
  uint gid [[thread_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_id [[threadgroup_position_in_grid]],
  uint tg_size [[threads_per_threadgroup]],
  uint grid_size [[threads_per_grid]])
{
  threadgroup float shared[32];  // enough for 1024/32 simdgroups

  float val = 0.0f;
  uint vec_len = length / 4;
  device const float4* IN4 = reinterpret_cast<device const float4*>(IN);
  for (uint i = gid; i < vec_len; i += grid_size) {
    float4 v = IN4[i];
    val += v.x + v.y + v.z + v.w;
  }
  for (uint i = vec_len * 4 + gid; i < length; i += grid_size) {
    val += IN[i];
  }

  float result = tg_simd_reduce_sum(shared, tid, tg_size, val);
  if (tid == 0) {
    OUT[tg_id] = result;
  }
}

// Sum along axis=0 for 2D matrix (rows x cols).
// 2D tiled: local_id.x spans columns (coalesced), local_id.y spans row workers.
// tg_id.y selects row block; each TG reduces bm rows starting at tg_id.y * bm.
// Output written to OUT[tg_id.y * cols + col].
constant constexpr uint SA0_BX = 32;
constant constexpr uint SA0_BY = 32;

kernel void sum_axis0_f32_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& rows,
  constant uint32_t& cols,
  constant uint32_t& bm,
  uint2 tg_id [[threadgroup_position_in_grid]],
  uint2 local_id [[thread_position_in_threadgroup]])
{
  uint col = tg_id.x * SA0_BX + local_id.x;
  if (col >= cols) return;

  uint row_start = tg_id.y * bm;
  uint row_end = min(row_start + bm, rows);

  float val = 0.0f;
  for (uint r = row_start + local_id.y; r < row_end; r += SA0_BY) {
    val += IN[r * cols + col];
  }

  // Reduce across row workers via shared memory
  threadgroup float shared[SA0_BY][SA0_BX];
  shared[local_id.y][local_id.x] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = SA0_BY / 2; s > 0; s >>= 1) {
    if (local_id.y < s) {
      shared[local_id.y][local_id.x] += shared[local_id.y + s][local_id.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (local_id.y == 0) {
    OUT[tg_id.y * cols + col] = shared[0][local_id.x];
  }
}

// LayerNorm: one threadgroup per row.
// Two passes: (1) compute mean+variance in single pass, (2) normalize+scale+shift.
// Uses simdgroup reductions to minimize barriers.
kernel void layer_norm_(
  device const float* IN,
  device float* OUT,
  device const float* gamma,
  device const float* beta,
  constant uint32_t& cols,
  constant float& eps,
  uint row_id [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_size [[threads_per_threadgroup]])
{
  threadgroup float shared[64];  // enough for 32 simdgroups (1024/32)

  device const float* row = IN + row_id * cols;
  device float* out_row = OUT + row_id * cols;
  uint vec_len = cols / 4;
  device const float4* row4 = reinterpret_cast<device const float4*>(row);

  // Pass 1: compute sum and sum-of-squares in a single pass (vectorized)
  float sum_val = 0.0f;
  float sq_val = 0.0f;
  for (uint i = tid; i < vec_len; i += tg_size) {
    float4 v = row4[i];
    sum_val += v.x + v.y + v.z + v.w;
    sq_val += dot(v, v);
  }
  for (uint i = vec_len * 4 + tid; i < cols; i += tg_size) {
    float v = row[i];
    sum_val += v;
    sq_val += v * v;
  }

  float total_sum = tg_simd_reduce_sum(shared, tid, tg_size, sum_val);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sq = tg_simd_reduce_sum(shared, tid, tg_size, sq_val);

  float mean = total_sum / float(cols);
  // var = E[x^2] - E[x]^2
  float inv_std = rsqrt(total_sq / float(cols) - mean * mean + eps);

  // Pass 2: normalize, scale, shift
  for (uint i = tid; i < vec_len; i += tg_size) {
    float4 v = row4[i];
    float4 g = reinterpret_cast<device const float4*>(gamma)[i];
    float4 b = reinterpret_cast<device const float4*>(beta)[i];
    reinterpret_cast<device float4*>(out_row)[i] =
        (v - float4(mean)) * float4(inv_std) * g + b;
  }
  for (uint i = vec_len * 4 + tid; i < cols; i += tg_size) {
    out_row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
  }
}

// Simdgroup + threadgroup two-level reduction for max.
float tg_simd_reduce_max(threadgroup float* shared, uint tid, uint tg_size, float val) {
  val = simd_max(val);
  uint simd_id = tid / 32;
  uint lane = tid % 32;
  uint num_simds = tg_size / 32;
  if (lane == 0) shared[simd_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    val = (lane < num_simds) ? shared[lane] : -HUGE_VALF;
    val = simd_max(val);
    if (lane == 0) shared[0] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return shared[0];
}

// Simdgroup + threadgroup two-level reduction for min.
float tg_simd_reduce_min(threadgroup float* shared, uint tid, uint tg_size, float val) {
  val = simd_min(val);
  uint simd_id = tid / 32;
  uint lane = tid % 32;
  uint num_simds = tg_size / 32;
  if (lane == 0) shared[simd_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    val = (lane < num_simds) ? shared[lane] : HUGE_VALF;
    val = simd_min(val);
    if (lane == 0) shared[0] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return shared[0];
}

kernel void min_f32_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& length,
  uint gid [[thread_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_id [[threadgroup_position_in_grid]],
  uint tg_size [[threads_per_threadgroup]],
  uint grid_size [[threads_per_grid]])
{
  threadgroup float shared[32];
  float4 acc = float4(HUGE_VALF);
  uint vec_len = length / 4;
  device const float4* IN4 = reinterpret_cast<device const float4*>(IN);
  for (uint i = gid; i < vec_len; i += grid_size) {
    acc = min(acc, IN4[i]);
  }
  float val = min(min(acc.x, acc.y), min(acc.z, acc.w));
  for (uint i = vec_len * 4 + gid; i < length; i += grid_size) {
    val = min(val, IN[i]);
  }
  float result = tg_simd_reduce_min(shared, tid, tg_size, val);
  if (tid == 0) OUT[tg_id] = result;
}

kernel void max_f32_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& length,
  uint gid [[thread_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_id [[threadgroup_position_in_grid]],
  uint tg_size [[threads_per_threadgroup]],
  uint grid_size [[threads_per_grid]])
{
  threadgroup float shared[32];
  float4 acc = float4(-HUGE_VALF);
  uint vec_len = length / 4;
  device const float4* IN4 = reinterpret_cast<device const float4*>(IN);
  for (uint i = gid; i < vec_len; i += grid_size) {
    acc = max(acc, IN4[i]);
  }
  float val = max(max(acc.x, acc.y), max(acc.z, acc.w));
  for (uint i = vec_len * 4 + gid; i < length; i += grid_size) {
    val = max(val, IN[i]);
  }
  float result = tg_simd_reduce_max(shared, tid, tg_size, val);
  if (tid == 0) OUT[tg_id] = result;
}

// Argmax: one threadgroup per row. Returns index of max value per row.
kernel void argmax_f32_(
  device const float* IN   [[buffer(0)]],
  device int* OUT          [[buffer(1)]],
  constant uint32_t& cols  [[buffer(2)]],
  uint row_id [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_size [[threads_per_threadgroup]])
{
  threadgroup float sval[32];
  threadgroup uint sidx[32];

  device const float* row = IN + row_id * cols;
  uint vec_len = cols / 4;
  device const float4* row4 = reinterpret_cast<device const float4*>(row);

  float max_val = -HUGE_VALF;
  uint max_idx = 0;

  for (uint i = tid; i < vec_len; i += tg_size) {
    float4 v = row4[i];
    uint base = i * 4;
    if (v.x > max_val) { max_val = v.x; max_idx = base; }
    if (v.y > max_val) { max_val = v.y; max_idx = base + 1; }
    if (v.z > max_val) { max_val = v.z; max_idx = base + 2; }
    if (v.w > max_val) { max_val = v.w; max_idx = base + 3; }
  }
  for (uint i = vec_len * 4 + tid; i < cols; i += tg_size) {
    float v = row[i];
    if (v > max_val) { max_val = v; max_idx = i; }
  }

  // Simd-level argmax reduction
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    float other_val = simd_shuffle_down(max_val, offset);
    uint other_idx = simd_shuffle_down(max_idx, offset);
    if (other_val > max_val) { max_val = other_val; max_idx = other_idx; }
  }

  // Inter-simdgroup reduction via shared memory
  uint simd_id = tid / 32;
  uint lane = tid % 32;
  uint num_simds = tg_size / 32;

  if (lane == 0) { sval[simd_id] = max_val; sidx[simd_id] = max_idx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    max_val = (lane < num_simds) ? sval[lane] : -HUGE_VALF;
    max_idx = (lane < num_simds) ? sidx[lane] : 0;
    for (ushort offset = 16; offset > 0; offset >>= 1) {
      float other_val = simd_shuffle_down(max_val, offset);
      uint other_idx = simd_shuffle_down(max_idx, offset);
      if (other_val > max_val) { max_val = other_val; max_idx = other_idx; }
    }
    if (lane == 0) OUT[row_id] = int(max_idx);
  }
}

// Online softmax: one threadgroup per row.
// Single-row softmax with register caching and float4 vectorization.
// 1 device memory read + 1 write. Each thread caches 1 float4 (4 elements).
// TG size = ceil(cols/4), rounded to simdgroup width. Max cols = 4096.
kernel void softmax_f32_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& cols,
  uint row_id [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_size [[threads_per_threadgroup]])
{
  threadgroup float shared[32];

  device const float* row = IN + row_id * cols;
  device float* out_row = OUT + row_id * cols;
  uint vec_len = cols / 4;

  // Phase 1: Load float4 into register (single device memory read)
  float4 cached;
  if (tid < vec_len) {
    cached = reinterpret_cast<device const float4*>(row)[tid];
  } else {
    cached = float4(-HUGE_VALF);
  }
  // Handle tail elements (cols not divisible by 4)
  uint base = tid * 4;
  if (base < cols && base + 4 > cols) {
    for (uint j = 0; j < 4; j++)
      cached[j] = (base + j < cols) ? row[base + j] : -HUGE_VALF;
  }

  // Phase 2: Per-thread max (from registers)
  float m = max(max(cached.x, cached.y), max(cached.z, cached.w));

  // Simd + threadgroup max reduction
  m = simd_max(m);
  uint simd_id = tid / 32;
  uint lane = tid % 32;
  uint num_simds = tg_size / 32;
  if (lane == 0) shared[simd_id] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    m = (lane < num_simds) ? shared[lane] : -HUGE_VALF;
    m = simd_max(m);
    if (lane == 0) shared[0] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = shared[0];

  // Phase 3: exp + sum (from registers, no re-read)
  float4 e;
  e.x = (base     < cols) ? fast::exp(cached.x - row_max) : 0.0f;
  e.y = (base + 1 < cols) ? fast::exp(cached.y - row_max) : 0.0f;
  e.z = (base + 2 < cols) ? fast::exp(cached.z - row_max) : 0.0f;
  e.w = (base + 3 < cols) ? fast::exp(cached.w - row_max) : 0.0f;
  float s = e.x + e.y + e.z + e.w;

  // Simd + threadgroup sum reduction
  s = simd_sum(s);
  if (lane == 0) shared[simd_id] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    s = (lane < num_simds) ? shared[lane] : 0.0f;
    s = simd_sum(s);
    if (lane == 0) shared[0] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv_sum = 1.0f / shared[0];

  // Phase 4: Normalize and write (from registers, single write)
  e *= inv_sum;
  if (tid < vec_len && base + 4 <= cols) {
    reinterpret_cast<device float4*>(out_row)[tid] = e;
  } else if (base < cols) {
    for (uint j = 0; j < 4 && base + j < cols; j++)
      out_row[base + j] = e[j];
  }
}

// Online softmax for cols > 4096: 2 reads of IN + 1 write of OUT.
// Pass 1 computes (max, sum) in a single IN read using running rescaling
//   s_new = s_old * exp(m_old - m_new) + exp(x - m_new)
// then reduces the (max, sum) tuple across the threadgroup.
// Pass 2 re-reads IN and writes exp(x-max)*inv_sum directly.
kernel void softmax_looped_f32_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& cols,
  uint row_id [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_size [[threads_per_threadgroup]])
{
  threadgroup float shared_m[32];
  threadgroup float shared_s[32];

  device const float* row = IN + row_id * cols;
  device float* out_row = OUT + row_id * cols;
  uint vec_len = cols / 4;
  device const float4* row4 = reinterpret_cast<device const float4*>(row);

  // Pass 1: single-read online (max, sum)
  float m = -HUGE_VALF;
  float s = 0.0f;
  for (uint i = tid; i < vec_len; i += tg_size) {
    float4 v = row4[i];
    float vmax = max(max(v.x, v.y), max(v.z, v.w));
    float new_m = max(m, vmax);
    s = s * fast::exp(m - new_m)
      + fast::exp(v.x - new_m) + fast::exp(v.y - new_m)
      + fast::exp(v.z - new_m) + fast::exp(v.w - new_m);
    m = new_m;
  }

  uint simd_id = tid / 32;
  uint lane = tid % 32;
  uint num_simds = tg_size / 32;

  // Simdgroup (max, sum) reduction — butterfly so the whole simd ends
  // up with the combined value (needed for the cross-simd stage below).
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    float m2 = simd_shuffle_xor(m, offset);
    float s2 = simd_shuffle_xor(s, offset);
    float new_m = max(m, m2);
    s = s * fast::exp(m - new_m) + s2 * fast::exp(m2 - new_m);
    m = new_m;
  }
  if (lane == 0) { shared_m[simd_id] = m; shared_s[simd_id] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    m = (lane < num_simds) ? shared_m[lane] : -HUGE_VALF;
    s = (lane < num_simds) ? shared_s[lane] : 0.0f;
    for (ushort offset = 16; offset > 0; offset >>= 1) {
      float m2 = simd_shuffle_xor(m, offset);
      float s2 = simd_shuffle_xor(s, offset);
      float new_m = max(m, m2);
      s = s * fast::exp(m - new_m) + s2 * fast::exp(m2 - new_m);
      m = new_m;
    }
    if (lane == 0) { shared_m[0] = m; shared_s[0] = s; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = shared_m[0];
  float inv_sum = 1.0f / shared_s[0];

  // Pass 2: normalized write
  device float4* out4 = reinterpret_cast<device float4*>(out_row);
  for (uint i = tid; i < vec_len; i += tg_size) {
    float4 v = row4[i];
    out4[i] = float4(fast::exp(v.x - row_max) * inv_sum,
                     fast::exp(v.y - row_max) * inv_sum,
                     fast::exp(v.z - row_max) * inv_sum,
                     fast::exp(v.w - row_max) * inv_sum);
  }
}

kernel void sigmoid_backward_f32_(
  device const float* dout  [[buffer(0)]],
  device const float* x     [[buffer(1)]],
  device float*       out   [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  float y = 1.0f / (1.0f + exp(-x[gid]));
  out[gid] = dout[gid] * y * (1.0f - y);
}

kernel void bias_sigmoid_f32_(
  device float*       data  [[buffer(0)]],
  device const float* bias  [[buffer(1)]],
  constant uint32_t&  cols  [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  float v = data[gid] + bias[gid % cols];
  data[gid] = 1.0f / (1.0f + exp(-v));
}

kernel void affine_f32_(
  device const float* in   [[buffer(0)]],
  device float*       out  [[buffer(1)]],
  constant float&     scale [[buffer(2)]],
  constant float&     offset [[buffer(3)]],
  uint gid [[thread_position_in_grid]])
{
  out[gid] = fma(in[gid], scale, offset);
}

// Tiled transpose: 32x32 tiles staged in threadgroup memory.
// Coalesced reads from input AND coalesced writes to output (after swap).
// +1 padding avoids bank conflicts.
kernel void transpose_f32_(
  device const float* input  [[buffer(0)]],
  device float*       output [[buffer(1)]],
  constant uint32_t&  rows   [[buffer(2)]],
  constant uint32_t&  cols   [[buffer(3)]],
  uint2 tgid [[threadgroup_position_in_grid]],
  uint2 lid  [[thread_position_in_threadgroup]])
{
  constexpr uint TILE = 32;
  threadgroup float tile[TILE][TILE + 1];

  // Coalesced read from input[r][c] where r = src row, c = src col
  uint src_c = tgid.x * TILE + lid.x;
  uint src_r = tgid.y * TILE + lid.y;
  if (src_r < rows && src_c < cols) {
    tile[lid.y][lid.x] = input[src_r * cols + src_c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Coalesced write to output[c][r] — swap src indices, write to dst
  uint dst_c = tgid.y * TILE + lid.x;
  uint dst_r = tgid.x * TILE + lid.y;
  if (dst_r < cols && dst_c < rows) {
    output[dst_r * rows + dst_c] = tile[lid.x][lid.y];
  }
}

// ---------------------------------------------------------------------------
// Implicit GEMM conv2d — STEEL-like tiled GEMM without im2col buffer.
// A tile is loaded on-the-fly from input using conv2d indexing.
// B tile (weight, pre-transposed to {K_dim, C_out}) uses standard loading.
// Requires C_in >= 16 for vectorized float4 loads within A tiles.
// ---------------------------------------------------------------------------

struct conv2d_gemm_params {
  uint M;       // N * H_out * W_out
  uint C_out;   // output channels (= N dimension of GEMM)
  uint K_dim;   // K * K * C_in
  uint N_batch, H, W, C_in;
  uint K, pad;
  uint H_out, W_out;
  uint swizzle_log;
};

// Fast variant for the common ResNet-like case where C_in is divisible by BK.
// MLX-style optimization: nest the K loop as (kh, kw) outer / ci_block inner,
// hoisting all spatial bounds checks out of the hot inner loop. With C_in=128
// and BK=16 this collapses 72 per-iteration bounds checks down to 9.
template <short BK>
void conv2d_gemm_fast_body_(
  device const float* input,
  device const float* weight,
  device float*       output,
  constant conv2d_gemm_params& p,
  threadgroup float* As,
  threadgroup float* Bs,
  uint3 tgid, uint tid, uint sid, uint lane)
{
  constexpr short BM = 64, BN = 64;
  constexpr short WM = 2, WN = 2;
  constexpr short tgp_size = WM * WN * 32;  // 128
  constexpr short pad_a = 4, pad_b = 4;
  constexpr short lda_nn = BK + pad_a;
  constexpr short ldb_nn = BN + pad_b;
  constexpr short a_vec = 8;
  constexpr short a_tcols = BK / a_vec;          // 2
  constexpr short a_trows = tgp_size / a_tcols;  // 64

  // Tile mapping with swizzle
  short swizzle = short(p.swizzle_log);
  short tiles_n = short((p.C_out + BN - 1) / BN);
  short tiles_m = short((p.M + BM - 1) / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  uint row0 = tid_y * BM;
  uint col0 = tid_x * BN;

  // B loader (unchanged): walks pre-transposed weight (K_dim, C_out) contiguously
  SteelLoader<BK, BN, ldb_nn, false, tgp_size> loader_b(
      weight + col0, int(p.C_out), Bs, sid, ushort(lane));

  // MMA accumulator
  SteelMMA<BM, BN, BK, WM, WN, false, false, lda_nn, ldb_nn> mma_op(sid, lane);

  // A loader thread mapping: each thread loads 1 row × 8 floats per K iter
  short a_bi = short(tid) / a_tcols;
  short a_bj = a_vec * (short(tid) % a_tcols);

  uint my_m = row0 + a_bi;
  bool my_m_valid = (my_m < p.M);

  uint spatial_size = p.H_out * p.W_out;
  uint WC = p.W * p.C_in;
  uint HWC = p.H * WC;

  // Hoisted: spatial coords + input base pointer, computed once per thread
  uint my_n = 0, my_oh = 0, my_ow = 0;
  device const float* my_input_base = input;
  if (my_m_valid) {
    my_n = my_m / spatial_size;
    uint s = my_m % spatial_size;
    my_oh = s / p.W_out;
    my_ow = s % p.W_out;
    my_input_base = input + my_n * HWC;
  }

  short K_spatial = short(p.K);
  short ci_blocks = short(p.C_in / BK);  // caller guarantees C_in % BK == 0

  // a_dst is invariant across all three loops — hoist out.
  // BK == a_vec * 2 (16 == 8 * 2): each thread fills its row with 2 float4s.
  static_assert(a_vec == 8, "loader assumes 8-float per-thread row");
  threadgroup float* a_dst = &As[a_bi * lda_nn + a_bj];

  // Outer loop: (kh, kw). Inner loop: ci_block. Together they walk K_dim in
  // the same (kh, kw, ci) order as the pre-transposed weight layout.
  for (short kh = 0; kh < K_spatial; kh++) {
    int ih = int(my_oh) + int(kh) - int(p.pad);
    int ih_clamped = clamp(ih, 0, int(p.H) - 1);
    bool h_in = my_m_valid && (ih >= 0) && (uint(ih) < p.H);
    int ih_off = ih_clamped * int(WC);

    for (short kw = 0; kw < K_spatial; kw++) {
      int iw = int(my_ow) + int(kw) - int(p.pad);
      int iw_clamped = clamp(iw, 0, int(p.W) - 1);
      bool valid = h_in && (iw >= 0) && (uint(iw) < p.W);
      // row_src is clamped to a valid spatial position so the float4 loads
      // below never touch out-of-bounds memory; the multiply by `mask` zeroes
      // out the result at padding edges (~165μs faster than an if/else branch
      // for ResNet deep, profiled 2026-04-10 — the eliminated divergent
      // branch dominates the saved load).
      device const float* row_src =
          my_input_base + ih_off + iw_clamped * int(p.C_in) + a_bj;
      float mask = valid ? 1.0f : 0.0f;

      for (short cb = 0; cb < ci_blocks; cb++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float4 v0 = *reinterpret_cast<device const float4*>(row_src) * mask;
        float4 v1 = *reinterpret_cast<device const float4*>(row_src + 4) * mask;
        *reinterpret_cast<threadgroup float4*>(a_dst) = v0;
        *reinterpret_cast<threadgroup float4*>(a_dst + 4) = v1;
        row_src += BK;  // advance by 2 float4s = a_vec floats

        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
        loader_b.next();
      }
    }
  }

  // Store output
  output += row0 * p.C_out + col0;
  short out_bm = short(min(uint(BM), p.M - row0));
  short out_bn = short(min(uint(BN), p.C_out - col0));
  if (out_bm == BM && out_bn == BN) {
    mma_op.store_result(output, int(p.C_out));
  } else {
    mma_op.store_result_safe(output, int(p.C_out), short2(out_bn, out_bm));
  }
}

template <short BK>
void conv2d_gemm_body_(
  device const float* input,
  device const float* weight,
  device float*       output,
  constant conv2d_gemm_params& p,
  threadgroup float* As,
  threadgroup float* Bs,
  uint3 tgid, uint tid, uint sid, uint lane)
{
  constexpr short BM = 64, BN = 64;
  constexpr short WM = 2, WN = 2;
  constexpr short tgp_size = WM * WN * 32;  // 128
  constexpr short pad_a = 4, pad_b = 4;
  constexpr short lda_nn = BK + pad_a;  // 20
  constexpr short ldb_nn = BN + pad_b;  // 68

  // Tile mapping with swizzle
  short swizzle = short(p.swizzle_log);
  short tiles_n = short((p.C_out + BN - 1) / BN);
  short tiles_m = short((p.M + BM - 1) / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  uint row0 = tid_y * BM;  // starting row in virtual im2col
  uint col0 = tid_x * BN;  // starting output channel

  // B loader (weight): standard contiguous loading
  SteelLoader<BK, BN, ldb_nn, false, tgp_size> loader_b(
      weight + col0, int(p.C_out), Bs, sid, ushort(lane));

  // MMA accumulator
  SteelMMA<BM, BN, BK, WM, WN, false, false, lda_nn, ldb_nn> mma_op(sid, lane);

  // A tile loading parameters (implicit im2col)
  // Thread assignment matching SteelLoader<BM, BK, lda_nn, true, 128>:
  // vec_size = (BK * BM) / tgp_size = 1024/128 = 8
  // TCOLS = BK / 8 = 2,  TROWS = 128 / 2 = 64
  constexpr short a_vec = 8, a_tcols = BK / a_vec, a_trows = tgp_size / a_tcols;
  short a_bi = short(tid) / a_tcols;  // row within tile (0..63)
  short a_bj = a_vec * (short(tid) % a_tcols);  // col offset (0 or 8)

  int K_dim = int(p.K_dim);
  int k_iters = K_dim / BK;
  uint spatial_size = p.H_out * p.W_out;
  uint HWC = p.H * p.W * p.C_in;
  uint WC = p.W * p.C_in;
  uint KC = p.K * p.C_in;

  // Precompute spatial coordinates for this thread's row (hoisted out of K loop)
  // Each thread loads 1 row of the A tile (a_bi), so precompute (n, oh, ow, input_base) once.
  uint my_m = row0 + a_bi;
  uint my_n = 0, my_oh = 0, my_ow = 0;
  device const float* my_input_base = input;
  bool my_m_valid = (my_m < p.M);
  if (my_m_valid) {
    my_n = my_m / spatial_size;
    uint s = my_m % spatial_size;
    my_oh = s / p.W_out;
    my_ow = s % p.W_out;
    my_input_base = input + my_n * HWC;
  }

  // Incremental K-position tracking (eliminates integer division in main loop).
  // Tracks (kh, kw, ci) for k=0 and advances by BK each iteration.
  // For this thread's column offset a_bj, compute initial position.
  uint init_k = a_bj;  // will be adjusted for remainder-first pattern
  uint t_ci = init_k % p.C_in;
  uint t_kw = (init_k / p.C_in) % p.K;
  uint t_kh = init_k / KC;

  // Helper: load 8-element A tile row using tracked (kh, kw, ci)
  auto load_a_vec = [&](short local_r) {
    threadgroup float* dst = &As[local_r * lda_nn + a_bj];
    if (!my_m_valid) {
      *reinterpret_cast<threadgroup float4*>(dst) = float4(0);
      *reinterpret_cast<threadgroup float4*>(dst + 4) = float4(0);
      return;
    }
    int ih = int(my_oh) + int(t_kh) - int(p.pad);
    int iw = int(my_ow) + int(t_kw) - int(p.pad);

    if (ih >= 0 && uint(ih) < p.H && iw >= 0 && uint(iw) < p.W) {
      device const float* src = my_input_base + ih * WC + iw * p.C_in + t_ci;
      if (t_ci + a_vec <= p.C_in) {
        // Contiguous ci values → vectorized load
        *reinterpret_cast<threadgroup float4*>(dst) =
            *reinterpret_cast<device const float4*>(src);
        *reinterpret_cast<threadgroup float4*>(dst + 4) =
            *reinterpret_cast<device const float4*>(src + 4);
      } else {
        // ci wraps around kw/kh boundary — scalar load
        for (short j = 0; j < a_vec; j++) {
          uint cj = t_ci + j;
          if (cj < p.C_in) {
            dst[j] = src[j];
          } else {
            // Crossed to next (kw, kh) — recompute
            uint kk = (init_k + j);  // relative to current k_base
            uint ckh = kk / KC, ckw = (kk / p.C_in) % p.K, cci = kk % p.C_in;
            int cih = int(my_oh) + int(ckh) - int(p.pad);
            int ciw = int(my_ow) + int(ckw) - int(p.pad);
            dst[j] = (cih >= 0 && uint(cih) < p.H && ciw >= 0 && uint(ciw) < p.W)
                ? my_input_base[cih * WC + ciw * p.C_in + cci] : 0.0f;
          }
        }
      }
    } else {
      *reinterpret_cast<threadgroup float4*>(dst) = float4(0);
      *reinterpret_cast<threadgroup float4*>(dst + 4) = float4(0);
    }
  };

  // Advance tracked position by BK
  auto advance_k = [&]() {
    t_ci += BK;
    if (t_ci >= p.C_in) {
      t_ci -= p.C_in;
      t_kw++;
      if (t_kw >= p.K) { t_kw = 0; t_kh++; }
    }
  };

  // K-remainder (handle tail first, MLX pattern)
  short k_rem = short(K_dim - k_iters * BK);
  if (k_rem > 0) {
    // Jump tracked position to remainder start
    uint rem_k = k_iters * BK + a_bj;
    t_ci = rem_k % p.C_in;
    t_kw = (rem_k / p.C_in) % p.K;
    t_kh = rem_k / KC;
    init_k = rem_k;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load A tile (remainder) — scalar path
    for (short i = 0; i < BM; i += a_trows) {
      short local_r = a_bi + i;
      threadgroup float* dst = &As[local_r * lda_nn];
      for (short j = 0; j < BK; j++) {
        short local_c = j;
        float val = 0.0f;
        if (my_m_valid && local_c < k_rem) {
          uint k = k_iters * BK + local_c;
          uint skh = k / KC, skw = (k / p.C_in) % p.K, sci = k % p.C_in;
          int sih = int(my_oh) + int(skh) - int(p.pad);
          int siw = int(my_ow) + int(skw) - int(p.pad);
          if (sih >= 0 && uint(sih) < p.H && siw >= 0 && uint(siw) < p.W)
            val = my_input_base[sih * WC + siw * p.C_in + sci];
        }
        if (j >= a_bj && j < a_bj + a_vec)
          dst[j] = val;
      }
    }

    // Load B tile (remainder)
    for (short i = tid; i < BK * ldb_nn; i += tgp_size) Bs[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      short b_bi = short(tid) / (BN / a_vec);
      short b_bj = a_vec * (short(tid) % (BN / a_vec));
      for (short i = 0; i < BK; i += (tgp_size / (BN / a_vec))) {
        short r = b_bi + i;
        if (r < k_rem) {
          device const float* src = weight + (k_iters * BK + r) * p.C_out + col0 + b_bj;
          for (short j = 0; j < a_vec && col0 + b_bj + j < p.C_out; j++)
            Bs[r * ldb_nn + b_bj + j] = src[j];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);

    // Reset tracked position to start for main loop
    t_ci = a_bj % p.C_in;
    t_kw = (a_bj / p.C_in) % p.K;
    t_kh = a_bj / KC;
    init_k = a_bj;
  }

  // Main loop: full BK tiles with incremental K tracking (no division)
  for (int k_tile = 0; k_tile < k_iters; k_tile++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load A tile: implicit im2col
    for (short i = 0; i < BM; i += a_trows)
      load_a_vec(a_bi + i);

    // Load B tile: standard vectorized
    loader_b.load_unsafe();

    advance_k();

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_b.next();
  }

  // Store output
  output += row0 * p.C_out + col0;
  short out_bm = short(min(uint(BM), p.M - row0));
  short out_bn = short(min(uint(BN), p.C_out - col0));
  if (out_bm == BM && out_bn == BN) {
    mma_op.store_result(output, int(p.C_out));
  } else {
    mma_op.store_result_safe(output, int(p.C_out), short2(out_bn, out_bm));
  }
}

// Small-channels variant for the ImageNet-first style case where C_in is tiny
// (e.g. 3 RGB channels). Reads weight directly from natural (C_out, K, K, C_in)
// layout — no separate transpose dispatch — and packs ci into a vec_size=4
// vector slot. Each K iteration covers TCOLS=4 (kh, kw) positions in parallel,
// so for 3×3 kernels the K loop is just ceil(9/4)=3 iterations of fully
// vectorized loads instead of the slow scalar K-remainder path.
//
// Hardcoded for n_channels=3 (most common). Extend by templating if needed.
// Templated by BM/BN so we can pick a 32×32 variant for low-C_out cases
// (e.g. ImageNet first: C_out=32 wastes half a 64-wide tile).
template <short BK, short BM, short BN>
void conv2d_gemm_smallch3_body_(
  device const float* input,
  device const float* weight,
  device float*       output,
  constant conv2d_gemm_params& p,
  threadgroup float* As,
  threadgroup float* Bs,
  uint3 tgid, uint tid, uint sid, uint lane)
{
  constexpr short WM = 2, WN = 2;
  constexpr short tgp_size = WM * WN * 32;  // 128
  constexpr short pad_a = 4, pad_b = 4;
  constexpr short lda_nn = BK + pad_a;  // 20
  constexpr short ldb_nn = BK + pad_b;  // 20 (B is stored (BN, BK) — transposed wrt main kernel!)
  constexpr short n_channels = 3;
  constexpr short vec_size = 4;          // 3 ci + 1 zero pad
  constexpr short TCOLS = BK / vec_size; // 4
  constexpr short TROWS_A = tgp_size / TCOLS;  // 32
  constexpr short n_rows_a = BM / TROWS_A;     // 2
  constexpr short TROWS_B = tgp_size / TCOLS;  // 32
  constexpr short n_rows_b = BN / TROWS_B;     // 2

  // Tile mapping with swizzle
  short swizzle = short(p.swizzle_log);
  short tiles_n = short((p.C_out + BN - 1) / BN);
  short tiles_m = short((p.M + BM - 1) / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  uint row0 = tid_y * BM;
  uint col0 = tid_x * BN;

  // Note: B is stored as (BN, BK) — transposed compared to the main kernel.
  // SteelMMA reads it with tB=true so the K dimension becomes the inner stride.
  SteelMMA<BM, BN, BK, WM, WN, /*tA=*/false, /*tB=*/true, lda_nn, ldb_nn> mma_op(sid, lane);

  // Thread mapping for the A loader
  short a_bi = short(tid) / TCOLS;            // 0..31, the spatial-row group
  short a_bj = vec_size * (short(tid) % TCOLS); // 0,4,8,12 within the BK columns
  short weight_hw_init = short(tid) % TCOLS;  // 0..3

  // Thread mapping for the B loader (mirrors A — same TCOLS, TROWS layout)
  short b_bi = short(tid) / TCOLS;            // 0..31, the C_out-row group
  short b_bj = vec_size * (short(tid) % TCOLS); // 0,4,8,12

  uint spatial_size = p.H_out * p.W_out;
  uint WC = p.W * p.C_in;
  uint HWC = p.H * WC;
  uint K_dim_natural = p.K * p.K * p.C_in;  // weight stride per c_out

  // Precompute spatial coordinates for each of the n_rows_a spatial rows this
  // thread loads. Held in registers, used inside the K loop.
  int read_n[n_rows_a];
  int read_ih[n_rows_a];
  int read_iw[n_rows_a];
  device const float* a_src_base[n_rows_a];

  for (short i = 0; i < n_rows_a; i++) {
    uint my_m = row0 + a_bi + i * TROWS_A;
    if (my_m < p.M) {
      uint n = my_m / spatial_size;
      uint s = my_m % spatial_size;
      uint oh = s / p.W_out;
      uint ow = s % p.W_out;
      read_n[i] = int(n);
      read_ih[i] = int(oh) - int(p.pad);
      read_iw[i] = int(ow) - int(p.pad);
      a_src_base[i] = input + n * HWC;
    } else {
      read_n[i] = -1;  // marks invalid row
      read_ih[i] = 0;
      read_iw[i] = 0;
      a_src_base[i] = input;
    }
  }

  // Precompute weight base pointers per c_out row
  device const float* b_src_base[n_rows_b];
  bool b_row_valid[n_rows_b];
  for (short i = 0; i < n_rows_b; i++) {
    uint c_out_idx = col0 + b_bi + i * TROWS_B;
    b_row_valid[i] = (c_out_idx < p.C_out);
    b_src_base[i] = weight + c_out_idx * K_dim_natural;
  }

  short k_positions = short(p.K * p.K);  // total (kh, kw) positions
  short k_iters = (k_positions + TCOLS - 1) / TCOLS;  // ceil

  for (short k_iter = 0; k_iter < k_iters; k_iter++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread handles weight_hw = weight_hw_init + k_iter * TCOLS
    short weight_hw = weight_hw_init + k_iter * TCOLS;
    bool wh_in_range = (weight_hw < k_positions);
    short wh = wh_in_range ? (weight_hw / short(p.K)) : 0;
    short ww = wh_in_range ? (weight_hw % short(p.K)) : 0;

    // ---- A load: fill As[a_bi+i*TROWS_A][a_bj..a_bj+vec_size-1] for each row
    for (short i = 0; i < n_rows_a; i++) {
      threadgroup float* dst = &As[(a_bi + i * TROWS_A) * lda_nn + a_bj];
      bool in_bounds = wh_in_range && (read_n[i] >= 0);
      int ih = read_ih[i] + int(wh);
      int iw = read_iw[i] + int(ww);
      in_bounds = in_bounds &&
                  (ih >= 0) && (uint(ih) < p.H) &&
                  (iw >= 0) && (uint(iw) < p.W);
      if (in_bounds) {
        device const float* src =
            a_src_base[i] + uint(ih) * WC + uint(iw) * p.C_in;
        // Unrolled: 3 channels then zero pad
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 0.0f;
      } else {
        *reinterpret_cast<threadgroup float4*>(dst) = float4(0);
      }
    }

    // ---- B load: weight[c_out, wh, ww, ci] for each row, padded ci to 4
    for (short i = 0; i < n_rows_b; i++) {
      threadgroup float* dst = &Bs[(b_bi + i * TROWS_B) * ldb_nn + b_bj];
      if (wh_in_range && b_row_valid[i]) {
        device const float* src =
            b_src_base[i] + uint(wh) * p.K * p.C_in + uint(ww) * p.C_in;
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 0.0f;
      } else {
        *reinterpret_cast<threadgroup float4*>(dst) = float4(0);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
  }

  // Store output
  output += row0 * p.C_out + col0;
  short out_bm = short(min(uint(BM), p.M - row0));
  short out_bn = short(min(uint(BN), p.C_out - col0));
  if (out_bm == BM && out_bn == BN) {
    mma_op.store_result(output, int(p.C_out));
  } else {
    mma_op.store_result_safe(output, int(p.C_out), short2(out_bn, out_bm));
  }
}

// Kernel entry points for BK=16 (default) and BK=32 (large C_in)
#define CONV2D_GEMM_KERNEL(name, bk) \
kernel void name( \
  device const float* input [[buffer(0)]], device const float* weight [[buffer(1)]], \
  device float* output [[buffer(2)]], constant conv2d_gemm_params& p [[buffer(3)]], \
  threadgroup float* As [[threadgroup(0)]], threadgroup float* Bs [[threadgroup(1)]], \
  uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
  uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ conv2d_gemm_body_<bk>(input, weight, output, p, As, Bs, tgid, tid, sid, lane); }

CONV2D_GEMM_KERNEL(conv2d_gemm_f32_, 16)
#undef CONV2D_GEMM_KERNEL

#define CONV2D_GEMM_FAST_KERNEL(name, bk) \
kernel void name( \
  device const float* input [[buffer(0)]], device const float* weight [[buffer(1)]], \
  device float* output [[buffer(2)]], constant conv2d_gemm_params& p [[buffer(3)]], \
  threadgroup float* As [[threadgroup(0)]], threadgroup float* Bs [[threadgroup(1)]], \
  uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
  uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ conv2d_gemm_fast_body_<bk>(input, weight, output, p, As, Bs, tgid, tid, sid, lane); }

CONV2D_GEMM_FAST_KERNEL(conv2d_gemm_fast_f32_, 16)
#undef CONV2D_GEMM_FAST_KERNEL

#define CONV2D_GEMM_SMALLCH3_KERNEL(name, bk, bm, bn) \
kernel void name( \
  device const float* input [[buffer(0)]], device const float* weight [[buffer(1)]], \
  device float* output [[buffer(2)]], constant conv2d_gemm_params& p [[buffer(3)]], \
  threadgroup float* As [[threadgroup(0)]], threadgroup float* Bs [[threadgroup(1)]], \
  uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
  uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ conv2d_gemm_smallch3_body_<bk, bm, bn>(input, weight, output, p, As, Bs, tgid, tid, sid, lane); }

CONV2D_GEMM_SMALLCH3_KERNEL(conv2d_gemm_smallch3_f32_,    16, 64, 64)
CONV2D_GEMM_SMALLCH3_KERNEL(conv2d_gemm_smallch3_32_f32_, 16, 32, 32)
#undef CONV2D_GEMM_SMALLCH3_KERNEL

