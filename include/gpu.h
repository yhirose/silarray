#pragma once

#include <types.h>
#include <device.h>

#include <sstream>
#include <stdexcept>

namespace sil {

//-----------------------------------------------------------------------------
// GPU context
//-----------------------------------------------------------------------------

// Single source of truth for the simple-PSO list. The X-macro keeps the enum
// values in `gpu::pso` and the kernel-name registration in `gpu_context()`
// in lockstep — adding a new PSO is a single line here.
// FC-specialized variants (sgemm_steel_*) are registered separately in the
// constructor because they need MTLFunctionConstantValues setup.
#define SIL_SIMPLE_PSO_LIST(X) \
  X(kAdd,                   "add")                          \
  X(kSub,                   "sub")                          \
  X(kMul,                   "mul")                          \
  X(kDiv,                   "div")                          \
  X(kPow,                   "pow")                          \
  X(kSigmoid,               "sigmoid_")                     \
  X(kRelu,                  "relu_")                        \
  X(kSumF32,                "sum_f32_")                     \
  X(kLayerNorm,             "layer_norm_")                  \
  X(kSoftmaxF32,            "softmax_f32_")                 \
  X(kAffineF32,             "affine_f32_")                  \
  X(kSigmoidBackwardF32,    "sigmoid_backward_f32_")        \
  X(kBiasSigmoidF32,        "bias_sigmoid_f32_")            \
  X(kSgemm32,               "sgemm_32_")                    \
  X(kSgemm64,               "sgemm_64_")                    \
  X(kMinF32,                "min_f32_")                     \
  X(kMaxF32,                "max_f32_")                     \
  X(kArgmaxF32,             "argmax_f32_")                  \
  X(kSgemvF32,              "sgemv_f32_")                   \
  X(kSgemvK8F32,            "sgemv_k8_f32_")                \
  X(kSoftmaxLoopedF32,      "softmax_looped_f32_")          \
  X(kSumAxis0F32,           "sum_axis0_f32_")               \
  X(kTransposeF32,          "transpose_f32_")               \
  X(kConv2dGemmF32,         "conv2d_gemm_f32_")             \
  X(kConv2dGemmFastF32,     "conv2d_gemm_fast_f32_")        \
  X(kConv2dGemmSmallch3F32, "conv2d_gemm_smallch3_f32_")    \
  X(kConv2dGemmSmallch332F32, "conv2d_gemm_smallch3_32_f32_")

class gpu_context {
 public:
  void* queue;

  struct pipeline {
    void* pso;
    size_t thread_width;
    size_t max_threads;
  };

  static gpu_context& instance() {
    static auto* ctx = new gpu_context();
    return *ctx;
  }

  const pipeline& pso(size_t index) { return psos_[index]; }

  void* command_buffer() {
    if (!cb_) {
      cb_ = objc::send(queue, objc::sel_::commandBuffer());
      gpu_pending_ = true;
    }
    return cb_;
  }

  void* compute_encoder() {
    if (!encoder_) encoder_ = objc::send(command_buffer(), objc::sel_::computeCommandEncoder());
    return encoder_;
  }

  void end_encoder() {
    if (encoder_) {
      objc::send(encoder_, objc::sel_::endEncoding());
      encoder_ = nullptr;
    }
  }

  // Submit GPU commands without waiting — enables CPU-GPU overlap
  void commit() {
    if (!cb_) return;
    end_encoder();
    objc::send(cb_, objc::sel_::commit());
    // Track for later wait in flush()
    last_cb_ = cb_;
    cb_ = nullptr;
  }

  // Register a retained, externally-committed command buffer (e.g. an
  // MPSCommandBuffer from MPSGraph) so the next flush() waits for it.
  // The previous retained buffer is released. flush() releases the last one.
  void register_committed(void* committed_cb) {
    if (retained_cb_) objc::release(retained_cb_);
    retained_cb_ = committed_cb;  // caller passes a +1 reference
    last_cb_ = committed_cb;
    gpu_pending_ = true;
  }

  // Submit (if needed) and wait for GPU completion of all submitted work
  void flush() {
    if (cb_) {
      end_encoder();
      objc::send(cb_, objc::sel_::commit());
      last_cb_ = cb_;
      cb_ = nullptr;
    }
    if (last_cb_) {
      objc::send(last_cb_, objc::sel_::waitUntilCompleted());
      last_cb_ = nullptr;
    }
    if (retained_cb_) {
      objc::release(retained_cb_);
      retained_cb_ = nullptr;
    }
    gpu_pending_ = false;
  }

 private:
  std::vector<pipeline> psos_;
  void* cb_ = nullptr;
  void* last_cb_ = nullptr;
  void* retained_cb_ = nullptr;  // explicitly retained (from register_committed)
  void* encoder_ = nullptr;

  gpu_context() {
    auto* device = buffer_pool::instance().device;
    queue = objc::send(device, "newCommandQueue");

    // Compile MSL source
    auto src = objc::cfstr(msl_source_());
    void* err = nullptr;
    auto lib = reinterpret_cast<void*(*)(void*, SEL, void*, void*, void**)>(
        objc_msgSend)(device, objc::sel("newLibraryWithSource:options:error:"),
                      src, nullptr, &err);
    objc::cfrelease(src);

    if (!lib) {
      auto desc = objc::send(err, "localizedDescription");
      auto s = reinterpret_cast<const char*>(objc::send(desc, "UTF8String"));
      throw std::runtime_error(std::string("gpu: Failed to compile MSL: ") + s);
    }

    // Create pipeline state objects (order must match `gpu::pso` enum, which
    // is generated from the same SIL_SIMPLE_PSO_LIST X-macro).
#define X(_id, str) psos_.push_back(create_pso_(device, lib, str));
    SIL_SIMPLE_PSO_LIST(X)
#undef X

    // STEEL FC-specialized variants: aligned + edge for each kernel
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_steel_",
                                    false, false, true, true));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_steel_",
                                    false, false, false, false));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_steel_",
                                    false, false, true, true));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_steel_",
                                    false, false, false, false));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_sigmoid_steel_",
                                    false, false, true, true));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_sigmoid_steel_",
                                    false, false, false, false));

    // STEEL 32x64 variants for small-M matrices
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_steel_32x64_",
                                    false, false, true, true));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_steel_32x64_",
                                    false, false, false, false));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_steel_32x64_",
                                    false, false, true, true));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_steel_32x64_",
                                    false, false, false, false));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_sigmoid_steel_32x64_",
                                    false, false, true, true));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_sigmoid_steel_32x64_",
                                    false, false, false, false));

    objc::release(lib);
  }

  pipeline create_pso_(void* device, void* library, const char* name) {
    auto fn_name = objc::cfstr(name);
    auto fn = objc::send(library, "newFunctionWithName:", fn_name);
    objc::cfrelease(fn_name);
    if (!fn) {
      throw std::runtime_error(
          std::string("gpu: Failed to find function: ") + name);
    }
    return finalize_pso_(device, fn, name);
  }

  static constexpr unsigned long kMTLDataTypeBool = 53;

  pipeline finalize_pso_(void* device, void* fn, const char* name) {
    void* err = nullptr;
    auto pso = reinterpret_cast<void*(*)(void*, SEL, void*, void**)>(
        objc_msgSend)(device, objc::sel("newComputePipelineStateWithFunction:error:"),
                      fn, &err);
    objc::release(fn);
    if (!pso) {
      throw std::runtime_error(
          std::string("gpu: Failed to create PSO for: ") + name);
    }
    auto w = objc::send_uint(pso, "threadExecutionWidth");
    auto max = objc::send_uint(pso, "maxTotalThreadsPerThreadgroup");
    return {pso, w, max};
  }

  pipeline create_fc_pso_(void* device, void* library, const char* name,
                           bool trans_a, bool trans_b,
                           bool mn_aligned, bool k_aligned) {
    auto fn_name = objc::cfstr(name);
    auto fc_vals = objc::send(objc::send(objc::cls("MTLFunctionConstantValues"),
                                          objc::sel_::alloc()), "init");
    auto set_sel = objc::sel("setConstantValue:type:atIndex:");
    reinterpret_cast<void(*)(void*, SEL, const void*, unsigned long, unsigned long)>(
        objc_msgSend)(fc_vals, set_sel, &trans_a, kMTLDataTypeBool, 0ul);
    reinterpret_cast<void(*)(void*, SEL, const void*, unsigned long, unsigned long)>(
        objc_msgSend)(fc_vals, set_sel, &trans_b, kMTLDataTypeBool, 1ul);
    reinterpret_cast<void(*)(void*, SEL, const void*, unsigned long, unsigned long)>(
        objc_msgSend)(fc_vals, set_sel, &mn_aligned, kMTLDataTypeBool, 2ul);
    reinterpret_cast<void(*)(void*, SEL, const void*, unsigned long, unsigned long)>(
        objc_msgSend)(fc_vals, set_sel, &k_aligned, kMTLDataTypeBool, 3ul);

    void* err = nullptr;
    auto fn = reinterpret_cast<void*(*)(void*, SEL, void*, void*, void**)>(
        objc_msgSend)(library, objc::sel("newFunctionWithName:constantValues:error:"),
                      fn_name, fc_vals, &err);
    objc::cfrelease(fn_name);
    objc::release(fc_vals);

    if (!fn) {
      auto desc = objc::send(err, "localizedDescription");
      auto s = reinterpret_cast<const char*>(objc::send(desc, "UTF8String"));
      throw std::runtime_error(std::string("gpu: FC function error: ") + s);
    }
    return finalize_pso_(device, fn, name);
  }

  static const char* msl_source_() {
    // MSL kernels live in include/gpu_kernels.metal so the file gets proper
    // syntax highlighting and editor jump-to-definition. We embed it as a
    // null-terminated byte array via C23 #embed (clang extension under c++23).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
    static constexpr const char source[] = {
#embed "gpu_kernels.metal"
      , '\0'
    };
#pragma clang diagnostic pop
    return source;
  }

};

//-----------------------------------------------------------------------------
// Public API
//-----------------------------------------------------------------------------

class gpu {
  // PSO indices — generated from SIL_SIMPLE_PSO_LIST so the enum and the
  // registration loop in gpu_context() can never go out of sync.
  enum pso : size_t {
#define X(name, _str) name,
    SIL_SIMPLE_PSO_LIST(X)
#undef X
    // FC-specialized variants follow at the indices after the simple PSOs;
    // these are registered manually in gpu_context() because they need
    // different per-variant constants and a different creation helper.
    kSgemmSteel, kSgemmSteelEdge,
    kSgemmBiasSteel, kSgemmBiasSteelEdge,
    kSgemmBiasSigmoidSteel, kSgemmBiasSigmoidSteelEdge,
    kSgemmSteel32, kSgemmSteel32Edge,
    kSgemmBiasSteel32, kSgemmBiasSteel32Edge,
    kSgemmBiasSigmoidSteel32, kSgemmBiasSigmoidSteel32Edge,
  };

  static constexpr unsigned long kMPSDataTypeFloat32 = 0x10000000 | 32;
  static constexpr size_t kMaxReductionTGSize = 1024;
  static constexpr size_t kMaxReductionTGs = 256;

 public:
  template <value_type T>
  static void add(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 0);
  }

  template <value_type T>
  static void sub(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 1);
  }

  template <value_type T>
  static void mul(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 2);
  }

  template <value_type T>
  static void div(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 3);
  }

  template <value_type T>
  static void pow(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 4);
  }

  static void sigmoid(const storage& IN, storage& OUT) {
    unary_dispatch_(kSigmoid, IN, OUT);
  }

  static void relu(const storage& IN, storage& OUT) {
    unary_dispatch_(kRelu, IN, OUT);
  }

  // Number of partial sums produced by sum_f32.
  static size_t sum_f32_num_tg(size_t length) {
    auto& pl = gpu_context::instance().pso(kSumF32);
    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);
    return std::min((length + tg_size - 1) / tg_size, kMaxReductionTGs);
  }

  // Sum reduction: dispatches threadgroups, each producing a partial sum.
  // Returns the number of partial sums written to OUT.
  static size_t sum_f32(const storage& IN, storage& OUT, size_t length) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kSumF32);

    auto len = static_cast<uint32_t>(length);
    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);
    size_t num_tg = std::min((length + tg_size - 1) / tg_size, kMaxReductionTGs);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &len, sizeof(uint32_t), size_t(2));

    objc::send_dispatch(enc,
                        {num_tg * tg_size, 1, 1},
                        {tg_size, 1, 1});

    return num_tg;
  }

  // Sum along axis=0: 2D tiled, coalesced reads with row-parallel reduction.
  // Single-pass dispatch helper.
  static void sum_axis0_f32_dispatch_(gpu_context& ctx,
                                      const storage& IN, storage& OUT,
                                      uint32_t rows, uint32_t cols, uint32_t bm) {
    auto& pl = ctx.pso(kSumAxis0F32);
    constexpr size_t BX = 32, BY = 32;
    size_t num_tg_x = (cols + BX - 1) / BX;
    size_t num_tg_y = (rows + bm - 1) / bm;

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc, &rows, sizeof(uint32_t), size_t(2));
    objc::send_set_bytes(enc, &cols, sizeof(uint32_t), size_t(3));
    objc::send_set_bytes(enc, &bm, sizeof(uint32_t), size_t(4));

    objc::send_dispatch(enc,
                        {num_tg_x * BX, num_tg_y * BY, 1},
                        {BX, BY, 1});
  }

  // Two-pass sum along axis=0 for large row counts.
  // Pass 1: partition rows into blocks, produce partial sums.
  // Pass 2: reduce partial sums to final output.
  static void sum_axis0_f32(const storage& IN, storage& OUT,
                            size_t rows, size_t cols,
                            const storage& partial, size_t num_row_blocks, size_t bm) {
    auto& ctx = gpu_context::instance();
    auto r = static_cast<uint32_t>(rows);
    auto c = static_cast<uint32_t>(cols);
    auto b = static_cast<uint32_t>(bm);

    if (num_row_blocks <= 1) {
      // Single pass
      sum_axis0_f32_dispatch_(ctx, IN, OUT, r, c, r);
    } else {
      // Pass 1: IN → partial (num_row_blocks × cols)
      sum_axis0_f32_dispatch_(ctx, IN,
                              const_cast<storage&>(partial), r, c, b);
      // End encoder to ensure pass 1 writes are visible to pass 2
      ctx.end_encoder();
      // Pass 2: partial → OUT
      auto nb = static_cast<uint32_t>(num_row_blocks);
      sum_axis0_f32_dispatch_(ctx, partial, OUT, nb, c, nb);
    }
  }

  // Min reduction: dispatches threadgroups, each producing a partial min.
  static size_t min_f32_num_tg(size_t length) {
    auto& pl = gpu_context::instance().pso(kMinF32);
    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);
    return std::min((length + tg_size - 1) / tg_size, kMaxReductionTGs);
  }

  static size_t min_f32(const storage& IN, storage& OUT, size_t length) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kMinF32);

    auto len = static_cast<uint32_t>(length);
    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);
    size_t num_tg = std::min((length + tg_size - 1) / tg_size, kMaxReductionTGs);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &len, sizeof(uint32_t), size_t(2));

    objc::send_dispatch(enc,
                        {num_tg * tg_size, 1, 1},
                        {tg_size, 1, 1});

    return num_tg;
  }

  // Max reduction: dispatches threadgroups, each producing a partial max.
  static size_t max_f32_num_tg(size_t length) {
    auto& pl = gpu_context::instance().pso(kMaxF32);
    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);
    return std::min((length + tg_size - 1) / tg_size, kMaxReductionTGs);
  }

  static size_t max_f32(const storage& IN, storage& OUT, size_t length) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kMaxF32);

    auto len = static_cast<uint32_t>(length);
    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);
    size_t num_tg = std::min((length + tg_size - 1) / tg_size, kMaxReductionTGs);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &len, sizeof(uint32_t), size_t(2));

    objc::send_dispatch(enc,
                        {num_tg * tg_size, 1, 1},
                        {tg_size, 1, 1});

    return num_tg;
  }

  // LayerNorm: one threadgroup per row
  static void layer_norm(const storage& IN, storage& OUT,
                         const storage& gamma, const storage& beta,
                         uint32_t rows, uint32_t cols, float eps) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kLayerNorm);

    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_buffer(enc,
               gamma.mtl_buf, gamma.off * sizeof(float), size_t(2));
    objc::send_set_buffer(enc,
               beta.mtl_buf, beta.off * sizeof(float), size_t(3));
    objc::send_set_bytes(enc,
               &cols, sizeof(uint32_t), size_t(4));
    objc::send_set_bytes(enc,
               &eps, sizeof(float), size_t(5));

    // One threadgroup per row
    objc::send_dispatch(enc,
                        {rows * tg_size, 1, 1},
                        {tg_size, 1, 1});
  }

  // SGEMM via simdgroup_matrix kernel — selects 32×32 or 64×64 tile
  struct gemm_params {
    uint32_t M, N, K, lda, ldb, trans_a, trans_b;
    uint32_t batch_stride_a, batch_stride_b, batch_stride_c;
  };

  static void sgemm(const storage& A, const storage& B, storage& C,
                     uint32_t M, uint32_t N, uint32_t K,
                     uint32_t lda, uint32_t ldb,
                     bool transA, bool transB) {
    sgemm_dispatch_(A, B, C, M, N, K, lda, ldb, transA, transB, 1, 0, 0, 0);
  }

  // Batched SGEMM: batch independent matmuls in a single dispatch.
  // A, B, C are contiguous with batch_stride_* between slices.
  static void sgemm_batched(const storage& A, const storage& B, storage& C,
                             uint32_t M, uint32_t N, uint32_t K,
                             uint32_t lda, uint32_t ldb,
                             uint32_t batch,
                             uint32_t stride_a, uint32_t stride_b, uint32_t stride_c) {
    sgemm_dispatch_(A, B, C, M, N, K, lda, ldb, false, false,
                    batch, stride_a, stride_b, stride_c);
  }

  // Batched STEEL SGEMM
  static void sgemm_steel_batched(const storage& A, const storage& B, storage& C,
                                   uint32_t M, uint32_t N, uint32_t K,
                                   uint32_t lda, uint32_t ldb,
                                   uint32_t batch,
                                   uint32_t stride_a, uint32_t stride_b, uint32_t stride_c) {
    steel_dispatch_batched_(A, B, C, M, N, K, lda, ldb,
                            batch, stride_a, stride_b, stride_c);
  }

 private:
  static void sgemm_dispatch_(const storage& A, const storage& B, storage& C,
                               uint32_t M, uint32_t N, uint32_t K,
                               uint32_t lda, uint32_t ldb,
                               bool transA, bool transB,
                               uint32_t batch,
                               uint32_t stride_a, uint32_t stride_b, uint32_t stride_c);

 public:

  // GEMV: y = x * B where x is 1×K, B is K×N. M==1 specialization.
  // Each thread handles 1 output column.
  static void sgemv(const storage& x, const storage& B, storage& y,
                    uint32_t N, uint32_t K) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso((K % 8 == 0) ? kSgemvK8F32 : kSgemvF32);

    size_t n_threads = N;
    size_t tg_size = std::min(pl.max_threads, size_t(256));
    size_t num_tg = (n_threads + tg_size - 1) / tg_size;

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               x.mtl_buf, x.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               B.mtl_buf, B.off * sizeof(float), size_t(1));
    objc::send_set_buffer(enc,
               y.mtl_buf, y.off * sizeof(float), size_t(2));
    objc::send_set_bytes(enc,
               &N, sizeof(uint32_t), size_t(3));
    objc::send_set_bytes(enc,
               &K, sizeof(uint32_t), size_t(4));

    objc::send_dispatch(enc,
                        {num_tg * tg_size, 1, 1},
                        {tg_size, 1, 1});
  }

  // STEEL 64×64 sgemm — auto-selects aligned vs edge variant
  static void sgemm_steel(const storage& A, const storage& B, storage& C,
                           uint32_t M, uint32_t N, uint32_t K,
                           uint32_t lda, uint32_t ldb) {
    steel_dispatch_(A, B, C, nullptr, 0, M, N, K, lda, ldb, false);
  }

  // Fused dot + bias — single dispatch replaces dot + add
  static void sgemm_bias_steel(
      const storage& A, const storage& B, storage& C,
      const storage& bias, uint32_t bias_len,
      uint32_t M, uint32_t N, uint32_t K,
      uint32_t lda, uint32_t ldb) {
    steel_dispatch_(A, B, C, &bias, bias_len, M, N, K, lda, ldb, false);
  }

 private:
  // Shared STEEL dispatch — selects 32x64 or 64x64, aligned vs edge PSO.
  static void steel_dispatch_(
      const storage& A, const storage& B, storage& C,
      const storage* bias, uint32_t bias_len,
      uint32_t M, uint32_t N, uint32_t K,
      uint32_t lda, uint32_t ldb, bool with_sigmoid);

  static void steel_dispatch_batched_(
      const storage& A, const storage& B, storage& C,
      uint32_t M, uint32_t N, uint32_t K,
      uint32_t lda, uint32_t ldb,
      uint32_t batch,
      uint32_t stride_a, uint32_t stride_b, uint32_t stride_c);

 public:
  // Fused dot + bias + sigmoid — single dispatch replaces dot + bias_sigmoid
  static void sgemm_bias_sigmoid_steel(
      const storage& A, const storage& B, storage& C,
      const storage& bias, uint32_t bias_len,
      uint32_t M, uint32_t N, uint32_t K,
      uint32_t lda, uint32_t ldb) {
    steel_dispatch_(A, B, C, &bias, bias_len, M, N, K, lda, ldb, true);
  }

  // GPU transpose: {rows, cols} → {cols, rows}
  static void transpose(const storage& input, storage& output,
                        uint32_t rows, uint32_t cols) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kTransposeF32);

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc, input.mtl_buf, input.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc, output.mtl_buf, output.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc, &rows, sizeof(uint32_t), size_t(2));
    objc::send_set_bytes(enc, &cols, sizeof(uint32_t), size_t(3));

    // Tiled transpose: 32x32 threadgroups
    constexpr size_t TILE = 32;
    size_t grid_x = (cols + TILE - 1) / TILE;
    size_t grid_y = (rows + TILE - 1) / TILE;
    static auto dispatchTG = objc::sel("dispatchThreadgroups:threadsPerThreadgroup:");
    reinterpret_cast<void(*)(void*, SEL, objc::mtl_size, objc::mtl_size)>(objc_msgSend)(
        enc, dispatchTG,
        objc::mtl_size{grid_x, grid_y, 1},
        objc::mtl_size{TILE, TILE, 1});
  }

  // Implicit GEMM conv2d: no im2col buffer, reads input on-the-fly during GEMM.
  // Weight must be pre-transposed to {K_dim, C_out}.
  // SmallChannels variant: takes weight in natural (C_out, K, K, C_in) layout
  // (no separate transpose dispatch). Currently specialized for n_channels=3.
  static void conv2d_gemm_smallch3(const storage& input, const storage& weight,
                                   storage& output,
                                   uint32_t N, uint32_t H, uint32_t W, uint32_t C_in,
                                   uint32_t K, uint32_t C_out);

  static void conv2d_gemm(const storage& input, const storage& weight, storage& output,
                          uint32_t N, uint32_t H, uint32_t W, uint32_t C_in,
                          uint32_t K, uint32_t C_out);

  // MPSGraph conv2d cache — one precompiled graph per shape.
  // Apple's AOT-compiled MPSGraph kernels outperform our JIT-compiled
  // conv2d_gemm for medium/large channel counts (ResNet mid/deep class).
  struct conv2d_mps_key {
    uint32_t N, H, W, C_in, K, C_out;
    bool operator==(const conv2d_mps_key&) const = default;
  };
  struct conv2d_mps_hash {
    size_t operator()(const conv2d_mps_key& k) const {
      return size_t(k.N) ^ (size_t(k.H) << 6) ^ (size_t(k.W) << 12) ^
             (size_t(k.C_in) << 20) ^ (size_t(k.K) << 30) ^ (size_t(k.C_out) << 40);
    }
  };
  struct conv2d_mps_entry {
    void* graph;          // MPSGraph*
    void* input_tensor;   // MPSGraphTensor* placeholder (NHWC)
    void* weight_tensor;  // MPSGraphTensor* placeholder (OHWI)
    void* output_tensor;  // MPSGraphTensor* result of conv2d op
    void* input_shape;    // NSArray<NSNumber*>*
    void* weight_shape;   // NSArray<NSNumber*>*
    void* output_shape;   // NSArray<NSNumber*>*
  };

  static const conv2d_mps_entry& get_conv2d_mps_entry_(
      uint32_t N, uint32_t H, uint32_t W,
      uint32_t C_in, uint32_t K, uint32_t C_out) {
    static std::unordered_map<conv2d_mps_key, conv2d_mps_entry, conv2d_mps_hash> cache;
    auto key = conv2d_mps_key{N, H, W, C_in, K, C_out};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    // Graph construction uses autoreleased objects; keep a local pool so
    // temporaries (descriptor, permutation array) are cleaned up immediately.
    void* pool = objc::send(objc::send(objc::cls("NSAutoreleasePool"), objc::sel_::alloc()), "init");

    void* graph = objc::send(objc::send(objc::cls("MPSGraph"), objc::sel_::alloc()), "init");

    int input_dims[4] = {int(N), int(H), int(W), int(C_in)};
    int weight_dims[4] = {int(C_out), int(K), int(K), int(C_in)};
    int output_dims[4] = {int(N), int(H), int(W), int(C_out)};
    void* input_shape = objc::retain(objc::ns_numbers(input_dims, 4));
    void* weight_shape = objc::retain(objc::ns_numbers(weight_dims, 4));
    void* output_shape = objc::retain(objc::ns_numbers(output_dims, 4));

    static auto placeholder_sel = objc::sel("placeholderWithShape:dataType:name:");
    void* input_tensor = reinterpret_cast<void*(*)(void*, SEL, void*, unsigned long, void*)>(
        objc_msgSend)(graph, placeholder_sel, input_shape, kMPSDataTypeFloat32, nullptr);
    void* weight_tensor = reinterpret_cast<void*(*)(void*, SEL, void*, unsigned long, void*)>(
        objc_msgSend)(graph, placeholder_sel, weight_shape, kMPSDataTypeFloat32, nullptr);

    // Weight layout: our tensor is (C_out, K_h, K_w, C_in) = OHWI.
    // MPSGraphTensorNamedDataLayoutOIHW requires (O, I, H, W). Transpose
    // within the graph — MPSGraph fuses this into the conv2d op.
    int perm[4] = {0, 3, 1, 2};
    void* perm_array = objc::ns_numbers(perm, 4);
    static auto transpose_sel = objc::sel("transposeTensor:permutation:name:");
    void* weight_oihw = reinterpret_cast<void*(*)(void*, SEL, void*, void*, void*)>(
        objc_msgSend)(graph, transpose_sel, weight_tensor, perm_array, nullptr);

    uint32_t pad = K / 2;
    constexpr unsigned long kPaddingExplicit = 0;
    constexpr unsigned long kLayoutNHWC = 1;
    constexpr unsigned long kLayoutOIHW = 2;

    static auto desc_sel = objc::sel(
        "descriptorWithStrideInX:strideInY:dilationRateInX:dilationRateInY:"
        "groups:paddingLeft:paddingRight:paddingTop:paddingBottom:"
        "paddingStyle:dataLayout:weightsLayout:");
    typedef void* (*DescFn)(void*, SEL,
        unsigned long, unsigned long,          // stride x,y
        unsigned long, unsigned long,          // dilation x,y
        unsigned long,                         // groups
        unsigned long, unsigned long, unsigned long, unsigned long,  // padding l,r,t,b
        unsigned long, unsigned long, unsigned long);                // style, dataLayout, weightsLayout
    void* desc = reinterpret_cast<DescFn>(objc_msgSend)(
        objc::cls("MPSGraphConvolution2DOpDescriptor"), desc_sel,
        1ul, 1ul, 1ul, 1ul, 1ul,
        (unsigned long)pad, (unsigned long)pad, (unsigned long)pad, (unsigned long)pad,
        kPaddingExplicit, kLayoutNHWC, kLayoutOIHW);

    static auto conv_sel = objc::sel(
        "convolution2DWithSourceTensor:weightsTensor:descriptor:name:");
    void* output_tensor = reinterpret_cast<void*(*)(void*, SEL, void*, void*, void*, void*)>(
        objc_msgSend)(graph, conv_sel, input_tensor, weight_oihw, desc, nullptr);

    // Explicitly retain the tensor handles — they are autoreleased returns
    // from graph.placeholderWith/convolution2DWith, and survive the pool
    // drain below only if we hold a reference.
    objc::retain(input_tensor);
    objc::retain(weight_tensor);
    objc::retain(output_tensor);
    cache[key] = {graph, input_tensor, weight_tensor, output_tensor,
                  input_shape, weight_shape, output_shape};

    objc::release(pool);
    return cache[key];
  }

  static void conv2d_mpsgraph(const storage& input, const storage& weight, storage& output,
                              uint32_t N, uint32_t H, uint32_t W, uint32_t C_in,
                              uint32_t K, uint32_t C_out) {
    const auto& entry = get_conv2d_mps_entry_(N, H, W, C_in, K, C_out);
    auto& ctx = gpu_context::instance();

    void* pool = objc::send(objc::send(objc::cls("NSAutoreleasePool"), objc::sel_::alloc()), "init");

    // Wrap buffers as MPSGraphTensorData. MTLBuffer offset isn't supported
    // directly — caller guarantees off == 0 (see conv2d dispatch gate).
    static auto td_init_sel = objc::sel("initWithMTLBuffer:shape:dataType:");
    auto make_td = [&](void* buf, void* shape) {
      void* obj = objc::send(objc::cls("MPSGraphTensorData"), objc::sel_::alloc());
      return reinterpret_cast<void*(*)(void*, SEL, void*, void*, unsigned long)>(
          objc_msgSend)(obj, td_init_sel, buf, shape, kMPSDataTypeFloat32);
    };
    void* input_td = make_td(input.mtl_buf, entry.input_shape);
    void* weight_td = make_td(weight.mtl_buf, entry.weight_shape);
    void* output_td = make_td(output.mtl_buf, entry.output_shape);

    // feeds = {input_tensor: input_td, weight_tensor: weight_td}
    // results = {output_tensor: output_td}
    void* feeds_keys[2] = {entry.input_tensor, entry.weight_tensor};
    void* feeds_vals[2] = {input_td, weight_td};
    static auto dict_sel = objc::sel("dictionaryWithObjects:forKeys:count:");
    void* feeds = reinterpret_cast<void*(*)(void*, SEL, void**, void**, size_t)>(
        objc_msgSend)(objc::cls("NSDictionary"), dict_sel, feeds_vals, feeds_keys, 2ul);
    void* results = reinterpret_cast<void*(*)(void*, SEL, void**, void**, size_t)>(
        objc_msgSend)(objc::cls("NSDictionary"), dict_sel, &output_td, (void**)&entry.output_tensor, 1ul);

    // Commit any pending compute work first. MPSGraph gets its own fresh
    // command buffer from the queue so its work runs after ours in FIFO order.
    ctx.commit();

    static auto from_queue_sel = objc::sel("commandBufferFromCommandQueue:");
    void* mps_cb = reinterpret_cast<void*(*)(void*, SEL, void*)>(objc_msgSend)(
        objc::cls("MPSCommandBuffer"), from_queue_sel, ctx.queue);
    // commandBufferFromCommandQueue: returns autoreleased; retain so it
    // survives past the pool drain at the end of this function (flush()
    // will waitUntilCompleted on it, then we release).
    objc::retain(mps_cb);

    static auto encode_sel = objc::sel(
        "encodeToCommandBuffer:feeds:targetOperations:resultsDictionary:executionDescriptor:");
    reinterpret_cast<void(*)(void*, SEL, void*, void*, void*, void*, void*)>(objc_msgSend)(
        entry.graph, encode_sel, mps_cb, feeds, nullptr, results, nullptr);

    objc::send(mps_cb, objc::sel_::commit());
    ctx.register_committed(mps_cb);

    objc::release(input_td);
    objc::release(weight_td);
    objc::release(output_td);
    objc::release(pool);
  }

  // Unified conv2d entry point. Picks the right kernel variant for the input
  // shape and handles the temporary transpose buffer internally so callers
  // don't have to know about the dispatch table.
  //
  //   * C_in == 3                → smallch3 path (reads weight in natural layout)
  //   * off==0 and C_in >= 16   → MPSGraph (Apple AOT kernels)
  //   * otherwise                → transpose dispatch + conv2d_gemm (JIT fallback)
  static void conv2d(const storage& input, const storage& weight, storage& output,
                     uint32_t N, uint32_t H, uint32_t W, uint32_t C_in,
                     uint32_t K, uint32_t C_out) {
    if (C_in == 3) {
      conv2d_gemm_smallch3(input, weight, output, N, H, W, C_in, K, C_out);
      return;
    }
    // MPSGraph path: requires 0-offset buffers (MPSGraphTensorData lacks an
    // offset parameter). All current callers pass freshly-allocated outputs
    // and non-sliced inputs, so this is satisfied in practice.
    if (input.off == 0 && weight.off == 0 && output.off == 0 && C_in >= 16) {
      conv2d_mpsgraph(input, weight, output, N, H, W, C_in, K, C_out);
      return;
    }
    uint32_t K_dim = K * K * C_in;
    auto w_t = storage::make(size_t(K_dim) * C_out * sizeof(float));
    transpose(weight, w_t, C_out, K_dim);
    conv2d_gemm(input, w_t, output, N, H, W, C_in, K, C_out);
  }

  // Cached MPS matmul kernel — avoids repeated alloc/init for same dimensions
  struct mps_matmul_key {
    size_t M, N, K;
    bool tA, tB;
    bool operator==(const mps_matmul_key&) const = default;
  };
  struct mps_matmul_hash {
    size_t operator()(const mps_matmul_key& k) const {
      return k.M ^ (k.N << 16) ^ (k.K << 32) ^ (size_t(k.tA) << 48) ^ (size_t(k.tB) << 49);
    }
  };

  static void* get_mps_matmul_(size_t M, size_t N, size_t K, bool tA, bool tB) {
    static std::unordered_map<mps_matmul_key, void*, mps_matmul_hash> cache;
    auto key = mps_matmul_key{M, N, K, tA, tB};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto obj = objc::send_mps_matmul_init(
        objc::send(objc::cls("MPSMatrixMultiplication"), objc::sel_::alloc()),
        buffer_pool::instance().device, tA, tB, M, N, K, 1.0, 0.0);
    cache[key] = obj;
    return obj;
  }

  // Cached MPS descriptor — avoids repeated class method calls for same dimensions
  struct mps_desc_key {
    size_t rows, cols;
    bool operator==(const mps_desc_key&) const = default;
  };
  struct mps_desc_hash {
    size_t operator()(const mps_desc_key& k) const { return k.rows ^ (k.cols << 32); }
  };

  static void* get_mps_desc_(size_t rows, size_t cols) {
    static std::unordered_map<mps_desc_key, void*, mps_desc_hash> cache;
    auto key = mps_desc_key{rows, cols};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto desc = objc::send_mps_desc(
        objc::cls("MPSMatrixDescriptor"), rows, cols,
        cols * sizeof(float), kMPSDataTypeFloat32);
    cache[key] = desc;
    return desc;
  }

  // Matrix multiplication via MPS with transpose support
  static void dot_f32_ex(const storage& A, const storage& B, storage& OUT,
                         size_t phys_A_rows, size_t phys_A_cols,
                         size_t phys_B_rows, size_t phys_B_cols,
                         size_t M, size_t N, size_t K,
                         bool transA, bool transB);

  // Argmax: one threadgroup per row, returns int index per row
  static void argmax_f32(const storage& IN, storage& OUT,
                         uint32_t rows, uint32_t cols) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kArgmaxF32);

    size_t needed = ((cols + 3) / 4 + 31) & ~size_t(31);
    if (needed < 32) needed = 32;  // warp reduction requires ≥1 simdgroup
    size_t tg_size = std::min(needed, std::min(pl.max_threads, kMaxReductionTGSize));

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(int), size_t(1));
    objc::send_set_bytes(enc,
               &cols, sizeof(uint32_t), size_t(2));

    // One threadgroup per row
    objc::send_dispatch(enc,
                        {rows * tg_size, 1, 1},
                        {tg_size, 1, 1});
  }

  // Softmax via custom Metal kernel — one threadgroup per row
  static void softmax(const storage& IN, storage& OUT,
                      uint32_t rows, uint32_t cols) {
    auto& ctx = gpu_context::instance();

    // Select kernel: register-cached for cols <= 4096, looped for larger
    bool use_looped = (cols > 4096);
    auto& pl = ctx.pso(use_looped ? kSoftmaxLoopedF32 : kSoftmaxF32);

    // TG size: ceil(cols/4) rounded to simdgroup width, capped at hardware max
    size_t needed = ((cols + 3) / 4 + 31) & ~size_t(31);
    size_t tg_size = std::min(needed, std::min(pl.max_threads, kMaxReductionTGSize));

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &cols, sizeof(uint32_t), size_t(2));

    // One threadgroup per row
    objc::send_dispatch(enc,
                        {rows * tg_size, 1, 1},
                        {tg_size, 1, 1});
  }

  // out[i] = dout[i] * sigmoid(x[i]) * (1 - sigmoid(x[i])) — fused backward
  static void sigmoid_backward(const storage& dout, const storage& x,
                                storage& OUT, size_t n) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kSigmoidBackwardF32);

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               dout.mtl_buf, dout.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               x.mtl_buf, x.off * sizeof(float), size_t(1));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(2));

    auto tw = pl.thread_width;
    auto tg = std::min(n, pl.max_threads - (pl.max_threads % tw));
    objc::send_dispatch(enc, {n, 1, 1}, {tg, 1, 1});
  }

  // In-place: data[i] = sigmoid(data[i] + bias[i % cols])
  static void bias_sigmoid(storage& data, const storage& bias,
                           size_t n, uint32_t cols) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kBiasSigmoidF32);

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               data.mtl_buf, data.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               bias.mtl_buf, bias.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &cols, sizeof(uint32_t), size_t(2));

    auto tw = pl.thread_width;
    auto tg = std::min(n, pl.max_threads - (pl.max_threads % tw));
    objc::send_dispatch(enc, {n, 1, 1}, {tg, 1, 1});
  }

  // out[i] = in[i] * scale + offset — GPU-side affine, no CPU sync needed
  static void affine(const storage& IN, storage& OUT, size_t n,
                     float scale, float offset) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kAffineF32);

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &scale, sizeof(float), size_t(2));
    objc::send_set_bytes(enc,
               &offset, sizeof(float), size_t(3));

    auto tw = pl.thread_width;
    auto tg = std::min(n, pl.max_threads - (pl.max_threads % tw));
    objc::send_dispatch(enc, {n, 1, 1}, {tg, 1, 1});
  }

 private:
  // Shared dispatch for unary float4-vectorized kernels (sigmoid, relu, etc.)
  static void unary_dispatch_(size_t pso_index,
                              const storage& IN, storage& OUT) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(pso_index);

    auto len = static_cast<uint32_t>(OUT.len);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &len, sizeof(uint32_t), size_t(2));

    auto grid_len = (OUT.len + 3) / 4;
    auto h = pl.max_threads / pl.thread_width;

    objc::send_dispatch(enc,
                        {grid_len, 1, 1},
                        {pl.thread_width, h, 1});
  }

  template <value_type T>
  static void arithmetic_dispatch_(const storage& A, const storage& B,
                                   storage& OUT, size_t pso_index);
};

//-----------------------------------------------------------------------------
// Implementation
//
// Large (≥30 line) gpu:: methods are defined here rather than inline in the
// class body to keep the public-facing class declaration navigable. Small
// dispatchers and one-line wrappers stay inline in the class.
//-----------------------------------------------------------------------------

inline void gpu::sgemm_dispatch_(const storage& A, const storage& B, storage& C,
                                 uint32_t M, uint32_t N, uint32_t K,
                                 uint32_t lda, uint32_t ldb,
                                 bool transA, bool transB,
                                 uint32_t batch,
                                 uint32_t stride_a, uint32_t stride_b, uint32_t stride_c) {
  auto& ctx = gpu_context::instance();

  auto BM = 32u, BN = 32u;
  auto& pl = ctx.pso(kSgemm32);

  gemm_params p{M, N, K, lda, ldb, transA ? 1u : 0u, transB ? 1u : 0u,
                stride_a, stride_b, stride_c};

  auto enc = ctx.compute_encoder();
  objc::send_set_pso(enc, pl.pso);
  objc::send_set_buffer(enc,
             A.mtl_buf, A.off * sizeof(float), size_t(0));
  objc::send_set_buffer(enc,
             B.mtl_buf, B.off * sizeof(float), size_t(1));
  objc::send_set_buffer(enc,
             C.mtl_buf, C.off * sizeof(float), size_t(2));
  objc::send_set_bytes(enc,
             &p, sizeof(p), size_t(3));

  size_t grid_x = (N + BN - 1) / BN;
  size_t grid_y = (M + BM - 1) / BM;

  reinterpret_cast<void(*)(void*, SEL, objc::mtl_size, objc::mtl_size)>(
      objc_msgSend)(
      enc, objc::sel("dispatchThreadgroups:threadsPerThreadgroup:"),
      objc::mtl_size{grid_x, grid_y, batch},
      objc::mtl_size{32, 2, 2});
}

inline void gpu::steel_dispatch_(
    const storage& A, const storage& B, storage& C,
    const storage* bias, uint32_t bias_len,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t lda, uint32_t ldb, bool with_sigmoid) {
  auto& ctx = gpu_context::instance();

  // Use 32x64 tiles when M is 33-96 to double threadgroup count.
  // M<=32 stays on 64x64 (1 tile fits perfectly, less overhead).
  // M>=97 has enough tiles in 64x64.
  bool use_32 = (M > 32 && M < 97);
  uint32_t BM = use_32 ? 32 : 64;

  bool aligned = (M % BM == 0) && (N % 64 == 0) && (K % 16 == 0);

  pso pso_id;
  if (use_32) {
    if (bias && with_sigmoid)
      pso_id = aligned ? kSgemmBiasSigmoidSteel32 : kSgemmBiasSigmoidSteel32Edge;
    else if (bias)
      pso_id = aligned ? kSgemmBiasSteel32 : kSgemmBiasSteel32Edge;
    else
      pso_id = aligned ? kSgemmSteel32 : kSgemmSteel32Edge;
  } else {
    if (bias && with_sigmoid)
      pso_id = aligned ? kSgemmBiasSigmoidSteel : kSgemmBiasSigmoidSteelEdge;
    else if (bias)
      pso_id = aligned ? kSgemmBiasSteel : kSgemmBiasSteelEdge;
    else
      pso_id = aligned ? kSgemmSteel : kSgemmSteelEdge;
  }
  auto& pl = ctx.pso(pso_id);

  uint32_t tiles_n = (N + 63) / 64, tiles_m = (M + BM - 1) / BM;
  uint32_t swizzle_log = 0;
  while ((tiles_n >> (swizzle_log + 1)) >= 1 && swizzle_log < 3)
    swizzle_log++;

  gemm_params p{M, N, K, lda, ldb, swizzle_log, 0, 0, 0, 0};

  auto enc = ctx.compute_encoder();
  objc::send_set_pso(enc, pl.pso);
  objc::send_set_buffer(enc, A.mtl_buf, A.off * sizeof(float), size_t(0));
  objc::send_set_buffer(enc, B.mtl_buf, B.off * sizeof(float), size_t(1));
  objc::send_set_buffer(enc, C.mtl_buf, C.off * sizeof(float), size_t(2));
  objc::send_set_bytes(enc, &p, sizeof(p), size_t(3));

  if (bias) {
    objc::send_set_buffer(enc, bias->mtl_buf, bias->off * sizeof(float), size_t(4));
    objc::send_set_bytes(enc, &bias_len, sizeof(bias_len), size_t(5));
  }

  size_t tgp_a = BM * (16 + 4) * sizeof(float);
  constexpr size_t tgp_b = 16 * (64 + 4) * sizeof(float);
  static auto setTgpMem = objc::sel("setThreadgroupMemoryLength:atIndex:");
  reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
      enc, setTgpMem, tgp_a, size_t(0));
  reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
      enc, setTgpMem, tgp_b, size_t(1));

  size_t grid_x = size_t(tiles_n) << swizzle_log;
  size_t grid_y = (size_t(tiles_m) + ((1u << swizzle_log) - 1)) >> swizzle_log;

  // Both 32x64 and 64x64 use 128 threads (WM=2, WN=2)
  reinterpret_cast<void(*)(void*, SEL, objc::mtl_size, objc::mtl_size)>(
      objc_msgSend)(enc, objc::sel_::dispatchTG(),
      objc::mtl_size{grid_x, grid_y, 1},
      objc::mtl_size{32, 2, 2});
}

inline void gpu::steel_dispatch_batched_(
    const storage& A, const storage& B, storage& C,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t lda, uint32_t ldb,
    uint32_t batch,
    uint32_t stride_a, uint32_t stride_b, uint32_t stride_c) {
  auto& ctx = gpu_context::instance();

  bool aligned = (M % 64 == 0) && (N % 64 == 0) && (K % 16 == 0);
  pso pso_id = aligned ? kSgemmSteel : kSgemmSteelEdge;
  auto& pl = ctx.pso(pso_id);

  uint32_t tiles_n = (N + 63) / 64, tiles_m = (M + 63) / 64;
  uint32_t swizzle_log = 0;
  while ((tiles_n >> (swizzle_log + 1)) >= 1 && swizzle_log < 3)
    swizzle_log++;

  gemm_params p{M, N, K, lda, ldb, swizzle_log, 0,
                stride_a, stride_b, stride_c};

  auto enc = ctx.compute_encoder();
  objc::send_set_pso(enc, pl.pso);
  objc::send_set_buffer(enc, A.mtl_buf, A.off * sizeof(float), size_t(0));
  objc::send_set_buffer(enc, B.mtl_buf, B.off * sizeof(float), size_t(1));
  objc::send_set_buffer(enc, C.mtl_buf, C.off * sizeof(float), size_t(2));
  objc::send_set_bytes(enc, &p, sizeof(p), size_t(3));

  constexpr size_t tgp_a = 64 * (16 + 4) * sizeof(float);
  constexpr size_t tgp_b = 16 * (64 + 4) * sizeof(float);
  static auto setTgpMem = objc::sel("setThreadgroupMemoryLength:atIndex:");
  reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
      enc, setTgpMem, tgp_a, size_t(0));
  reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
      enc, setTgpMem, tgp_b, size_t(1));

  size_t grid_x = size_t(tiles_n) << swizzle_log;
  size_t grid_y = (size_t(tiles_m) + ((1u << swizzle_log) - 1)) >> swizzle_log;

  reinterpret_cast<void(*)(void*, SEL, objc::mtl_size, objc::mtl_size)>(
      objc_msgSend)(enc, objc::sel_::dispatchTG(),
      objc::mtl_size{grid_x, grid_y, batch},
      objc::mtl_size{32, 2, 2});
}

inline void gpu::conv2d_gemm_smallch3(const storage& input, const storage& weight,
                                      storage& output,
                                      uint32_t N, uint32_t H, uint32_t W, uint32_t C_in,
                                      uint32_t K, uint32_t C_out) {
  auto& ctx = gpu_context::instance();
  constexpr uint32_t BK = 16;

  uint32_t pad = K / 2;
  uint32_t H_out = H, W_out = W;
  uint32_t M = N * H_out * W_out;
  uint32_t K_dim = K * K * C_in;

  // Pick 32×32 tile when C_out is small (≤ 32) to avoid wasting half of a
  // 64-wide N tile. ImageNet first (C_out=32) and similar shallow first
  // layers benefit dramatically.
  bool use_32 = (C_out <= 32);
  uint32_t BM = use_32 ? 32 : 64;
  uint32_t BN = use_32 ? 32 : 64;
  auto& pl = ctx.pso(use_32 ? kConv2dGemmSmallch332F32 : kConv2dGemmSmallch3F32);
  uint32_t tiles_m = (M + BM - 1) / BM, tiles_n = (C_out + BN - 1) / BN;
  uint32_t swizzle_log = 0;
  while ((tiles_n >> (swizzle_log + 1)) >= 1 && swizzle_log < 3)
    swizzle_log++;

  struct { uint32_t M, C_out, K_dim, N_batch, H, W, C_in, K_sz, pad, H_out, W_out, swizzle; } p
      {M, C_out, K_dim, N, H, W, C_in, K, pad, H_out, W_out, swizzle_log};

  auto enc = ctx.compute_encoder();
  objc::send_set_pso(enc, pl.pso);
  objc::send_set_buffer(enc, input.mtl_buf, input.off * sizeof(float), size_t(0));
  objc::send_set_buffer(enc, weight.mtl_buf, weight.off * sizeof(float), size_t(1));
  objc::send_set_buffer(enc, output.mtl_buf, output.off * sizeof(float), size_t(2));
  objc::send_set_bytes(enc, &p, sizeof(p), size_t(3));

  // A is (BM, BK), B is (BN, BK) — both have BK=16 columns with pad_a/pad_b=4
  size_t tgp_a = BM * (BK + 4) * sizeof(float);
  size_t tgp_b = BN * (BK + 4) * sizeof(float);
  static auto setTgpMem = objc::sel("setThreadgroupMemoryLength:atIndex:");
  reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
      enc, setTgpMem, tgp_a, size_t(0));
  reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
      enc, setTgpMem, tgp_b, size_t(1));

  size_t grid_x = size_t(tiles_n) << swizzle_log;
  size_t grid_y = (size_t(tiles_m) + ((1u << swizzle_log) - 1)) >> swizzle_log;

  reinterpret_cast<void(*)(void*, SEL, objc::mtl_size, objc::mtl_size)>(
      objc_msgSend)(enc, objc::sel_::dispatchTG(),
      objc::mtl_size{grid_x, grid_y, 1},
      objc::mtl_size{32, 2, 2});
}

inline void gpu::conv2d_gemm(const storage& input, const storage& weight, storage& output,
                             uint32_t N, uint32_t H, uint32_t W, uint32_t C_in,
                             uint32_t K, uint32_t C_out) {
  auto& ctx = gpu_context::instance();

  constexpr uint32_t BM = 64, BN = 64, BK = 16;

  uint32_t pad = K / 2;
  uint32_t H_out = H, W_out = W;
  uint32_t M = N * H_out * W_out;
  uint32_t K_dim = K * K * C_in;

  // Fast variant: hoists bounds checks out of the K-block inner loop. Requires
  // C_in divisible by BK so each (kh,kw) cleanly contains an integer number of
  // ci_block iterations. Covers all ResNet-like layers (C_in ∈ {64,128,256,...}).
  bool use_fast = (C_in % BK == 0);
  auto& pl = ctx.pso(use_fast ? kConv2dGemmFastF32 : kConv2dGemmF32);
  uint32_t tiles_m = (M + BM - 1) / BM, tiles_n = (C_out + BN - 1) / BN;
  uint32_t swizzle_log = 0;
  while ((tiles_n >> (swizzle_log + 1)) >= 1 && swizzle_log < 3)
    swizzle_log++;

  struct { uint32_t M, C_out, K_dim, N_batch, H, W, C_in, K_sz, pad, H_out, W_out, swizzle; } p
      {M, C_out, K_dim, N, H, W, C_in, K, pad, H_out, W_out, swizzle_log};

  auto enc = ctx.compute_encoder();
  objc::send_set_pso(enc, pl.pso);
  objc::send_set_buffer(enc, input.mtl_buf, input.off * sizeof(float), size_t(0));
  objc::send_set_buffer(enc, weight.mtl_buf, weight.off * sizeof(float), size_t(1));
  objc::send_set_buffer(enc, output.mtl_buf, output.off * sizeof(float), size_t(2));
  objc::send_set_bytes(enc, &p, sizeof(p), size_t(3));

  size_t tgp_a = BM * (BK + 4) * sizeof(float);
  size_t tgp_b = BK * (BN + 4) * sizeof(float);
  static auto setTgpMem = objc::sel("setThreadgroupMemoryLength:atIndex:");
  reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
      enc, setTgpMem, tgp_a, size_t(0));
  reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
      enc, setTgpMem, tgp_b, size_t(1));

  size_t grid_x = size_t(tiles_n) << swizzle_log;
  size_t grid_y = (size_t(tiles_m) + ((1u << swizzle_log) - 1)) >> swizzle_log;

  reinterpret_cast<void(*)(void*, SEL, objc::mtl_size, objc::mtl_size)>(
      objc_msgSend)(enc, objc::sel_::dispatchTG(),
      objc::mtl_size{grid_x, grid_y, 1},
      objc::mtl_size{32, 2, 2});
}

inline void gpu::dot_f32_ex(const storage& A, const storage& B, storage& OUT,
                            size_t phys_A_rows, size_t phys_A_cols,
                            size_t phys_B_rows, size_t phys_B_cols,
                            size_t M, size_t N, size_t K,
                            bool transA, bool transB) {
  auto& ctx = gpu_context::instance();
  static auto mat_cls = objc::cls("MPSMatrix");

  auto descA = get_mps_desc_(phys_A_rows, phys_A_cols);
  auto descB = get_mps_desc_(phys_B_rows, phys_B_cols);
  auto descC = get_mps_desc_(M, N);

  auto matA = objc::send(objc::send(mat_cls, objc::sel_::alloc()),
                         objc::sel_::initBuffer(),
                         A.mtl_buf, A.off * sizeof(float), (size_t)(uintptr_t)descA);
  auto matB = objc::send(objc::send(mat_cls, objc::sel_::alloc()),
                         objc::sel_::initBuffer(),
                         B.mtl_buf, B.off * sizeof(float), (size_t)(uintptr_t)descB);
  auto matC = objc::send(objc::send(mat_cls, objc::sel_::alloc()),
                         objc::sel_::initBuffer(),
                         OUT.mtl_buf, OUT.off * sizeof(float), (size_t)(uintptr_t)descC);

  auto matMul = get_mps_matmul_(M, N, K, transA, transB);

  ctx.end_encoder();
  auto cb = ctx.command_buffer();
  objc::send_mps_encode(matMul, cb, matA, matB, matC);

  // Only release MPSMatrix wrappers (matMul and descriptors are cached)
  objc::release(matA);
  objc::release(matB);
  objc::release(matC);
}

template <value_type T>
inline void gpu::arithmetic_dispatch_(const storage& A, const storage& B,
                                      storage& OUT, size_t pso_index) {
  auto& ctx = gpu_context::instance();
  auto& pl = ctx.pso(pso_index);

  auto a_len = static_cast<uint32_t>(A.len);
  auto b_len = static_cast<uint32_t>(B.len);
  uint32_t dtype = std::is_same_v<T, float> ? 0u : 1u;
  auto out_len = static_cast<uint32_t>(OUT.len);
  // Precompute Barrett inverse for fast modulo on GPU
  auto b_inv = b_len > 0 ? static_cast<uint32_t>((uint64_t(1) << 32) / b_len + 1) : 0u;

  auto enc = ctx.compute_encoder();

  objc::send_set_pso(enc, pl.pso);
  objc::send_set_buffer(enc,
             A.mtl_buf, A.off * sizeof(T), size_t(0));
  objc::send_set_buffer(enc,
             B.mtl_buf, B.off * sizeof(T), size_t(1));
  objc::send_set_buffer(enc,
             OUT.mtl_buf, OUT.off * sizeof(T), size_t(2));
  objc::send_set_bytes(enc,
             &a_len, sizeof(uint32_t), size_t(3));
  objc::send_set_bytes(enc,
             &b_len, sizeof(uint32_t), size_t(4));
  objc::send_set_bytes(enc,
             &dtype, sizeof(uint32_t), size_t(5));
  objc::send_set_bytes(enc,
             &out_len, sizeof(uint32_t), size_t(6));
  objc::send_set_bytes(enc,
             &b_inv, sizeof(uint32_t), size_t(7));

  auto grid_len = std::is_same_v<T, float> ? (OUT.len + 3) / 4 : OUT.len;
  auto h = pl.max_threads / pl.thread_width;

  objc::send_dispatch(enc,
                      {grid_len, 1, 1},
                      {pl.thread_width, h, 1});
}

// Backward compatibility
using msl = gpu;
using mps = gpu;

inline void synchronize() {
  gpu_context::instance().flush();
}

};  // namespace sil
