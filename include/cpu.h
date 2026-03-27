#pragma once

#include <types.h>
#include <device.h>

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <cmath>
#include <numeric>

namespace sil {

class cpu {
 public:
  template <value_type T>
  static void add(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void sub(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void mul(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void div(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void pow(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void dot(const storage &A, const storage &B,
                  storage &OUT, uint32_t A_cols, uint32_t OUT_rows,
                  uint32_t OUT_cols);

  template <value_type T>
  static T sum(const T *data, size_t n);

  template <value_type T>
  static void sum_axis0(const T *src, T *dst, size_t rows, size_t cols);

  static void sigmoid(const float *src, float *dst, size_t n);
  static void sigmoid_backward(const float *dout, const float *x, float *dst, size_t n);
  static void bias_sigmoid(float *data, const float *bias, size_t n, size_t cols);
  static void relu(const float *src, float *dst, size_t n);

  static void layer_norm(const float *src, float *dst,
                         const float *gamma, const float *beta,
                         size_t rows, size_t cols, float eps);

  // out[i] = in[i] * scale + offset — single-pass NEON FMA
  static void affine(const float *in, float *out, size_t n,
                     float scale, float offset);

 private:
  template <value_type T>
  static const T *ptr(const storage &s) {
    return static_cast<const T *>(s.data) + s.off;
  }

  template <value_type T>
  static T *mutable_ptr(storage &s) {
    return static_cast<T *>(s.data) + s.off;
  }

};

//-----------------------------------------------------------------------------
// Implementation
//-----------------------------------------------------------------------------

template <value_type T>
inline void cpu::add(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      vDSP_vadd(a, 1, b, 1, out, 1, n);
      return;
    }
    if (B.len == 1) { vDSP_vsadd(a, 1, b, out, 1, n); return; }
    if (A.len == 1) { vDSP_vsadd(b, 1, a, out, 1, n); return; }
    // Broadcast: repeat shorter side row-by-row with vDSP
    if (A.len == n && B.len > 1 && n % B.len == 0) {
      for (size_t i = 0; i < n; i += B.len)
        vDSP_vadd(a + i, 1, b, 1, out + i, 1, B.len);
      return;
    }
    if (B.len == n && A.len > 1 && n % A.len == 0) {
      for (size_t i = 0; i < n; i += A.len)
        vDSP_vadd(a, 1, b + i, 1, out + i, 1, A.len);
      return;
    }
  }
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
  } else {
    for (size_t i = 0; i < n; i++) out[i] = a[i % A.len] + b[i % B.len];
  }
}

template <value_type T>
inline void cpu::sub(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      vDSP_vsub(b, 1, a, 1, out, 1, n);
      return;
    }
    if (B.len == 1) {
      float neg_b = -b[0];
      vDSP_vsadd(a, 1, &neg_b, out, 1, n);
      return;
    }
    if (A.len == n && B.len > 1 && n % B.len == 0) {
      for (size_t i = 0; i < n; i += B.len)
        vDSP_vsub(b, 1, a + i, 1, out + i, 1, B.len);
      return;
    }
    if (B.len == n && A.len > 1 && n % A.len == 0) {
      for (size_t i = 0; i < n; i += A.len)
        vDSP_vsub(b + i, 1, a, 1, out + i, 1, A.len);
      return;
    }
  }
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] - b[i];
  } else {
    for (size_t i = 0; i < n; i++) out[i] = a[i % A.len] - b[i % B.len];
  }
}

template <value_type T>
inline void cpu::mul(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      vDSP_vmul(a, 1, b, 1, out, 1, n);
      return;
    }
    if (B.len == 1) { vDSP_vsmul(a, 1, b, out, 1, n); return; }
    if (A.len == 1) { vDSP_vsmul(b, 1, a, out, 1, n); return; }
    if (A.len == n && B.len > 1 && n % B.len == 0) {
      for (size_t i = 0; i < n; i += B.len)
        vDSP_vmul(a + i, 1, b, 1, out + i, 1, B.len);
      return;
    }
    if (B.len == n && A.len > 1 && n % A.len == 0) {
      for (size_t i = 0; i < n; i += A.len)
        vDSP_vmul(a, 1, b + i, 1, out + i, 1, A.len);
      return;
    }
  }
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
  } else {
    for (size_t i = 0; i < n; i++) out[i] = a[i % A.len] * b[i % B.len];
  }
}

template <value_type T>
inline void cpu::div(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      vDSP_vdiv(b, 1, a, 1, out, 1, n);
      return;
    }
    if (B.len == 1) { vDSP_vsdiv(a, 1, b, out, 1, n); return; }
    if (A.len == 1) { vDSP_svdiv(a, b, 1, out, 1, n); return; }
    if (A.len == n && B.len > 1 && n % B.len == 0) {
      for (size_t i = 0; i < n; i += B.len)
        vDSP_vdiv(b, 1, a + i, 1, out + i, 1, B.len);
      return;
    }
    if (B.len == n && A.len > 1 && n % A.len == 0) {
      for (size_t i = 0; i < n; i += A.len)
        vDSP_vdiv(b + i, 1, a, 1, out + i, 1, A.len);
      return;
    }
  }
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] / b[i];
  } else {
    for (size_t i = 0; i < n; i++) out[i] = a[i % A.len] / b[i % B.len];
  }
}

template <value_type T>
inline void cpu::pow(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      int n_int = static_cast<int>(n);
      vvpowf(out, b, a, &n_int);
      return;
    }
  }

  // Explicit cast: std::pow returns double, and the runtime dispatcher can
  // instantiate cpu::pow<bool> via apply_binary_operation_<bool> on operator==
  // even though Pow is never reached on bool at runtime.
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = static_cast<T>(std::pow(a[i], b[i]));
  } else {
    for (size_t i = 0; i < n; i++)
      out[i] = static_cast<T>(std::pow(a[i % A.len], b[i % B.len]));
  }
}

template <value_type T>
inline void cpu::dot(const storage &A, const storage &B,
                      storage &OUT, uint32_t A_cols, uint32_t OUT_rows,
                      uint32_t OUT_cols) {
  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, OUT_rows, OUT_cols,
                A_cols, 1.0f, ptr<T>(A), A_cols, ptr<T>(B), OUT_cols, 0.0f,
                mutable_ptr<T>(OUT), OUT_cols);
    return;
  }

  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  std::memset(out, 0, OUT_rows * OUT_cols * sizeof(T));

  // i,k,j loop order for cache-friendly access to B
  for (uint32_t i = 0; i < OUT_rows; i++) {
    for (uint32_t k = 0; k < A_cols; k++) {
      auto a_ik = a[i * A_cols + k];
      for (uint32_t j = 0; j < OUT_cols; j++) {
        out[i * OUT_cols + j] += a_ik * b[k * OUT_cols + j];
      }
    }
  }
}

template <value_type T>
inline T cpu::sum(const T *data, size_t n) {
  if constexpr (std::is_same_v<T, float>) {
    float result;
    vDSP_sve(data, 1, &result, n);
    return result;
  }
  return std::accumulate(data, data + n, T{});
}

template <value_type T>
inline void cpu::sum_axis0(const T *src, T *dst, size_t rows, size_t cols) {
  if constexpr (std::is_same_v<T, float>) {
    std::memset(dst, 0, cols * sizeof(float));
    for (size_t r = 0; r < rows; r++)
      vDSP_vadd(dst, 1, src + r * cols, 1, dst, 1, cols);
  } else {
    std::memset(dst, 0, cols * sizeof(T));
    for (size_t r = 0; r < rows; r++)
      for (size_t c = 0; c < cols; c++)
        dst[c] += src[r * cols + c];
  }
}

inline void cpu::sigmoid(const float *src, float *dst, size_t n) {
  auto len = static_cast<int>(n);
  vDSP_vneg(src, 1, dst, 1, n);
  vvexpf(dst, dst, &len);
  float one = 1.0f;
  vDSP_vsadd(dst, 1, &one, dst, 1, n);
  vvrecf(dst, dst, &len);
}

inline void cpu::sigmoid_backward(const float *dout, const float *x,
                                   float *dst, size_t n) {
  // dst = sigmoid(x) via vectorized vDSP
  sigmoid(x, dst, n);
  // dst = dout * sigmoid * (1 - sigmoid) — compiler auto-vectorizes with NEON
  for (size_t i = 0; i < n; i++) {
    auto s = dst[i];
    dst[i] = dout[i] * s * (1.0f - s);
  }
}

inline void cpu::bias_sigmoid(float *data, const float *bias,
                               size_t n, size_t cols) {
  // Add bias (broadcast row-wise)
  for (size_t i = 0; i < n; i += cols)
    vDSP_vadd(data + i, 1, bias, 1, data + i, 1, cols);
  // In-place sigmoid
  auto len = static_cast<int>(n);
  vDSP_vneg(data, 1, data, 1, n);
  vvexpf(data, data, &len);
  float one = 1.0f;
  vDSP_vsadd(data, 1, &one, data, 1, n);
  vvrecf(data, data, &len);
}

inline void cpu::relu(const float *src, float *dst, size_t n) {
  float zero = 0.0f;
  vDSP_vthres(src, 1, &zero, dst, 1, n);
}

inline void cpu::affine(const float *in, float *out, size_t n,
                        float scale, float offset) {
  float32x4_t vs = vdupq_n_f32(scale);
  float32x4_t vo = vdupq_n_f32(offset);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t v0 = vld1q_f32(in + i);
    float32x4_t v1 = vld1q_f32(in + i + 4);
    float32x4_t v2 = vld1q_f32(in + i + 8);
    float32x4_t v3 = vld1q_f32(in + i + 12);
    vst1q_f32(out + i,      vfmaq_f32(vo, v0, vs));
    vst1q_f32(out + i + 4,  vfmaq_f32(vo, v1, vs));
    vst1q_f32(out + i + 8,  vfmaq_f32(vo, v2, vs));
    vst1q_f32(out + i + 12, vfmaq_f32(vo, v3, vs));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(in + i);
    vst1q_f32(out + i, vfmaq_f32(vo, v, vs));
  }
  for (; i < n; i++)
    out[i] = in[i] * scale + offset;
}

inline void cpu::layer_norm(const float *src, float *dst,
                            const float *gamma, const float *beta,
                            size_t rows, size_t cols, float eps) {
  for (size_t r = 0; r < rows; r++) {
    const float *row = src + r * cols;
    float *out = dst + r * cols;

    float mu;
    vDSP_meanv(row, 1, &mu, cols);

    float neg_mu = -mu;
    vDSP_vsadd(row, 1, &neg_mu, out, 1, cols);    // out = row - mu

    float sum_sq;
    vDSP_dotpr(out, 1, out, 1, &sum_sq, cols);    // sum_sq = sum((row - mu)^2)
    float inv_std = 1.0f / sqrtf(sum_sq / cols + eps);

    vDSP_vsmul(out, 1, &inv_std, out, 1, cols);   // out *= inv_std
    vDSP_vmul(out, 1, gamma, 1, out, 1, cols);    // out *= gamma
    vDSP_vadd(out, 1, beta, 1, out, 1, cols);     // out += beta
  }
}

};  // namespace sil
