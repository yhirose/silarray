// Helper utilities for ggml benchmarking — hides boilerplate.
#pragma once

#ifdef BENCH_HAS_GGML

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// Suppress all ggml log output.
inline void ggml_suppress_logs() {
  ggml_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);
}

// Redirect stderr to /dev/null to suppress pipeline compilation logs.
#include <unistd.h>
#include <fcntl.h>
struct GgmlQuiet {
  int saved_fd;
  GgmlQuiet() : saved_fd(dup(STDERR_FILENO)) {
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDERR_FILENO);
    close(devnull);
  }
  ~GgmlQuiet() {
    fflush(stderr);
    dup2(saved_fd, STDERR_FILENO);
    close(saved_fd);
  }
};

// Singleton ggml Metal backend — initialized once, reused across benchmarks.
inline ggml_backend_t ggml_metal_backend() {
  static ggml_backend_t backend = [] {
    ggml_suppress_logs();
    GgmlQuiet q;
    auto* b = ggml_backend_metal_init();
    if (!b) {
      fprintf(stdout, "ggml: Metal init failed\n");
      std::exit(1);
    }
    return b;
  }();
  return backend;
}

// RAII wrapper for a set of input tensors on the Metal backend.
struct GgmlInputs {
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;

  GgmlInputs(size_t n_tensors) {
    ggml_init_params p = {
      /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(p);
  }

  ~GgmlInputs() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
  }

  ggml_tensor* new_tensor_1d(size_t n) {
    return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
  }

  ggml_tensor* new_tensor_2d(size_t cols, size_t rows) {
    return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
  }

  void alloc_and_set(ggml_tensor* t, const float* data) {
    if (!buffer) {
      buffer = ggml_backend_alloc_ctx_tensors(ctx, ggml_metal_backend());
    }
    ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
  }
};

// Build and compute a single-op graph. ctx_graph is freed by caller.
// Pipeline compilation logs are suppressed via stderr redirect.
inline void ggml_compute_single(ggml_tensor* result, ggml_context* ctx_graph) {
  auto* gf = ggml_new_graph(ctx_graph);
  ggml_build_forward_expand(gf, result);
  auto* buf = ggml_backend_alloc_ctx_tensors(ctx_graph, ggml_metal_backend());
  {
    GgmlQuiet q;
    ggml_backend_graph_compute(ggml_metal_backend(), gf);
  }
  ggml_backend_buffer_free(buf);
}

// Create a graph context for a single operation.
inline ggml_context* ggml_graph_ctx(size_t n_ops = 4) {
  ggml_init_params p = {
    /*.mem_size   =*/ n_ops * ggml_tensor_overhead() + ggml_graph_overhead(),
    /*.mem_buffer =*/ nullptr,
    /*.no_alloc   =*/ true,
  };
  return ggml_init(p);
}


#else  // BENCH_HAS_GGML not defined

// Forward-declaration stubs so benchmark sources that reference ggml types
// unconditionally (e.g. `ggml_binary_op` aliases or struct fields) still
// compile when ggml isn't on the include path. All real ggml usage in those
// sources is guarded by `#ifdef BENCH_HAS_GGML` and stays dead here.
struct ggml_context;
struct ggml_tensor;

#endif // BENCH_HAS_GGML
