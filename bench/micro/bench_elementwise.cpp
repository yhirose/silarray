#include <silarray.h>

#include "../bench_common.h"

#ifdef BENCH_HAS_EIGEN
#include <eigen3/Eigen/Core>
#endif

#ifdef BENCH_HAS_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

#ifdef BENCH_HAS_LIBTORCH
#include <torch/torch.h>
#endif

// Unconditional include: bench_ggml.h provides forward-declaration stubs for
// ggml_context/ggml_tensor when BENCH_HAS_GGML is undefined, so code below
// that references these types unconditionally still compiles.
#include "../bench_ggml.h"

using ggml_binary_op = ggml_tensor* (*)(ggml_context*, ggml_tensor*, ggml_tensor*);

// GPU benchmarks for a binary elementwise op
void bench_op_gpu(std::vector<BenchGroup>& groups, bool csv, const char* op_name,
                  auto sil_fn, auto mlx_fn, auto torch_fn,
                  ggml_binary_op ggml_fn) {
  for (auto n : {1'000'000ul, 10'000'000ul}) {
    size_t iters = n <= 1'000'000 ? 200 : 50;
    std::vector<BenchEntry> entries;

    {
      auto a = sil::ones<float>({n});
      auto b = sil::ones<float>({n});
      auto c = sil::array<float>();
      sil::synchronize();
      entries.push_back({"sil-gpu", measure(iters, [&] {
        c = sil_fn(a, b); sil::eval(c); sil::synchronize();
      })});
    }

#ifdef BENCH_HAS_GGML
    if (ggml_fn) {
      GgmlInputs inputs(2);
      auto* ga = inputs.new_tensor_1d(n);
      auto* gb = inputs.new_tensor_1d(n);
      std::vector<float> ones(n, 1.0f);
      inputs.alloc_and_set(ga, ones.data());
      inputs.alloc_and_set(gb, ones.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_fn(ctx_g, ga, gb);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto ma = mx::ones({static_cast<int>(n)});
      auto mb = mx::ones({static_cast<int>(n)});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back({"mlx", measure(iters, [&] {
                            mc = mlx_fn(ma, mb);
                            mx::eval(mc);
                          })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto a = torch::ones({static_cast<long>(n)}, dev);
      auto b = torch::ones({static_cast<long>(n)}, dev);
      auto c = torch::Tensor();
      entries.push_back(
          {"torch", measure(iters, [&] {
             c = torch_fn(a, b);
             torch::mps::synchronize();
           })});
    }
#endif

    auto group = BenchGroup{std::format("{} ({})", op_name, n), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

// CPU benchmarks for a binary elementwise op
void bench_op_cpu(std::vector<BenchGroup>& groups, bool csv, const char* op_name,
                  auto sil_fn, auto eigen_fn) {
  sil::use_cpu();
  for (auto n : {100'000ul, 1'000'000ul}) {
    size_t iters = n <= 100'000 ? 500 : 200;
    std::vector<BenchEntry> entries;

    {
      auto a = sil::ones<float>({n});
      auto b = sil::ones<float>({n});
      auto c = sil::array<float>();
      entries.push_back({"sil-cpu", measure(iters, [&] { c = sil_fn(a, b); })});
    }

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::VectorXf aa = Eigen::VectorXf::Ones(n);
      Eigen::VectorXf bb = Eigen::VectorXf::Ones(n);
      Eigen::VectorXf cc(n);
      entries.push_back(
          {"eigen", measure(iters, [&] { eigen_fn(cc, aa, bb); })});
    }
#endif

    auto group = BenchGroup{std::format("{} ({})", op_name, n), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
  sil::use_mps();
}

int main(int argc, const char** argv) {
  auto mode = parse_output_mode(argc, argv);
  bool csv = (mode != OutputMode::bar);
  std::vector<BenchGroup> groups;

#ifdef BENCH_HAS_GGML
  ggml_metal_backend();
#endif

  struct {
    const char* name;
    std::function<sil::array<float>(const sil::array<float>&, const sil::array<float>&)> sil_fn;
#ifdef BENCH_HAS_EIGEN
    std::function<void(Eigen::VectorXf&, const Eigen::VectorXf&, const Eigen::VectorXf&)> eigen_fn;
#endif
#ifdef BENCH_HAS_MLX
    std::function<mx::array(const mx::array&, const mx::array&)> mlx_fn;
#endif
#ifdef BENCH_HAS_LIBTORCH
    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> torch_fn;
#endif
    ggml_binary_op ggml_fn;
  } ops[] = {
    // ... too complex with conditional fields
  };

  // Simpler: just call each op inline

  // --- add ---
  if (!csv) print_section("add (GPU)");
  bench_op_gpu(groups, csv, "add",
    [](auto& a, auto& b) { return a + b; },
#ifdef BENCH_HAS_MLX
    [](auto& a, auto& b) { return mx::add(a, b); },
#else
    [](auto&, auto&) { return 0; },
#endif
#ifdef BENCH_HAS_LIBTORCH
    [](auto& a, auto& b) { return torch::add(a, b); },
#else
    [](auto&, auto&) { return 0; },
#endif
#ifdef BENCH_HAS_GGML
    ggml_add
#else
    nullptr
#endif
  );

  if (!csv) print_section("add (CPU)");
  bench_op_cpu(groups, csv, "add",
    [](auto& a, auto& b) { return a + b; },
#ifdef BENCH_HAS_EIGEN
    [](auto& c, auto& a, auto& b) { c = a + b; }
#else
    [](auto&, auto&, auto&) {}
#endif
  );

  // --- mul ---
  if (!csv) print_section("mul (GPU)");
  bench_op_gpu(groups, csv, "mul",
    [](auto& a, auto& b) { return a * b; },
#ifdef BENCH_HAS_MLX
    [](auto& a, auto& b) { return mx::multiply(a, b); },
#else
    [](auto&, auto&) { return 0; },
#endif
#ifdef BENCH_HAS_LIBTORCH
    [](auto& a, auto& b) { return torch::mul(a, b); },
#else
    [](auto&, auto&) { return 0; },
#endif
#ifdef BENCH_HAS_GGML
    ggml_mul
#else
    nullptr
#endif
  );

  if (!csv) print_section("mul (CPU)");
  bench_op_cpu(groups, csv, "mul",
    [](auto& a, auto& b) { return a * b; },
#ifdef BENCH_HAS_EIGEN
    [](auto& c, auto& a, auto& b) { c = a.cwiseProduct(b); }
#else
    [](auto&, auto&, auto&) {}
#endif
  );

  // --- div ---
  if (!csv) print_section("div (GPU)");
  bench_op_gpu(groups, csv, "div",
    [](auto& a, auto& b) { return a / b; },
#ifdef BENCH_HAS_MLX
    [](auto& a, auto& b) { return mx::divide(a, b); },
#else
    [](auto&, auto&) { return 0; },
#endif
#ifdef BENCH_HAS_LIBTORCH
    [](auto& a, auto& b) { return torch::div(a, b); },
#else
    [](auto&, auto&) { return 0; },
#endif
#ifdef BENCH_HAS_GGML
    ggml_div
#else
    nullptr
#endif
  );

  if (!csv) print_section("div (CPU)");
  bench_op_cpu(groups, csv, "div",
    [](auto& a, auto& b) { return a / b; },
#ifdef BENCH_HAS_EIGEN
    [](auto& c, auto& a, auto& b) { c = a.cwiseQuotient(b); }
#else
    [](auto&, auto&, auto&) {}
#endif
  );

  // --- pow (GPU only) ---
  if (!csv) print_section("pow (GPU)");
  for (auto n : {1'000'000ul, 10'000'000ul}) {
    size_t iters = n <= 1'000'000 ? 200 : 50;
    std::vector<BenchEntry> entries;

    {
      auto a = sil::ones<float>({n});
      auto b = sil::ones<float>({n});
      auto c = sil::array<float>();
      sil::synchronize();
      entries.push_back({"sil-gpu", measure(iters, [&] { c = a.pow(b); sil::eval(c); sil::synchronize(); })});
    }

#ifdef BENCH_HAS_MLX
    {
      auto ma = mx::ones({static_cast<int>(n)});
      auto mb = mx::ones({static_cast<int>(n)});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back({"mlx", measure(iters, [&] {
                            mc = mx::power(ma, mb);
                            mx::eval(mc);
                          })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto a = torch::ones({static_cast<long>(n)}, dev);
      auto b = torch::ones({static_cast<long>(n)}, dev);
      auto c = torch::Tensor();
      entries.push_back(
          {"torch", measure(iters, [&] {
             c = torch::pow(a, b);
             torch::mps::synchronize();
           })});
    }
#endif

    auto group = BenchGroup{std::format("pow ({})", n), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "Elementwise", "Per-element vector operations: GPU (1M-10M) and CPU (100K-1M)");
}
