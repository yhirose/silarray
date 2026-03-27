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

#ifdef BENCH_HAS_GGML
#include "../bench_ggml.h"
#endif

void bench_sgemm(std::vector<BenchGroup>& groups, bool csv) {
  // Shared sil lambda — sil::synchronize() is a no-op on CPU
  auto run_sil = [](size_t m, size_t iters, int trials) {
    auto a = sil::ones<float>({m, m});
    auto b = sil::ones<float>({m, m});
    sil::synchronize();
    auto c = sil::array<float>();
    return measure(iters, [&] { c = a.dot(b); sil::eval(c); sil::synchronize(); }, 30, trials);
  };

  // --- GPU (sil-gpu, ggml, mlx, torch) ---
  if (!csv) print_section("sgemm GPU (square)");
  for (auto m : {1024ul, 2048ul, 4096ul, 8192ul}) {
    size_t iters = m <= 1024 ? 30 : m <= 4096 ? 10 : 5;
    int trials = m >= 4096 ? 2 : 3;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil(m, iters, trials)});

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(2);
      auto* ga = inputs.new_tensor_2d(m, m);
      auto* gb_t = inputs.new_tensor_2d(m, m);
      std::vector<float> ones(m * m, 1.0f);
      inputs.alloc_and_set(ga, ones.data());
      inputs.alloc_and_set(gb_t, ones.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_mul_mat(ctx_g, ga, gb_t);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      }, 30, trials)});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      int mi = static_cast<int>(m);
      auto ma = mx::ones({mi, mi});
      auto mb = mx::ones({mi, mi});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back(
          {"mlx", measure(iters, [&] {
             mc = mx::matmul(ma, mb);
             mx::eval(mc);
           }, 30, trials)});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto a =
          torch::ones({static_cast<long>(m), static_cast<long>(m)}, dev);
      auto b =
          torch::ones({static_cast<long>(m), static_cast<long>(m)}, dev);
      auto c = torch::Tensor();
      entries.push_back({"torch", measure(iters, [&] {
                            c = torch::mm(a, b);
                            torch::mps::synchronize();
                          }, 30, trials)});
    }
#endif

    auto best = std::ranges::min_element(entries, {}, &BenchEntry::seconds)
                    ->seconds;
    auto group = BenchGroup{
        std::format("sgemm {}x{} ({:.1f} GFLOPS)", m, m,
                     gflops_gemm(m, m, m, best)),
        std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  // --- CPU (sil-cpu, eigen) ---
  if (!csv) print_section("sgemm CPU (square)");
  sil::use_cpu();
  for (auto m : {128ul, 256ul, 512ul, 1024ul}) {
    size_t iters = m <= 256 ? 200 : m <= 512 ? 100 : 30;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-cpu", run_sil(m, iters, 5)});

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::MatrixXf aa = Eigen::MatrixXf::Ones(m, m);
      Eigen::MatrixXf bb = Eigen::MatrixXf::Ones(m, m);
      Eigen::MatrixXf cc;
      entries.push_back({"eigen", measure(iters, [&] { cc = aa * bb; }, 30, 2)});
    }
#endif

    auto best = std::ranges::min_element(entries, {}, &BenchEntry::seconds)
                    ->seconds;
    auto group = BenchGroup{
        std::format("sgemm {}x{} ({:.1f} GFLOPS)", m, m,
                     gflops_gemm(m, m, m, best)),
        std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
  sil::use_mps();
}

// Non-square matmul — GPU only (DL-scale shapes)
void bench_sgemm_rect(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("sgemm GPU (non-square)");

  struct { size_t M, N, K; const char* desc; } shapes[] = {
    {1,    4096, 4096, "single-vector inference"},
    {32,   4096, 768,  "small-batch embedding"},
    {256,  4096, 768,  "medium-batch projection"},
    {1024, 4096, 768,  "large-batch projection"},
    {2048, 768,  4096, "FFN down-projection"},
  };

  for (auto& [M, N, K, desc] : shapes) {
    size_t iters = (M * N * K <= 1'000'000) ? 500
                 : (M * N * K <= 100'000'000) ? 100 : 20;
    int trials = 2;

    std::vector<BenchEntry> entries;

    {
      auto a = sil::ones<float>({M, K});
      auto b = sil::ones<float>({K, N});
      auto c = sil::array<float>();
      sil::synchronize();
      entries.push_back({"sil-gpu", measure(iters, [&] {
        c = a.dot(b); sil::eval(c); sil::synchronize();
      }, 30, trials)});
    }

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(2);
      auto* ga = inputs.new_tensor_2d(K, N);
      auto* gb_t = inputs.new_tensor_2d(K, M);
      std::vector<float> ones_a(N * K, 1.0f);
      std::vector<float> ones_b(M * K, 1.0f);
      inputs.alloc_and_set(ga, ones_a.data());
      inputs.alloc_and_set(gb_t, ones_b.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_mul_mat(ctx_g, ga, gb_t);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      }, 30, trials)});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto ma = mx::ones({(int)M, (int)K});
      auto mb = mx::ones({(int)K, (int)N});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back({"mlx", measure(iters, [&] { mc = mx::matmul(ma, mb); mx::eval(mc); }, 30, trials)});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto a = torch::ones({(long)M, (long)K}, dev);
      auto b = torch::ones({(long)K, (long)N}, dev);
      auto c = torch::Tensor();
      entries.push_back({"torch", measure(iters, [&] { c = torch::mm(a, b); torch::mps::synchronize(); }, 30, trials)});
    }
#endif

    auto best = std::ranges::min_element(entries, {}, &BenchEntry::seconds)->seconds;
    auto group = BenchGroup{
        std::format("{}x{}x{} ({}, {:.1f} GFLOPS)", M, N, K, desc,
                     gflops_gemm(M, N, K, best)),
        std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  auto mode = parse_output_mode(argc, argv);
  bool csv = (mode != OutputMode::bar);
  std::vector<BenchGroup> groups;

#ifdef BENCH_HAS_GGML
  ggml_metal_backend();
#endif

  bench_sgemm(groups, csv);
  bench_sgemm_rect(groups, csv);
  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "SGEMM", "Single-precision matrix multiplication: GPU (1024-8192) and CPU (128-1024) with non-square DL shapes");
}
