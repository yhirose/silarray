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

void bench_broadcast(std::vector<BenchGroup>& groups, bool csv) {
  struct Size { size_t rows, cols; };

  // Shared sil lambda
  auto run_sil = [](size_t rows, size_t cols, size_t iters) {
    auto a = sil::ones<float>({rows, cols});
    auto b = sil::ones<float>({cols});
    sil::synchronize();
    auto c = sil::array<float>();
    return measure(iters, [&] { c = a + b; sil::eval(c); sil::synchronize(); });
  };

  // --- GPU (sil-gpu, ggml, mlx, torch) ---
  if (!csv) print_section("broadcast (GPU)");
  for (auto [rows, cols] :
       std::initializer_list<Size>{{1024, 1024}, {4096, 512}, {4096, 4096}}) {
    size_t iters = rows * cols <= 1'000'000 ? 200 : 30;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil(rows, cols, iters)});

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(2);
      auto* ga = inputs.new_tensor_2d(cols, rows);
      auto* gb = inputs.new_tensor_1d(cols);
      std::vector<float> ones_mat(rows * cols, 1.0f);
      std::vector<float> ones_vec(cols, 1.0f);
      inputs.alloc_and_set(ga, ones_mat.data());
      inputs.alloc_and_set(gb, ones_vec.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_add(ctx_g, ga, gb);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      int r = static_cast<int>(rows), c = static_cast<int>(cols);
      auto ma = mx::ones({r, c});
      auto mb = mx::ones({c});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back({"mlx", measure(iters, [&] {
                            mc = mx::add(ma, mb);
                            mx::eval(mc);
                          })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      long r = rows, c = cols;
      auto a = torch::ones({r, c}, dev);
      auto b = torch::ones({c}, dev);
      auto tc = torch::Tensor();
      entries.push_back(
          {"torch", measure(iters, [&] {
             tc = torch::add(a, b);
             torch::mps::synchronize();
           })});
    }
#endif

    auto group = BenchGroup{
        std::format("broadcast ({}x{})+({})", rows, cols, cols),
        std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  // --- CPU (sil-cpu, eigen) ---
  if (!csv) print_section("broadcast (CPU)");
  sil::use_cpu();
  for (auto [rows, cols] :
       std::initializer_list<Size>{{256, 256}, {1024, 256}, {1024, 1024}}) {
    size_t iters = rows * cols <= 100'000 ? 500 : 200;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-cpu", run_sil(rows, cols, iters)});

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::MatrixXf aa = Eigen::MatrixXf::Ones(rows, cols);
      Eigen::VectorXf bb = Eigen::VectorXf::Ones(cols);
      Eigen::MatrixXf cc(rows, cols);
      entries.push_back(
          {"eigen",
           measure(iters, [&] { cc = aa.rowwise() + bb.transpose(); })});
    }
#endif

    auto group = BenchGroup{
        std::format("broadcast ({}x{})+({})", rows, cols, cols),
        std::move(entries)};
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

  bench_broadcast(groups, csv);
  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "Broadcast", "Bias-add broadcast `(N,M) + (M)`: GPU (1024-4096) and CPU (256-1024)");
}
