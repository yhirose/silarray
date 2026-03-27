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

// MLP inference: 768 -> 2048 -> sigmoid -> 768 -> sigmoid

void bench_mlp(std::vector<BenchGroup>& groups, bool csv) {
  size_t D = 768, H = 2048;

  // Shared sil lambda
  auto run_sil = [&](size_t batch, size_t iters, int trials) {
    auto x = sil::random({batch, D});
    auto W1 = sil::random({D, H});
    auto b1 = sil::zeros<float>({H});
    auto W2 = sil::random({H, D});
    auto b2 = sil::zeros<float>({D});
    sil::synchronize();
    return measure(iters, [&] {
      auto h1 = x.linear_sigmoid(W1, b1);
      auto out = h1.linear_sigmoid(W2, b2);
      sil::eval(out);
      sil::synchronize();
    }, 30, trials);
  };

  // --- GPU (sil-gpu, ggml, mlx, torch) ---
  if (!csv) print_section("mlp inference (GPU)");
  for (auto batch : {128ul, 256ul, 1024ul}) {
    size_t iters = batch <= 128 ? 100 : 20;
    int trials = 2;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil(batch, iters, trials)});

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(5);
      auto* gx = inputs.new_tensor_2d(D, batch);
      auto* gW1 = inputs.new_tensor_2d(D, H);
      auto* gb1 = inputs.new_tensor_1d(H);
      auto* gW2 = inputs.new_tensor_2d(H, D);
      auto* gb2 = inputs.new_tensor_1d(D);
      std::vector<float> ones_x(batch * D, 1.0f);
      std::vector<float> ones_W1(D * H, 1.0f);
      std::vector<float> zeros_b1(H, 0.0f);
      std::vector<float> ones_W2(H * D, 1.0f);
      std::vector<float> zeros_b2(D, 0.0f);
      inputs.alloc_and_set(gx, ones_x.data());
      inputs.alloc_and_set(gW1, ones_W1.data());
      inputs.alloc_and_set(gb1, zeros_b1.data());
      inputs.alloc_and_set(gW2, ones_W2.data());
      inputs.alloc_and_set(gb2, zeros_b2.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx(20);
        auto* h1 = ggml_sigmoid(ctx_g, ggml_add(ctx_g, ggml_mul_mat(ctx_g, gW1, gx), gb1));
        auto* out = ggml_sigmoid(ctx_g, ggml_add(ctx_g, ggml_mul_mat(ctx_g, gW2, h1), gb2));
        auto* gf = ggml_new_graph(ctx_g);
        ggml_build_forward_expand(gf, out);
        auto* buf = ggml_backend_alloc_ctx_tensors(ctx_g, ggml_metal_backend());
        {
          GgmlQuiet q;
          ggml_backend_graph_compute(ggml_metal_backend(), gf);
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_g);
      }, 30, trials)});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      int b = static_cast<int>(batch);
      int d = static_cast<int>(D), h = static_cast<int>(H);
      auto x = mx::random::normal({b, d});
      auto W1 = mx::random::normal({d, h});
      auto b1 = mx::zeros({h});
      auto W2 = mx::random::normal({h, d});
      auto b2 = mx::zeros({d});
      mx::eval(x, W1, b1, W2, b2);

      entries.push_back({"mlx", measure(iters, [&] {
                            auto h1 = mx::sigmoid(mx::addmm(b1, x, W1));
                            auto out = mx::sigmoid(mx::addmm(b2, h1, W2));
                            mx::eval(out);
                          }, 30, trials)});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      long b = batch, d = D, h = H;
      auto x = torch::randn({b, d}, dev);
      auto W1 = torch::randn({d, h}, dev);
      auto b1 = torch::zeros({h}, dev);
      auto W2 = torch::randn({h, d}, dev);
      auto b2 = torch::zeros({d}, dev);

      entries.push_back({"torch", measure(iters, [&] {
                            auto h1 = torch::sigmoid(torch::addmm(b1, x, W1));
                            auto out = torch::sigmoid(torch::addmm(b2, h1, W2));
                            torch::mps::synchronize();
                          }, 30, trials)});
    }
#endif

    auto group = BenchGroup{
        std::format("mlp inference (batch={})", batch), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  // --- CPU (sil-cpu, eigen) ---
  if (!csv) print_section("mlp inference (CPU)");
  sil::use_cpu();
  for (auto batch : {32ul, 64ul, 128ul}) {
    size_t iters = 100;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-cpu", run_sil(batch, iters, 5)});

#ifdef BENCH_HAS_EIGEN
    {
      auto sigmoid = [](const Eigen::MatrixXf& x) -> Eigen::MatrixXf {
        return (1.0f + (-x).array().exp()).inverse().matrix();
      };
      Eigen::MatrixXf ex = Eigen::MatrixXf::Random(batch, D);
      Eigen::MatrixXf eW1 = Eigen::MatrixXf::Random(D, H);
      Eigen::VectorXf eb1 = Eigen::VectorXf::Zero(H);
      Eigen::MatrixXf eW2 = Eigen::MatrixXf::Random(H, D);
      Eigen::VectorXf eb2 = Eigen::VectorXf::Zero(D);
      entries.push_back({"eigen", measure(iters, [&] {
                            auto h1 = sigmoid((ex * eW1).rowwise() + eb1.transpose());
                            Eigen::MatrixXf out = sigmoid((h1 * eW2).rowwise() + eb2.transpose());
                          }, 30, 2)});
    }
#endif

    auto group = BenchGroup{
        std::format("mlp inference (batch={})", batch), std::move(entries)};
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

  bench_mlp(groups, csv);
  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "MLP Inference", "2-layer MLP (768->2048->768, sigmoid): GPU (batch 128-1024) and CPU (batch 32-128)");
}
