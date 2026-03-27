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

// Softmax — GPU only (CPU is 100x+ slower)
void bench_softmax(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("softmax (GPU)");
  for (auto [rows, cols] : {std::pair{256ul, 512ul}, {1024ul, 1024ul}, {4096ul, 2048ul}, {256ul, 8192ul}}) {
    size_t iters = (rows * cols <= 1'000'000) ? 200 : 50;
    std::vector<BenchEntry> entries;

    {
      auto x = sil::random({rows, cols});
      auto y = sil::array<float>();
      sil::synchronize();
      entries.push_back({"sil-gpu", measure(iters, [&] {
        y = x.softmax(); sil::eval(y); sil::synchronize();
      })});
    }

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(1);
      auto* ga = inputs.new_tensor_2d(cols, rows);
      std::vector<float> ones(rows * cols, 1.0f);
      inputs.alloc_and_set(ga, ones.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_soft_max(ctx_g, ga);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto x = mx::random::normal({(int)rows, (int)cols});
      mx::eval(x);
      auto y = mx::array(0.f);
      entries.push_back({"mlx", measure(iters, [&] { y = mx::softmax(x, -1); mx::eval(y); })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto x = torch::randn({(long)rows, (long)cols}, torch::kMPS);
      entries.push_back({"torch", measure(iters, [&] {
        auto y = torch::softmax(x, -1);
        torch::mps::synchronize();
      })});
    }
#endif

    auto group = BenchGroup{
        std::format("softmax ({}x{})", rows, cols), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

void bench_layernorm(std::vector<BenchGroup>& groups, bool csv) {
  // Shared sil lambda
  auto run_sil = [](size_t rows, size_t cols, size_t iters) {
    auto x = sil::random({rows, cols});
    auto gamma = sil::ones<float>({cols});
    auto beta = sil::zeros<float>({cols});
    sil::synchronize();
    auto y = sil::array<float>();
    return measure(iters, [&] { y = x.layer_norm(gamma, beta); sil::eval(y); sil::synchronize(); });
  };

  // --- GPU (sil-gpu, ggml, mlx, torch) ---
  if (!csv) print_section("layer norm (GPU)");
  for (auto [rows, cols] : {std::pair{256ul, 512ul}, {1024ul, 1024ul}, {4096ul, 2048ul}}) {
    size_t iters = (rows * cols <= 1'000'000) ? 200 : 50;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil(rows, cols, iters)});

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(1);
      auto* ga = inputs.new_tensor_2d(cols, rows);
      std::vector<float> ones(rows * cols, 1.0f);
      inputs.alloc_and_set(ga, ones.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_norm(ctx_g, ga, 1e-5f);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto x = mx::random::normal({(int)rows, (int)cols});
      auto gamma = mx::ones({(int)cols});
      auto beta = mx::zeros({(int)cols});
      mx::eval(x, gamma, beta);
      auto y = mx::array(0.f);
      entries.push_back({"mlx", measure(iters, [&] {
        auto mean = mx::mean(x, -1, true);
        auto var = mx::var(x, -1, true);
        y = (x - mean) * mx::rsqrt(var + 1e-5f) * gamma + beta;
        mx::eval(y);
      })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto x = torch::randn({(long)rows, (long)cols}, torch::kMPS);
      auto ln = torch::nn::LayerNorm(torch::nn::LayerNormOptions({(long)cols}));
      ln->to(torch::kMPS);
      entries.push_back({"torch", measure(iters, [&] {
        auto y = ln->forward(x);
        torch::mps::synchronize();
      })});
    }
#endif

    auto group = BenchGroup{
        std::format("layer_norm ({}x{})", rows, cols), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  // --- CPU (sil-cpu, eigen) ---
  if (!csv) print_section("layer norm (CPU)");
  sil::use_cpu();
  for (auto [rows, cols] : {std::pair{64ul, 256ul}, {256ul, 512ul}}) {
    size_t iters = 500;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-cpu", run_sil(rows, cols, iters)});

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::MatrixXf x = Eigen::MatrixXf::Random(rows, cols);
      Eigen::VectorXf gamma = Eigen::VectorXf::Ones(cols);
      Eigen::VectorXf beta = Eigen::VectorXf::Zero(cols);
      Eigen::MatrixXf y(rows, cols);
      entries.push_back({"eigen", measure(iters, [&] {
        Eigen::VectorXf mean = x.rowwise().mean();
        Eigen::MatrixXf centered = x.colwise() - mean;
        Eigen::VectorXf var = centered.array().square().rowwise().mean();
        Eigen::VectorXf inv_std = (var.array() + 1e-5f).rsqrt();
        y = ((centered.array().colwise() * inv_std.array()).rowwise() * gamma.array().transpose()).rowwise()
            + beta.array().transpose();
      })});
    }
#endif

    auto group = BenchGroup{
        std::format("layer_norm ({}x{})", rows, cols), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
  sil::use_mps();
}

// Conv2d — GPU only (sil and Eigen lack native conv2d)
void bench_conv2d(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("conv2d (GPU)");

  struct ConvConfig {
    long batch, in_ch, out_ch, h, w, k;
    const char* desc;
  };

  ConvConfig configs[] = {
    {1,  3,   32,  224, 224, 3, "ImageNet first layer"},
    {16, 64,  128, 56,  56,  3, "ResNet mid layer"},
    {16, 128, 256, 28,  28,  3, "ResNet deep layer"},
  };

  for (auto& [batch, in_ch, out_ch, h, w, k, desc] : configs) {
    size_t iters = (h >= 224) ? 10 : 20;
    std::vector<BenchEntry> entries;

    {
      auto x = sil::random({(size_t)batch, (size_t)h, (size_t)w, (size_t)in_ch});
      auto weight = sil::random({(size_t)out_ch, (size_t)k, (size_t)k, (size_t)in_ch});
      sil::synchronize();
      auto y = sil::array<float>();
      entries.push_back({"sil-gpu", measure(iters, [&] {
        y = x.conv2d(weight, k);
        sil::eval(y); sil::synchronize();
      }, 30, 2)});
    }

#ifdef BENCH_HAS_MLX
    {
      auto x = mx::random::normal({(int)batch, (int)h, (int)w, (int)in_ch});
      auto weight = mx::random::normal({(int)out_ch, (int)k, (int)k, (int)in_ch});
      mx::eval(x, weight);
      auto y = mx::array(0.f);
      entries.push_back({"mlx", measure(iters, [&] {
        y = mx::conv2d(x, weight, {1, 1}, {1, 1});
        mx::eval(y);
      }, 30, 2)});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto x = torch::randn({batch, in_ch, h, w}, torch::kMPS);
      auto conv = torch::nn::Conv2d(
          torch::nn::Conv2dOptions(in_ch, out_ch, k).padding(k / 2).bias(false));
      conv->to(torch::kMPS);
      entries.push_back({"torch", measure(iters, [&] {
        auto y = conv->forward(x);
        torch::mps::synchronize();
      }, 30, 2)});
    }
#endif

    auto group = BenchGroup{
        std::format("conv2d {} ({}x{}x{}x{}, k={})", desc, batch, in_ch, h, w, k),
        std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

void bench_batch_matmul(std::vector<BenchGroup>& groups, bool csv) {
  struct BMConfig {
    long batch, M, N, K;
    const char* desc;
  };

  // Shared sil lambda
  auto run_sil = [](long batch, long M, long N, long K, size_t iters) {
    auto a = sil::random({(size_t)batch, (size_t)M, (size_t)K});
    auto b = sil::random({(size_t)batch, (size_t)K, (size_t)N});
    sil::synchronize();
    auto c = sil::array<float>();
    return measure(iters, [&] { c = a.batched_dot(b); sil::eval(c); sil::synchronize(); }, 30, 2);
  };

  // --- GPU (sil-gpu, ggml, mlx, torch) ---
  if (!csv) print_section("batch matmul (GPU)");

  BMConfig gpu_configs[] = {
    {8,  128, 128, 64,  "attention (8h, seq=128, d=64)"},
    {8,  512, 512, 64,  "attention (8h, seq=512, d=64)"},
    {16, 256, 256, 128, "attention (16h, seq=256, d=128)"},
  };

  for (auto& [batch, M, N, K, desc] : gpu_configs) {
    size_t iters = 50;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil(batch, M, N, K, iters)});

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(batch * 2);
      std::vector<ggml_tensor*> gas(batch), gbs(batch);
      for (long i = 0; i < batch; i++) {
        gas[i] = inputs.new_tensor_2d(K, M);
        gbs[i] = inputs.new_tensor_2d(K, N);
      }
      std::vector<float> ones_a(M * K, 1.0f), ones_b(K * N, 1.0f);
      for (long i = 0; i < batch; i++) {
        inputs.alloc_and_set(gas[i], ones_a.data());
        inputs.alloc_and_set(gbs[i], ones_b.data());
      }

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx(batch * 2);
        ggml_tensor* last = nullptr;
        for (long i = 0; i < batch; i++) {
          last = ggml_mul_mat(ctx_g, gas[i], gbs[i]);
        }
        auto* gf = ggml_new_graph(ctx_g);
        ggml_build_forward_expand(gf, last);
        auto* buf = ggml_backend_alloc_ctx_tensors(ctx_g, ggml_metal_backend());
        {
          GgmlQuiet q;
          ggml_backend_graph_compute(ggml_metal_backend(), gf);
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_g);
      }, 30, 2)});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto a = mx::random::normal({(int)batch, (int)M, (int)K});
      auto b = mx::random::normal({(int)batch, (int)K, (int)N});
      mx::eval(a, b);
      auto c = mx::array(0.f);
      entries.push_back({"mlx", measure(iters, [&] {
        c = mx::matmul(a, b);
        mx::eval(c);
      }, 30, 2)});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto a = torch::randn({batch, M, K}, torch::kMPS);
      auto b = torch::randn({batch, K, N}, torch::kMPS);
      entries.push_back({"torch", measure(iters, [&] {
        auto c = torch::bmm(a, b);
        torch::mps::synchronize();
      }, 30, 2)});
    }
#endif

    auto group = BenchGroup{std::format("bmm {}", desc), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  // --- CPU (sil-cpu, eigen) ---
  if (!csv) print_section("batch matmul (CPU)");

  BMConfig cpu_configs[] = {
    {4,  64,  64,  32,  "attention (4h, seq=64, d=32)"},
    {8,  128, 128, 64,  "attention (8h, seq=128, d=64)"},
  };

  sil::use_cpu();
  for (auto& [batch, M, N, K, desc] : cpu_configs) {
    size_t iters = 100;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-cpu", run_sil(batch, M, N, K, iters)});

#ifdef BENCH_HAS_EIGEN
    {
      std::vector<Eigen::MatrixXf> as(batch), bs(batch), cs(batch);
      for (long i = 0; i < batch; i++) {
        as[i] = Eigen::MatrixXf::Random(M, K);
        bs[i] = Eigen::MatrixXf::Random(K, N);
        cs[i].resize(M, N);
      }
      entries.push_back({"eigen", measure(iters, [&] {
        for (long i = 0; i < batch; i++)
          cs[i].noalias() = as[i] * bs[i];
      }, 30, 2)});
    }
#endif

    auto group = BenchGroup{std::format("bmm {}", desc), std::move(entries)};
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

  bench_softmax(groups, csv);
  bench_layernorm(groups, csv);
  bench_conv2d(groups, csv);
  bench_batch_matmul(groups, csv);
  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "NN Ops", "Neural network primitives: GPU (softmax, layer norm, conv2d, bmm) and CPU (layer norm, bmm)");
}
