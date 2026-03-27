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

// Training step: forward + MSE loss + manual backward + SGD update.
// Network: 768 -> 2048 -> 768 (Transformer FFN scale, 2-layer with sigmoid).

void bench_train(std::vector<BenchGroup>& groups, bool csv) {
  size_t D = 768, H = 2048;
  float lr = 0.01f;

  // Shared sil lambda
  auto run_sil = [&](size_t batch, size_t iters) {
    auto W1 = sil::random({D, H}) * (1.0f / sqrtf(float(D)));
    auto b1 = sil::zeros<float>({H});
    auto W2 = sil::random({H, D}) * (1.0f / sqrtf(float(H)));
    auto b2 = sil::zeros<float>({D});
    auto x = sil::random({batch, D});
    auto Y = sil::random({batch, D});
    auto two = sil::array<float>(2.0f);
    auto inv_n = sil::array<float>(1.0f / static_cast<float>(batch * D));
    auto neg_lr = sil::array<float>(-lr);
    sil::synchronize();

    return measure(iters, [&] {
      auto n1 = x.linear(W1, b1);
      auto o1 = n1.sigmoid();
      auto n2 = o1.linear(W2, b2);
      auto o2 = n2.sigmoid();

      auto dout = (o2 - Y) * two * inv_n;

      dout = n2.sigmoid_backward(dout);
      auto dW2 = o1.transpose().dot(dout);
      auto db2 = dout.sum(0);
      auto dout1 = dout.dot(W2.transpose());

      dout1 = n1.sigmoid_backward(dout1);
      auto dW1 = x.transpose().dot(dout1);
      auto db1 = dout1.sum(0);

      W1 = W1 + dW1 * neg_lr; b1 = b1 + db1 * neg_lr;
      W2 = W2 + dW2 * neg_lr; b2 = b2 + db2 * neg_lr;

      sil::eval(W1, b1, W2, b2);
    }, 30, 2);
  };

  // --- GPU (sil-gpu, mlx, torch) ---
  if (!csv) print_section("training step (GPU)");
  for (auto batch : {64ul, 128ul}) {
    size_t iters = 20;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil(batch, iters)});

#ifdef BENCH_HAS_MLX
    {
      int b = static_cast<int>(batch);
      int d = static_cast<int>(D), h = static_cast<int>(H);
      auto W1 = mx::random::normal({d, h}) * (1.0f / sqrtf(float(D)));
      auto b1 = mx::zeros({h});
      auto W2 = mx::random::normal({h, d}) * (1.0f / sqrtf(float(H)));
      auto b2 = mx::zeros({d});
      auto x = mx::random::normal({b, d});
      auto Y = mx::random::normal({b, d});
      mx::eval(W1, b1, W2, b2, x, Y);

      entries.push_back({"mlx", measure(iters, [&] {
        auto n1 = mx::addmm(b1, x, W1);
        auto o1 = mx::sigmoid(n1);
        auto n2 = mx::addmm(b2, o1, W2);
        auto o2 = mx::sigmoid(n2);

        auto dout = mx::multiply(mx::subtract(o2, Y),
                                  mx::array(2.0f / (b * d)));

        auto s2 = mx::sigmoid(n2);
        dout = mx::multiply(dout, mx::multiply(s2, mx::subtract(mx::array(1.0f), s2)));
        auto dW2 = mx::matmul(mx::transpose(o1), dout);
        auto db2 = mx::sum(dout, 0);
        auto dout1 = mx::matmul(dout, mx::transpose(W2));

        auto s1 = mx::sigmoid(n1);
        dout1 = mx::multiply(dout1, mx::multiply(s1, mx::subtract(mx::array(1.0f), s1)));
        auto dW1 = mx::matmul(mx::transpose(x), dout1);
        auto db1 = mx::sum(dout1, 0);

        auto mlr = mx::array(lr);
        W1 = mx::subtract(W1, mx::multiply(dW1, mlr));
        b1 = mx::subtract(b1, mx::multiply(db1, mlr));
        W2 = mx::subtract(W2, mx::multiply(dW2, mlr));
        b2 = mx::subtract(b2, mx::multiply(db2, mlr));

        mx::eval(W1, b1, W2, b2);
      }, 30, 2)});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      long b = batch, d = D, h = H;
      auto W1 = torch::randn({d, h}, dev) * (1.0f / sqrtf(float(D)));
      auto b1 = torch::zeros({h}, dev);
      auto W2 = torch::randn({h, d}, dev) * (1.0f / sqrtf(float(H)));
      auto b2 = torch::zeros({d}, dev);
      auto x = torch::randn({b, d}, dev);
      auto Y = torch::randn({b, d}, dev);

      entries.push_back({"torch", measure(iters, [&] {
        auto n1 = torch::addmm(b1, x, W1);
        auto o1 = torch::sigmoid(n1);
        auto n2 = torch::addmm(b2, o1, W2);
        auto o2 = torch::sigmoid(n2);

        auto dout = (o2 - Y) * (2.0f / (b * d));

        auto s2 = torch::sigmoid(n2);
        dout = dout * (s2 * (1.0f - s2));
        auto dW2 = o1.t().mm(dout);
        auto db2 = dout.sum(0);
        auto dout1 = dout.mm(W2.t());

        auto s1 = torch::sigmoid(n1);
        dout1 = dout1 * (s1 * (1.0f - s1));
        auto dW1 = x.t().mm(dout1);
        auto db1 = dout1.sum(0);

        W1 -= dW1 * lr; b1 -= db1 * lr;
        W2 -= dW2 * lr; b2 -= db2 * lr;

        torch::mps::synchronize();
      }, 30, 2)});
    }
#endif

    auto group = BenchGroup{
        std::format("train step (batch={})", batch), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  // --- CPU (sil-cpu, eigen) ---
  if (!csv) print_section("training step (CPU)");
  sil::use_cpu();
  for (auto batch : {32ul, 64ul}) {
    size_t iters = 20;
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-cpu", run_sil(batch, iters)});

#ifdef BENCH_HAS_EIGEN
    {
      auto sigmoid = [](const Eigen::MatrixXf& x) -> Eigen::MatrixXf {
        return (1.0f + (-x).array().exp()).inverse().matrix();
      };
      Eigen::MatrixXf ex = Eigen::MatrixXf::Random(batch, D);
      Eigen::MatrixXf eY = Eigen::MatrixXf::Random(batch, D);
      Eigen::MatrixXf eW1 = Eigen::MatrixXf::Random(D, H) * (1.0f / sqrtf(float(D)));
      Eigen::VectorXf eb1 = Eigen::VectorXf::Zero(H);
      Eigen::MatrixXf eW2 = Eigen::MatrixXf::Random(H, D) * (1.0f / sqrtf(float(H)));
      Eigen::VectorXf eb2 = Eigen::VectorXf::Zero(D);

      entries.push_back({"eigen", measure(iters, [&] {
        auto n1 = (ex * eW1).rowwise() + eb1.transpose();
        auto o1 = sigmoid(n1);
        auto n2 = (o1 * eW2).rowwise() + eb2.transpose();
        auto o2 = sigmoid(n2);

        Eigen::MatrixXf dout = (o2 - eY) * (2.0f / (batch * D));
        auto s2 = sigmoid(n2);
        dout = dout.array() * s2.array() * (1.0f - s2.array());
        Eigen::MatrixXf dW2 = o1.transpose() * dout;
        Eigen::VectorXf db2 = dout.colwise().sum();
        Eigen::MatrixXf dout1 = dout * eW2.transpose();

        auto s1 = sigmoid(n1);
        dout1 = dout1.array() * s1.array() * (1.0f - s1.array());
        Eigen::MatrixXf dW1 = ex.transpose() * dout1;
        Eigen::VectorXf db1 = dout1.colwise().sum();

        eW1 -= dW1 * lr; eb1 -= db1 * lr;
        eW2 -= dW2 * lr; eb2 -= db2 * lr;
      }, 30, 2)});
    }
#endif

    auto group = BenchGroup{
        std::format("train step (batch={})", batch), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
  sil::use_mps();
}

int main(int argc, const char** argv) {
  auto mode = parse_output_mode(argc, argv);
  bool csv = (mode != OutputMode::bar);
  std::vector<BenchGroup> groups;
  bench_train(groups, csv);
  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "Training", "Full training step (forward + backward + SGD): GPU (batch 64-128) and CPU (batch 32-64)");
}
