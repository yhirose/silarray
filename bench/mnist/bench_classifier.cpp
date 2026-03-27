#include <silarray.h>

#include "../bench_common.h"
#include "mnist_data.h"

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

// MNIST Classifier: 784 -> 50 -> sigmoid -> 10 -> sigmoid
// Training: 1 epoch (600 batches x 100 images), SGD
// Inference: 10000 test images

static const char* kTrainImages = "../test/train-images-idx3-ubyte";
static const char* kTrainLabels = "../test/train-labels-idx1-ubyte";
static const char* kTestImages  = "../test/t10k-images-idx3-ubyte";
static const char* kTestLabels  = "../test/t10k-labels-idx1-ubyte";

void bench_train(std::vector<BenchGroup>& groups, bool csv) {
  mnist_data train;
  if (!train.load(kTrainImages, kTrainLabels)) return;

  constexpr size_t D = 784, H = 50, C = 10, batch = 100;
  float lr = 0.5f;

  // Shared sil lambda
  auto run_sil = [&]() {
    auto W1 = (sil::random({D, H}) * 2.0f - 1.0f) * (1.0f / sqrtf(float(D)));
    auto b1 = sil::zeros<float>({H});
    auto W2 = (sil::random({H, C}) * 2.0f - 1.0f) * (1.0f / sqrtf(float(H)));
    auto b2 = sil::zeros<float>({C});
    auto two = sil::array<float>(2.0f);
    auto inv_n = sil::array<float>(1.0f / float(batch * C));
    auto neg_lr = sil::array<float>(-lr);

    // Pre-allocate dataset (zero-copy slicing per batch)
    auto all_x = sil::array<float>({train.count, D}, train.images.data());
    // Pre-compute one-hot labels
    auto all_labels = sil::array<float>({train.count}, train.labels.data());
    auto all_Y = all_labels.template one_hot<float>(C);
    sil::synchronize();

    return measure(0, [&] {
      for (size_t i = 0; i + batch <= train.count; i += batch) {
        auto x = all_x.slice(i, batch);
        auto Y = all_Y.slice(i, batch);

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
      }
      sil::synchronize();
    }, 0, 1);
  };

  // --- GPU training ---
  if (!csv) print_section("classifier training (GPU)");
  {
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil()});

#ifdef BENCH_HAS_MLX
    {
      auto W1 = mx::random::normal({int(D), int(H)}) * (1.0f / sqrtf(float(D)));
      auto b1 = mx::zeros({int(H)});
      auto W2 = mx::random::normal({int(H), int(C)}) * (1.0f / sqrtf(float(H)));
      auto b2 = mx::zeros({int(C)});
      mx::eval(W1, b1, W2, b2);

      std::vector<mx::array> batch_x, batch_y;
      for (size_t i = 0; i + batch <= train.count; i += batch) {
        batch_x.push_back(mx::array(&train.images[i * D], {int(batch), int(D)}));
        std::vector<float> oh(batch * C, 0.0f);
        for (size_t j = 0; j < batch; j++) oh[j * C + train.labels[i + j]] = 1.0f;
        batch_y.push_back(mx::array(oh.data(), {int(batch), int(C)}));
      }
      for (auto& a : batch_x) mx::eval(a);
      for (auto& a : batch_y) mx::eval(a);

      entries.push_back({"mlx", measure(0, [&] {
        for (size_t k = 0; k < batch_x.size(); k++) {
          auto& x = batch_x[k];
          auto& Y = batch_y[k];

          auto n1 = mx::addmm(b1, x, W1);
          auto o1 = mx::sigmoid(n1);
          auto n2 = mx::addmm(b2, o1, W2);
          auto o2 = mx::sigmoid(n2);

          auto dout = mx::multiply(mx::subtract(o2, Y), mx::array(2.0f / (batch * C)));
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
        }
      }, 0, 1)});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto W1 = torch::randn({long(D), long(H)}, dev) * (1.0f / sqrtf(float(D)));
      auto b1 = torch::zeros({long(H)}, dev);
      auto W2 = torch::randn({long(H), long(C)}, dev) * (1.0f / sqrtf(float(H)));
      auto b2 = torch::zeros({long(C)}, dev);

      std::vector<torch::Tensor> batch_x, batch_y;
      for (size_t i = 0; i + batch <= train.count; i += batch) {
        batch_x.push_back(torch::from_blob(
            const_cast<float*>(&train.images[i * D]),
            {long(batch), long(D)}).to(dev));
        std::vector<float> oh(batch * C, 0.0f);
        for (size_t j = 0; j < batch; j++) oh[j * C + train.labels[i + j]] = 1.0f;
        batch_y.push_back(torch::from_blob(oh.data(),
            {long(batch), long(C)}).clone().to(dev));
      }
      torch::mps::synchronize();

      entries.push_back({"torch", measure(0, [&] {
        for (size_t k = 0; k < batch_x.size(); k++) {
          auto& x = batch_x[k];
          auto& Y = batch_y[k];

          auto n1 = torch::addmm(b1, x, W1);
          auto o1 = torch::sigmoid(n1);
          auto n2 = torch::addmm(b2, o1, W2);
          auto o2 = torch::sigmoid(n2);

          auto dout = (o2 - Y) * (2.0f / (batch * C));
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
        }
      }, 0, 1)});
    }
#endif

    auto group = BenchGroup{"train 1 epoch (60000 images)", std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  // --- CPU training ---
  if (!csv) print_section("classifier training (CPU)");
  {
    std::vector<BenchEntry> entries;

    sil::use_cpu();
    entries.push_back({"sil-cpu", run_sil()});
    sil::use_mps();

#ifdef BENCH_HAS_EIGEN
    {
      auto sigmoid = [](const Eigen::MatrixXf& x) -> Eigen::MatrixXf {
        return (1.0f + (-x).array().exp()).inverse().matrix();
      };

      Eigen::MatrixXf W1 = Eigen::MatrixXf::Random(D, H) * (1.0f / sqrtf(float(D)));
      Eigen::VectorXf b1 = Eigen::VectorXf::Zero(H);
      Eigen::MatrixXf W2 = Eigen::MatrixXf::Random(H, C) * (1.0f / sqrtf(float(H)));
      Eigen::VectorXf b2 = Eigen::VectorXf::Zero(C);

      entries.push_back({"eigen", measure(0, [&] {
        for (size_t i = 0; i + batch <= train.count; i += batch) {
          Eigen::Map<const Eigen::MatrixXf> x(&train.images[i * D], batch, D);
          Eigen::MatrixXf Y = Eigen::MatrixXf::Zero(batch, C);
          for (size_t j = 0; j < batch; j++) Y(j, train.labels[i + j]) = 1.0f;

          auto n1 = (x * W1).rowwise() + b1.transpose();
          auto o1 = sigmoid(n1);
          auto n2 = (o1 * W2).rowwise() + b2.transpose();
          auto o2 = sigmoid(n2);

          Eigen::MatrixXf dout = (o2 - Y) * (2.0f / (batch * C));
          auto s2 = sigmoid(n2);
          dout = dout.array() * s2.array() * (1.0f - s2.array());
          Eigen::MatrixXf dW2 = o1.transpose() * dout;
          Eigen::VectorXf db2 = dout.colwise().sum();
          Eigen::MatrixXf dout1 = dout * W2.transpose();
          auto s1 = sigmoid(n1);
          dout1 = dout1.array() * s1.array() * (1.0f - s1.array());
          Eigen::MatrixXf dW1 = x.transpose() * dout1;
          Eigen::VectorXf db1 = dout1.colwise().sum();

          W1 -= dW1 * lr; b1 -= db1 * lr;
          W2 -= dW2 * lr; b2 -= db2 * lr;
        }
      }, 0, 1)});
    }
#endif

    auto group = BenchGroup{"train 1 epoch (60000 images)", std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

void bench_inference(std::vector<BenchGroup>& groups, bool csv) {
  mnist_data test;
  if (!test.load(kTestImages, kTestLabels)) return;

  constexpr size_t D = 784, H = 50, C = 10;

  // Shared sil lambda
  auto run_sil = [&]() {
    auto W1 = sil::random({D, H});
    auto b1 = sil::zeros<float>({H});
    auto W2 = sil::random({H, C});
    auto b2 = sil::zeros<float>({C});
    auto x = sil::array<float>({test.count, D}, test.images.data());
    sil::synchronize();

    return measure(20, [&] {
      auto o1 = x.linear(W1, b1).sigmoid();
      auto o2 = o1.linear(W2, b2).sigmoid();
      sil::eval(o2);
    });
  };

  // --- GPU inference ---
  if (!csv) print_section("classifier inference (GPU)");
  {
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil()});

#ifdef BENCH_HAS_MLX
    {
      auto W1 = mx::random::normal({int(D), int(H)});
      auto b1 = mx::zeros({int(H)});
      auto W2 = mx::random::normal({int(H), int(C)});
      auto b2 = mx::zeros({int(C)});
      auto x = mx::array(test.images.data(), {int(test.count), int(D)});
      mx::eval(W1, b1, W2, b2, x);

      entries.push_back({"mlx", measure(20, [&] {
        auto o1 = mx::sigmoid(mx::addmm(b1, x, W1));
        auto o2 = mx::sigmoid(mx::addmm(b2, o1, W2));
        mx::eval(o2);
      })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto W1 = torch::randn({long(D), long(H)}, dev);
      auto b1 = torch::zeros({long(H)}, dev);
      auto W2 = torch::randn({long(H), long(C)}, dev);
      auto b2 = torch::zeros({long(C)}, dev);
      auto x = torch::from_blob(const_cast<float*>(test.images.data()),
                                 {long(test.count), long(D)}).to(dev);
      torch::mps::synchronize();

      entries.push_back({"torch", measure(20, [&] {
        auto o1 = torch::sigmoid(torch::addmm(b1, x, W1));
        auto o2 = torch::sigmoid(torch::addmm(b2, o1, W2));
        torch::mps::synchronize();
      })});
    }
#endif

    auto group = BenchGroup{"inference (10000 images)", std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  // --- CPU inference ---
  if (!csv) print_section("classifier inference (CPU)");
  {
    std::vector<BenchEntry> entries;

    sil::use_cpu();
    entries.push_back({"sil-cpu", run_sil()});
    sil::use_mps();

#ifdef BENCH_HAS_EIGEN
    {
      auto sigmoid = [](const Eigen::MatrixXf& x) -> Eigen::MatrixXf {
        return (1.0f + (-x).array().exp()).inverse().matrix();
      };
      Eigen::MatrixXf W1 = Eigen::MatrixXf::Random(D, H);
      Eigen::VectorXf b1 = Eigen::VectorXf::Zero(H);
      Eigen::MatrixXf W2 = Eigen::MatrixXf::Random(H, C);
      Eigen::VectorXf b2 = Eigen::VectorXf::Zero(C);
      Eigen::Map<const Eigen::MatrixXf> x(test.images.data(), test.count, D);

      entries.push_back({"eigen", measure(20, [&] {
        auto o1 = sigmoid((x * W1).rowwise() + b1.transpose());
        Eigen::MatrixXf o2 = sigmoid((o1 * W2).rowwise() + b2.transpose());
      })});
    }
#endif

    auto group = BenchGroup{"inference (10000 images)", std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  auto mode = parse_output_mode(argc, argv);
  bool csv = (mode != OutputMode::bar);
  std::vector<BenchGroup> groups;

  bench_train(groups, csv);
  bench_inference(groups, csv);

  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "MNIST Classifier",
      "784->50->10 (sigmoid, MSE, SGD). GPU (sil/mlx) and CPU (sil/eigen).");
}
