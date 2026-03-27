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

// MNIST Autoencoder: 784 -> 512 -> 256 -> 64 -> 256 -> 512 -> 784
// Training: 1 epoch (600 batches x 100 images), SGD with MSE loss.
// Inference: encode+decode 10000 test images.

static const char* kTrainImages = "../test/train-images-idx3-ubyte";
static const char* kTrainLabels = "../test/train-labels-idx1-ubyte";
static const char* kTestImages  = "../test/t10k-images-idx3-ubyte";
static const char* kTestLabels  = "../test/t10k-labels-idx1-ubyte";

static constexpr size_t L[] = {784, 512, 256, 64, 256, 512, 784};
static constexpr size_t N_LAYERS = 6;

void bench_train(std::vector<BenchGroup>& groups, bool csv) {
  mnist_data train;
  if (!train.load(kTrainImages, kTrainLabels)) return;

  constexpr size_t batch = 100;
  float lr = 0.01f;

  // Shared sil lambda
  auto run_sil = [&]() {
    sil::array<float> W[N_LAYERS], b[N_LAYERS];
    for (size_t l = 0; l < N_LAYERS; l++) {
      W[l] = sil::random({L[l], L[l + 1]}) * (2.0f / sqrtf(float(L[l])));
      b[l] = sil::zeros<float>({L[l + 1]});
    }

    auto two = sil::array<float>(2.0f);
    auto inv_n = sil::array<float>(1.0f / float(batch * 784));
    auto neg_lr = sil::array<float>(-lr);
    // Pre-allocate dataset once (zero-copy slicing per batch)
    auto all_x = sil::array<float>({train.count, 784}, train.images.data());
    sil::synchronize();

    return measure(0, [&] {
      for (size_t i = 0; i + batch <= train.count; i += batch) {
        auto x = all_x.slice(i, batch);

        sil::array<float> net[N_LAYERS], out[N_LAYERS];
        auto h = x;
        for (size_t l = 0; l < N_LAYERS; l++) {
          net[l] = h.linear(W[l], b[l]);
          out[l] = net[l].sigmoid();
          h = out[l];
        }

        auto dout = (h - x) * two * inv_n;

        for (int l = N_LAYERS - 1; l >= 0; l--) {
          dout = net[l].sigmoid_backward(dout);
          auto& input = (l > 0) ? out[l - 1] : x;
          auto dW = input.transpose().dot(dout);
          auto db = dout.sum(0);
          if (l > 0) dout = dout.dot(W[l].transpose());
          W[l] = W[l] + dW * neg_lr;
          b[l] = b[l] + db * neg_lr;
        }
        sil::eval(W[0], b[0], W[1], b[1], W[2], b[2],
                  W[3], b[3], W[4], b[4], W[5], b[5]);
      }
      sil::synchronize();
    }, 0, 1);
  };

  // --- GPU training ---
  if (!csv) print_section("autoencoder training (GPU)");
  {
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil()});

#ifdef BENCH_HAS_MLX
    {
      int d[] = {784, 512, 256, 64, 256, 512, 784};
      std::vector<mx::array> mW(N_LAYERS, mx::array(0.f)), mb(N_LAYERS, mx::array(0.f));
      for (size_t l = 0; l < N_LAYERS; l++) {
        mW[l] = mx::random::normal({d[l], d[l + 1]}) * (2.0f / sqrtf(float(d[l])));
        mb[l] = mx::zeros({d[l + 1]});
      }
      for (size_t l = 0; l < N_LAYERS; l++) mx::eval(mW[l], mb[l]);

      std::vector<mx::array> batch_x;
      for (size_t i = 0; i + batch <= train.count; i += batch)
        batch_x.push_back(mx::array(&train.images[i * 784], {int(batch), 784}));
      for (auto& a : batch_x) mx::eval(a);

      entries.push_back({"mlx", measure(3, [&] {
        for (auto& x : batch_x) {
          std::vector<mx::array> mnet(N_LAYERS, mx::array(0.f)), mout(N_LAYERS, mx::array(0.f));
          auto h = x;
          for (size_t l = 0; l < N_LAYERS; l++) {
            mnet[l] = mx::addmm(mb[l], h, mW[l]);
            mout[l] = mx::sigmoid(mnet[l]);
            h = mout[l];
          }

          auto dout = mx::multiply(mx::subtract(h, x), mx::array(2.0f / (batch * 784)));

          for (int l = N_LAYERS - 1; l >= 0; l--) {
            auto s = mx::sigmoid(mnet[l]);
            dout = mx::multiply(dout, mx::multiply(s, mx::subtract(mx::array(1.0f), s)));
            auto& input = (l > 0) ? mout[l - 1] : x;
            auto dW = mx::matmul(mx::transpose(input), dout);
            auto db = mx::sum(dout, 0);
            if (l > 0) dout = mx::matmul(dout, mx::transpose(mW[l]));
            auto mlr = mx::array(lr);
            mW[l] = mx::subtract(mW[l], mx::multiply(dW, mlr));
            mb[l] = mx::subtract(mb[l], mx::multiply(db, mlr));
          }
          mx::eval(mW[0], mb[0], mW[1], mb[1], mW[2], mb[2],
                   mW[3], mb[3], mW[4], mb[4], mW[5], mb[5]);
        }
      }, 0, 1)});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      long d[] = {784, 512, 256, 64, 256, 512, 784};
      torch::Tensor tW[N_LAYERS], tb[N_LAYERS];
      for (size_t l = 0; l < N_LAYERS; l++) {
        tW[l] = torch::randn({d[l], d[l + 1]}, dev) * (2.0f / sqrtf(float(d[l])));
        tb[l] = torch::zeros({d[l + 1]}, dev);
      }

      std::vector<torch::Tensor> batch_x;
      for (size_t i = 0; i + batch <= train.count; i += batch)
        batch_x.push_back(torch::from_blob(
            const_cast<float*>(&train.images[i * 784]),
            {long(batch), 784}).to(dev));

      entries.push_back({"torch", measure(3, [&] {
        for (auto& x : batch_x) {
          torch::Tensor tnet[N_LAYERS], tout[N_LAYERS];
          auto h = x;
          for (size_t l = 0; l < N_LAYERS; l++) {
            tnet[l] = torch::addmm(tb[l], h, tW[l]);
            tout[l] = torch::sigmoid(tnet[l]);
            h = tout[l];
          }

          auto dout = (h - x) * (2.0f / (batch * 784));

          for (int l = N_LAYERS - 1; l >= 0; l--) {
            auto s = torch::sigmoid(tnet[l]);
            dout = dout * (s * (1.0f - s));
            auto& input = (l > 0) ? tout[l - 1] : x;
            auto dW = input.t().mm(dout);
            auto db = dout.sum(0);
            if (l > 0) dout = dout.mm(tW[l].t());
            tW[l] -= dW * lr;
            tb[l] -= db * lr;
          }
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
  if (!csv) print_section("autoencoder training (CPU)");
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

      Eigen::MatrixXf W[N_LAYERS], b_mat[N_LAYERS];
      for (size_t l = 0; l < N_LAYERS; l++) {
        float scale = 2.0f / std::sqrt(float(L[l]));
        W[l] = Eigen::MatrixXf::Random(L[l], L[l + 1]) * scale;
        b_mat[l] = Eigen::MatrixXf::Zero(1, L[l + 1]);
      }
      Eigen::MatrixXf images = Eigen::Map<const Eigen::MatrixXf>(
          train.images.data(), train.count, 784);

      entries.push_back({"eigen", measure(0, [&] {
        for (size_t i = 0; i < train.count; i += batch) {
          Eigen::MatrixXf x = images.middleRows(i, batch);
          // Forward
          Eigen::MatrixXf h[N_LAYERS + 1];
          h[0] = x;
          for (size_t l = 0; l < N_LAYERS; l++)
            h[l + 1] = sigmoid((h[l] * W[l]).rowwise() + b_mat[l].row(0));
          // Backward (MSE loss)
          Eigen::MatrixXf dout = (h[N_LAYERS] - x) * (2.0f / (batch * 784));
          for (int l = N_LAYERS - 1; l >= 0; l--) {
            Eigen::MatrixXf sig_grad = h[l + 1].array() * (1.0f - h[l + 1].array());
            dout = dout.array() * sig_grad.array();
            Eigen::MatrixXf dW = h[l].transpose() * dout;
            Eigen::VectorXf db = dout.colwise().sum();
            if (l > 0) dout = dout * W[l].transpose();
            W[l] -= lr * dW;
            b_mat[l].row(0) -= lr * db.transpose();
          }
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

  // Shared sil lambda
  auto run_sil = [&]() {
    sil::array<float> W[N_LAYERS], b[N_LAYERS];
    for (size_t l = 0; l < N_LAYERS; l++) {
      W[l] = sil::random({L[l], L[l + 1]});
      b[l] = sil::zeros<float>({L[l + 1]});
    }
    auto x = sil::array<float>({test.count, 784}, test.images.data());
    sil::synchronize();

    return measure(0, [&] {
      auto h = x;
      for (size_t l = 0; l < N_LAYERS; l++)
        h = h.linear(W[l], b[l]).sigmoid();
      sil::eval(h);
    }, 0, 1);
  };

  // --- GPU inference ---
  if (!csv) print_section("autoencoder inference (GPU)");
  {
    std::vector<BenchEntry> entries;

    entries.push_back({"sil-gpu", run_sil()});

#ifdef BENCH_HAS_MLX
    {
      int d[] = {784, 512, 256, 64, 256, 512, 784};
      std::vector<mx::array> mW(N_LAYERS, mx::array(0.f)), mb(N_LAYERS, mx::array(0.f));
      for (size_t l = 0; l < N_LAYERS; l++) {
        mW[l] = mx::random::normal({d[l], d[l + 1]});
        mb[l] = mx::zeros({d[l + 1]});
      }
      auto x = mx::array(test.images.data(), {int(test.count), 784});
      for (size_t l = 0; l < N_LAYERS; l++) mx::eval(mW[l], mb[l]);
      mx::eval(x);

      entries.push_back({"mlx", measure(0, [&] {
        auto h = x;
        for (size_t l = 0; l < N_LAYERS; l++)
          h = mx::sigmoid(mx::addmm(mb[l], h, mW[l]));
        mx::eval(h);
      }, 0, 1)});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      long d[] = {784, 512, 256, 64, 256, 512, 784};
      torch::Tensor tW[N_LAYERS], tb[N_LAYERS];
      for (size_t l = 0; l < N_LAYERS; l++) {
        tW[l] = torch::randn({d[l], d[l + 1]}, dev);
        tb[l] = torch::zeros({d[l + 1]}, dev);
      }
      auto x = torch::from_blob(const_cast<float*>(test.images.data()),
                                 {long(test.count), 784}).to(dev);

      entries.push_back({"torch", measure(0, [&] {
        auto h = x;
        for (size_t l = 0; l < N_LAYERS; l++)
          h = torch::sigmoid(torch::addmm(tb[l], h, tW[l]));
        torch::mps::synchronize();
      }, 0, 1)});
    }
#endif

    auto group = BenchGroup{"inference (10000 images)", std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  // --- CPU inference ---
  if (!csv) print_section("autoencoder inference (CPU)");
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

      Eigen::MatrixXf W[N_LAYERS], b_mat[N_LAYERS];
      for (size_t l = 0; l < N_LAYERS; l++) {
        W[l] = Eigen::MatrixXf::Random(L[l], L[l + 1]);
        b_mat[l] = Eigen::MatrixXf::Zero(1, L[l + 1]);
      }
      Eigen::MatrixXf images = Eigen::Map<const Eigen::MatrixXf>(
          test.images.data(), test.count, 784);

      entries.push_back({"eigen", measure(0, [&] {
        Eigen::MatrixXf h = images;
        for (size_t l = 0; l < N_LAYERS; l++)
          h = sigmoid((h * W[l]).rowwise() + b_mat[l].row(0));
      }, 0, 1)});
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
  if (mode == OutputMode::table) print_table(groups, "MNIST Autoencoder",
      "784->512->256->64->256->512->784 (sigmoid, MSE, SGD). GPU (sil/mlx/torch) and CPU (sil).");
}
