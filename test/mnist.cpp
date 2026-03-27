#include <silarray.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

class memory_mapped_file {
 public:
  memory_mapped_file(const char* path) {
    fd_ = open(path, O_RDONLY);
    assert(fd_ != -1);

    struct stat sb;
    fstat(fd_, &sb);
    size_ = sb.st_size;

    addr_ = ::mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    assert(addr_ != MAP_FAILED);
  }

  ~memory_mapped_file() {
    munmap(addr_, size_);
    close(fd_);
  }

  const char* data() const { return (const char*)addr_; }

 private:
  int fd_ = -1;
  size_t size_ = 0;
  void* addr_ = MAP_FAILED;
};

class mnist_data {
 public:
  mnist_data(const char* label_data_path, const char* image_data_path)
      : label_data_(label_data_path), image_data_(image_data_path) {
    if (read32(label_data_, 0) != 2049) {
      throw std::runtime_error("invalid label data format.");
    }

    if (read32(image_data_, 0) != 2051) {
      throw std::runtime_error("invalid image data format.");
    }

    number_of_items_ = read32(label_data_, 4);
    if (number_of_items_ != read32(image_data_, 4)) {
      throw std::runtime_error("image data doesn't match label data.");
    }

    labels_ = reinterpret_cast<const uint8_t*>(label_data_.data() + 8);

    number_of_rows_ = read32(image_data_, 8);
    number_of_columns_ = read32(image_data_, 12);
    pixels_ = reinterpret_cast<const uint8_t*>(image_data_.data() + 16);
  }

  size_t size() const { return number_of_items_; }

  // Label
  const uint8_t* label_data() const { return labels_; }

  size_t label(size_t i) const { return labels_[i]; }

  // Image
  size_t image_rows() const { return number_of_rows_; }

  size_t image_columns() const { return number_of_columns_; }

  size_t image_pixel_size() const {
    return number_of_rows_ * number_of_columns_;
  }

  uint8_t pixel(size_t row, size_t col) const {
    return pixels_[number_of_columns_ * row + col];
  }

  // Normalized data
  const float* normalized_label_data() const {
    if (normalized_labels_.empty()) {
      auto total = size();
      normalized_labels_.resize(total);
      for (size_t i = 0; i < total; i++) {
        normalized_labels_[i] = (float)labels_[i];
      }
    }
    return normalized_labels_.data();
  }

  const float* normalized_image_data() const {
    if (normalized_pixels_.empty()) {
      auto total = image_pixel_size() * size();
      normalized_pixels_.resize(total);
      for (size_t i = 0; i < total; i++) {
        normalized_pixels_[i] = (float)pixels_[i] / (float)255;
      }
    }
    return normalized_pixels_.data();
  }

 private:
  uint32_t read32(const memory_mapped_file& mm, size_t off) {
    auto p = mm.data() + off;
    return __builtin_bswap32(*reinterpret_cast<const uint32_t*>(p));
  }

  size_t number_of_items_ = 0;
  size_t number_of_rows_ = 0;
  size_t number_of_columns_ = 0;

  const uint8_t* labels_ = nullptr;
  const uint8_t* pixels_ = nullptr;

  memory_mapped_file label_data_;
  memory_mapped_file image_data_;

  mutable std::vector<float> normalized_pixels_;
  mutable std::vector<float> normalized_labels_;
};

sil::array<float> mean_square_error_derivative(float dout,
                                               const sil::array<float>& out,
                                               const sil::array<float>& Y) {
  return dout * (2 * (out - Y)) / Y.length();
}

sil::array<float> sigmoid_derivative(const sil::array<float>& dout,
                                     const sil::array<float>& x) {
  auto y = x.sigmoid();
  return dout * (y * (1 - y));
}

std::tuple<sil::array<float>, sil::array<float>, sil::array<float>>
linear_derivative(const sil::array<float>& dout, const sil::array<float>& x,
                  const sil::array<float>& W) {
  auto dx = dout.dot(W.transpose());
  auto dW = x.transpose().dot(dout);
  auto db = dout.sum(0);
  return {dx, dW, db};
}

struct MnistNetwork {
  sil::array<float> W1;
  sil::array<float> b1;
  sil::array<float> W2;
  sil::array<float> b2;

  sil::array<float> x_;
  sil::array<float> net1;
  sil::array<float> out1;
  sil::array<float> net2;
  sil::array<float> out2;

  sil::array<float> Y;

  MnistNetwork() {
    // Xavier initialization to prevent sigmoid saturation
    W1 = (sil::random({784, 50}) * 2.0 - 1.0) * (1.0 / sqrt(784.0));
    b1 = sil::zeros<float>({50});
    W2 = (sil::random({50, 10}) * 2.0 - 1.0) * (1.0 / sqrt(50.0));
    b2 = sil::zeros<float>({10});
  }

  sil::array<float> forward(const sil::array<float>& x) {
    auto n1 = x.linear(W1, b1);
    auto o1 = n1.sigmoid();
    auto n2 = o1.linear(W2, b2);
    auto o2 = n2.sigmoid();

    this->x_ = x;
    this->net1 = std::move(n1);
    this->out1 = std::move(o1);
    this->net2 = std::move(n2);
    this->out2 = std::move(o2);

    return this->out2;
  }

  float loss(const sil::array<float>& out, const sil::array<float>& Y) {
    this->Y = Y;
    return out.mean_square_error(Y);
  }

  std::tuple<sil::array<float>, sil::array<float>, sil::array<float>,
             sil::array<float>>
  backward() {
    auto dout = mean_square_error_derivative(1.0, this->out2, this->Y);
    dout = sigmoid_derivative(dout, this->net2);

    const auto& [dout1, dW2, db2] =
        linear_derivative(dout, this->out1, this->W2);

    dout = sigmoid_derivative(dout1, this->net1);

    const auto& [dx, dW1, db1] = linear_derivative(dout, this->x_, this->W1);

    return {dW1, db1, dW2, db2};
  }
};

sil::array<float> predict(MnistNetwork& model, const sil::array<float>& x) {
  return model.forward(x).softmax();
}

void train(MnistNetwork& model, const mnist_data& data, size_t epochs,
           float learning_rate) {
  size_t batch_size = 100;
  size_t data_size = data.size();
  size_t pixel_size = data.image_pixel_size();
  auto image_data = data.normalized_image_data();
  auto label_data = data.normalized_label_data();

  std::mt19937 rng(42);
  std::vector<size_t> indices(data_size);
  std::iota(indices.begin(), indices.end(), 0);

  std::vector<float> batch_images(batch_size * pixel_size);
  std::vector<float> batch_labels(batch_size);

  for (size_t epoch = 0; epoch < epochs; epoch++) {
    std::shuffle(indices.begin(), indices.end(), rng);
    float total_loss = 0;
    size_t batch_count = 0;

    for (size_t i = 0; i + batch_size <= data_size; i += batch_size) {
      for (size_t j = 0; j < batch_size; j++) {
        size_t idx = indices[i + j];
        std::copy(image_data + idx * pixel_size,
                  image_data + (idx + 1) * pixel_size,
                  batch_images.data() + j * pixel_size);
        batch_labels[j] = label_data[idx];
      }

      auto batch_X =
          sil::array<float>({batch_size, pixel_size}, batch_images.data());
      auto batch_Y =
          sil::array<float>({batch_size}, batch_labels.data()).one_hot(10);

      auto out = model.forward(batch_X);
      auto loss = model.loss(out, batch_Y);

      const auto& [dW1, db1, dW2, db2] = model.backward();

      model.W1 -= dW1 * learning_rate;
      model.b1 -= db1 * learning_rate;
      model.W2 -= dW2 * learning_rate;
      model.b2 -= db2 * learning_rate;

      total_loss += loss;
      batch_count++;
    }

    printf("Epoch: %zu, Avg Loss: %f\n", epoch, total_loss / batch_count);
  }
}

int main(int argc, const char** argv) {
  try {
    if (argc > 1) {
      if (std::string("--cpu") == argv[1]) {
        sil::use_cpu();
      } else if (std::string("--gpu") == argv[1]) {
        sil::use_mps();
      }
    }

    // Training
    auto train_data =
        mnist_data("train-labels-idx1-ubyte", "train-images-idx3-ubyte");

    MnistNetwork m;

    train(m, train_data, 50, 0.1);

    // Evaluation on test data
    auto test_data =
        mnist_data("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte");

    auto p = test_data.normalized_image_data();
    size_t batch_size = 100;
    size_t accuracy_cnt = 0;

    for (size_t i = 0; i < test_data.size(); i += batch_size) {
      auto x = sil::array<float>({batch_size, test_data.image_pixel_size()},
                                 p + test_data.image_pixel_size() * i);
      auto y = predict(m, x);
      auto e = sil::array<int>({batch_size}, test_data.label_data() + i);
      auto a = y.argmax();

      auto r = e == a;
      accuracy_cnt += r.count();
    }

    auto accuracy = (double)accuracy_cnt / (double)test_data.size();
    std::cout << "MNIST Test Accuracy: " << accuracy << std::endl;
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
  }
}
