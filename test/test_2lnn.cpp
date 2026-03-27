#include <silarray.h>

#include "doctest.h"

sil::array<float> mean_square_error_derivative(float dout,
                                               const sil::array<float>& out,
                                               const sil::array<float>& Y) {
  return dout * (2 * (out - Y));
}

sil::array<float> sigmoid_derivative(const sil::array<float>& dout,
                                     const sil::array<float>& x) {
  return x.sigmoid_backward(dout);
}

std::tuple<sil::array<float>, sil::array<float>, sil::array<float>>
linear_derivative(const sil::array<float>& dout, const sil::array<float>& x,
                  const sil::array<float>& W) {
  auto dx = dout.dot(W.transpose());
  auto dW = x.transpose().dot(dout);
  auto db = dout.sum(0);
  return {dx, dW, db};
}

struct TwoLayerNeuralNetwork {
  sil::array<float> W1 = sil::random({2, 3}) * 2.0 - 1.0;
  sil::array<float> b1 = sil::random({3}) * 2.0 - 1.0;

  sil::array<float> W2 = sil::random({3, 1}) * 2.0 - 1.0;
  sil::array<float> b2 = sil::random({1}) * 2.0 - 1.0;

  sil::array<float> x;
  sil::array<float> net1;
  sil::array<float> out1;
  sil::array<float> net2;
  sil::array<float> out2;

  sil::array<float> Y;

  sil::array<float> forward(const sil::array<float>& x) {
    // Input → Hidden
    auto net1 = x.linear(W1, b1);
    auto out1 = net1.sigmoid();

    // Hidden → Output
    auto net2 = out1.linear(W2, b2);
    auto out2 = net2.sigmoid();

    // Save variables for backpropagation
    this->x = x;
    this->net1 = net1;
    this->out1 = out1;
    this->net2 = net2;
    this->out2 = out2;

    return out2;
  }

  float loss(const sil::array<float>& out, const sil::array<float>& Y) {
    // Save variables for back propagation
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

    const auto& [dx, dW1, db1] = linear_derivative(dout, this->x, this->W1);

    return {dW1, db1, dW2, db2};
  }
};

sil::array<float> predict(TwoLayerNeuralNetwork& model,
                          const sil::array<float>& x) {
  auto out = model.forward(x);  // 0..1
  return sil::where<float>(out > 0.5, 1, 0);
}

void train(TwoLayerNeuralNetwork& model, const sil::array<float>& X,
           const sil::array<float>& Y, size_t epochs, float learning_rate) {
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    // Save variables for back propagation
    auto out = model.forward(X);
    auto loss = model.loss(out, Y);

    // Get gradients of weight parameters
    const auto& [dW1, db1, dW2, db2] = model.backward();

    // Update weights
    model.W1 -= dW1 * learning_rate;
    model.b1 -= db1 * learning_rate;
    model.W2 -= dW2 * learning_rate;
    model.b2 -= db2 * learning_rate;

    // Show progress message
    if (epoch % (epochs / 10) == 0) {
      printf("Epoch: %zu, Loss: %f\n", epoch, loss);
    }
  }
}

TEST_CASE("array: mean_square_error") {
  auto a = sil::array<float>{1, 2, 3, 4};
  auto b = sil::array<float>{0, 2, 3, 6};
  auto mean = a.mean_square_error(b);
  CHECK(mean == 1.25);
}

TEST_CASE("2 layer NN: xor") {
  auto X = sil::array<float>{
      {0, 0},
      {0, 1},
      {1, 0},
      {1, 1},
  };

  auto Y_XOR = sil::array<float>{
      {0},
      {1},
      {1},
      {0},
  };

  TwoLayerNeuralNetwork m;

  train(m, X, Y_XOR, 2000, 0.5);

  auto out = predict(m, X);
  std::cout << out << std::endl;
}
