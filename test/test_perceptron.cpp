#include <silarray.h>

#include "doctest.h"

class LogicGate {
 private:
  float w0 = 0.1;
  float w1 = 0.1;
  float b = 0.1;

  auto range(size_t count) {
    return std::views::iota(1) | std::views::take(count);
  }

  int predict(int x0, int x1) {
    auto y = (x0 * w0) + (x1 * w1) + b;
    return y > 0 ? 1 : 0;
  }

 public:
  LogicGate(sil::array<int>&& dataset) {
    auto max_iteration = 10;
    auto learning_rate = 1.0;

    for (auto n : range(max_iteration)) {
      for (auto [x0, x1, t] : dataset.rows<3>()) {
        auto y = predict(x0, x1);
        auto diff = t - y;
        auto update = diff * learning_rate;

        w0 += update * x0;
        w1 += update * x1;
        b += update;
      }
    }
  }

  int operator()(int x0, int x1) { return predict(x0, x1); }
};

TEST_CASE("perceptron: nand") {
  auto AND = LogicGate({
      {0, 0, 0},
      {0, 1, 0},
      {1, 0, 0},
      {1, 1, 1},
  });

  auto OR = LogicGate({
      {0, 0, 0},
      {0, 1, 1},
      {1, 0, 1},
      {1, 1, 1},
  });

  auto NAND = LogicGate({
      {0, 0, 1},
      {0, 1, 1},
      {1, 0, 1},
      {1, 1, 0},
  });

  auto XOR = [&](int x0, int x1) { return AND(NAND(x0, x1), OR(x0, x1)); };

  CHECK(AND(0, 0) == 0);
  CHECK(AND(0, 1) == 0);
  CHECK(AND(1, 0) == 0);
  CHECK(AND(1, 1) == 1);

  CHECK(OR(0, 0) == 0);
  CHECK(OR(0, 1) == 1);
  CHECK(OR(1, 0) == 1);
  CHECK(OR(1, 1) == 1);

  CHECK(NAND(0, 0) == 1);
  CHECK(NAND(0, 1) == 1);
  CHECK(NAND(1, 0) == 1);
  CHECK(NAND(1, 1) == 0);

  CHECK(XOR(0, 0) == 0);
  CHECK(XOR(0, 1) == 1);
  CHECK(XOR(1, 0) == 1);
  CHECK(XOR(1, 1) == 0);
}
