#include <silarray.h>

#include "doctest.h"

using namespace sil;

TEST_CASE("example: create empty array") {
  auto i = empty<int>({2, 3, 2});
  auto f = empty<float>({2, 3, 2});
  // auto d = empty<double>({2, 3}); // cannot compile...
}

TEST_CASE("example: create array with constants") {
  auto s = array<float>(1);
  auto v = array<float>{1, 2, 3, 4, 5, 6};
  auto m = array<float>{{1, 2}, {3, 4}, {5, 6}};
  auto t = array<float>{{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}};

  // std::cout << s.print_info() << std::endl << s << std::endl << std::endl;
  // std::cout << v.print_info() << std::endl << v << std::endl << std::endl;
  // std::cout << m.print_info() << std::endl << m << std::endl << std::endl;
  // std::cout << t.print_info() << std::endl << t << std::endl << std::endl;

  // dtype: float, dim: 0, shape: {}, strides: {1}
  // 1
  //
  // dtype: float, dim: 1, shape: {6}, strides: {1}
  // {1, 2, 3, 4, 5, 6}
  //
  // dtype: float, dim: 2, shape: {3, 2}, strides: {2, 1}
  // {{1, 2},
  //  {3, 4},
  //  {5, 6}}
  //
  // dtype: float, dim: 3, shape: {2, 3, 2}, strides: {6, 2, 1}
  // {{{1, 2},
  //   {3, 4},
  //   {5, 6}},
  //
  //  {{7, 8},
  //   {9, 10},
  //   {11, 12}}}
}

TEST_CASE("example: create array with shape") {
  auto zeros1 = array<float>({2, 3, 2}, 0);
  auto zeros2 = zeros<float>({2, 3, 2});
  CHECK(array_equal(zeros1, zeros2));

  auto ones1 = array<float>({2, 3, 2}, 1);
  auto ones2 = ones<float>({2, 3, 2});
  CHECK(array_equal(ones1, ones2));

  auto rand = random({2, 3, 2});
  CHECK(rand.all([](auto val) { return 0 <= val && val < 1.0; }));

  auto v = std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto from_iterator = array<float>({2, 3, 2}, v.begin());
  auto from_range1 = array<float>({2, 3, 2}, v);
  auto from_range2 = array<float>({2, 3, 2}, std::views::iota(1));
  auto expected = array<float>({
      {{1, 2}, {3, 4}, {5, 6}},
      {{7, 8}, {9, 10}, {11, 12}},
  });
  CHECK(array_equal(from_iterator, expected));
  CHECK(array_equal(from_range1, expected));
  CHECK(array_equal(from_range2, expected));
}

TEST_CASE("example: clone array") {
  auto a = ones<float>({4});

  auto cloned = a.clone();
  cloned.zeros();
  CHECK(array_equal(a, {1, 1, 1, 1}));

  auto assigned = a;
  assigned.zeros();
  CHECK(array_equal(a, {0, 0, 0, 0}));
}

TEST_CASE("example: arithmatic operations") {
  auto a = array<float>{{1, 2}, {3, 4}};
  auto b = array<float>{{1, 2}, {3, 4}};

  auto add = a + b;
  CHECK(array_equal(add, {{2, 4}, {6, 8}}));

  auto sub = a - b;
  CHECK(array_equal(sub, {{0, 0}, {0, 0}}));

  auto mul = a * b;
  CHECK(array_equal(mul, {{1, 4}, {9, 16}}));

  auto div = a / b;
  CHECK(array_equal(div, {{1, 1}, {1, 1}}));
}

TEST_CASE("example: dot operation") {
  auto x = array<float>{1, 2, 3};
  auto W = array<float>{{1, 2}, {3, 4}, {5, 6}};

  auto y = x.dot(W);
  CHECK(array_equal(y, {22, 28}));
}

