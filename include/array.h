#pragma once

#include <cpu.h>
#include <gpu.h>

#include <algorithm>
#include <concepts>
#include <format>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <unordered_set>

namespace sil {

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

using shape_type = std::vector<size_t>;
using strides_type = shape_type;

// Compute row-major contiguous strides from shape
inline strides_type contiguous_strides(const shape_type &shape) {
  if (shape.empty()) return {1};  // scalar: single stride of 1
  strides_type st(shape.size());
  st.back() = 1;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--)
    st[i] = st[i + 1] * shape[i + 1];
  return st;
}

// Compute broadcast output shape; throws on incompatible shapes
inline shape_type broadcast_shape(const shape_type &ls, const shape_type &rs) {
  if (ls == rs) return ls;
  if (ls.empty()) return rs;
  if (rs.empty()) return ls;
  if (ls.size() != rs.size()) {
    const auto &big = ls.size() > rs.size() ? ls : rs;
    const auto &small = ls.size() > rs.size() ? rs : ls;
    auto diff = big.size() - small.size();
    for (size_t i = 0; i < small.size(); i++) {
      if (small[i] != big[i + diff] && small[i] != 1 && big[i + diff] != 1)
        throw std::runtime_error("array: invalid operation.");
    }
    return big;
  }
  shape_type out(ls.size());
  for (size_t i = 0; i < ls.size(); i++) {
    if (ls[i] != rs[i] && ls[i] != 1 && rs[i] != 1)
      throw std::runtime_error("array: invalid operation.");
    out[i] = std::max(ls[i], rs[i]);
  }
  return out;
}

//------------------------------------------------------------------------------
// Expression templates for fusing element-wise float operations
//------------------------------------------------------------------------------

namespace expr {

struct leaf {
  const float *data;
  size_t len;
  const shape_type *shape_ptr;  // pointer to original array's shape
};

struct scalar {
  float val;
};

struct op_add {
  static float apply(float a, float b) { return a + b; }
};
struct op_sub {
  static float apply(float a, float b) { return a - b; }
};
struct op_mul {
  static float apply(float a, float b) { return a * b; }
};
struct op_div {
  static float apply(float a, float b) { return a / b; }
};

template <typename Op, typename L, typename R>
struct binary {
  L lhs;
  R rhs;
};

// Concept for expression nodes (leaf, scalar, or binary)
template <typename T>
concept node =
    std::same_as<std::remove_cvref_t<T>, leaf> ||
    std::same_as<std::remove_cvref_t<T>, scalar> ||
    requires(const T &t) { t.lhs; t.rhs; };

inline float eval_at(const leaf &e, size_t i) { return e.data[i % e.len]; }
inline float eval_at(const scalar &e, size_t) { return e.val; }
template <typename Op, typename L, typename R>
inline float eval_at(const binary<Op, L, R> &e, size_t i) {
  return Op::apply(eval_at(e.lhs, i), eval_at(e.rhs, i));
}

// Fast path: scalar access has no index dependency
inline float eval_at_direct(const scalar &e, size_t) { return e.val; }
// Fast path: leaf is guaranteed full-size when is_uniform is true
inline float eval_at_direct(const leaf &e, size_t i) { return e.data[i]; }
template <typename Op, typename L, typename R>
inline float eval_at_direct(const binary<Op, L, R> &e, size_t i) {
  return Op::apply(eval_at_direct(e.lhs, i), eval_at_direct(e.rhs, i));
}

// Check if all leaves have the same size (or are scalars)
inline bool is_uniform(const leaf &e, size_t n) { return e.len == n || e.len == 1; }
inline bool is_uniform(const scalar &, size_t) { return true; }
template <typename Op, typename L, typename R>
inline bool is_uniform(const binary<Op, L, R> &e, size_t n) {
  return is_uniform(e.lhs, n) && is_uniform(e.rhs, n);
}

inline size_t size_of(const leaf &e) { return e.len; }
inline size_t size_of(const scalar &) { return 1; }
template <typename Op, typename L, typename R>
inline size_t size_of(const binary<Op, L, R> &e) {
  return std::max(size_of(e.lhs), size_of(e.rhs));
}

// shape_of: returns the shape of the largest operand
inline const shape_type *shape_of(const leaf &e) { return e.shape_ptr; }
inline const shape_type *shape_of(const scalar &) { return nullptr; }
template <typename Op, typename L, typename R>
inline const shape_type *shape_of(const binary<Op, L, R> &e) {
  auto ls = size_of(e.lhs), rs = size_of(e.rhs);
  return ls >= rs ? shape_of(e.lhs) : shape_of(e.rhs);
}

// Operators between expression nodes
template <node L, node R>
auto operator+(const L &l, const R &r) { return binary<op_add, L, R>{l, r}; }
template <node L, node R>
auto operator-(const L &l, const R &r) { return binary<op_sub, L, R>{l, r}; }
template <node L, node R>
auto operator*(const L &l, const R &r) { return binary<op_mul, L, R>{l, r}; }
template <node L, node R>
auto operator/(const L &l, const R &r) { return binary<op_div, L, R>{l, r}; }

// Node + scalar
template <node L>
auto operator+(const L &l, float r) { return binary<op_add, L, scalar>{l, {r}}; }
template <node L>
auto operator-(const L &l, float r) { return binary<op_sub, L, scalar>{l, {r}}; }
template <node L>
auto operator*(const L &l, float r) { return binary<op_mul, L, scalar>{l, {r}}; }
template <node L>
auto operator/(const L &l, float r) { return binary<op_div, L, scalar>{l, {r}}; }

// Scalar + node
template <node R>
auto operator+(float l, const R &r) { return binary<op_add, scalar, R>{{l}, r}; }
template <node R>
auto operator-(float l, const R &r) { return binary<op_sub, scalar, R>{{l}, r}; }
template <node R>
auto operator*(float l, const R &r) { return binary<op_mul, scalar, R>{{l}, r}; }
template <node R>
auto operator/(float l, const R &r) { return binary<op_div, scalar, R>{{l}, r}; }

}  // namespace expr

//------------------------------------------------------------------------------
// Lazy evaluation: deferred element-wise operations for float arrays
//------------------------------------------------------------------------------

namespace detail {

struct lazy_node {
  enum class op {
    noop,
    // binary arithmetic
    add, sub, mul, div, pow,
    // unary
    sigmoid, relu, softmax,
    sigmoid_backward, layer_norm,
    // matrix
    dot, batched_dot, linear, linear_sigmoid, conv2d,
    // reduction
    sum_axis, mean_axis,
    // graph optimization
    affine,
  };

  op operation = op::noop;
  std::array<std::shared_ptr<lazy_node>, 3> inputs;
  uint8_t num_inputs = 0;

  storage data;
  shape_type shape;
  strides_type strides;
  bool evaluated = false;
  bool visited = false;

  struct params_t {
    float scale = 1.0f, offset = 0.0f;
    float eps = 0.0f;
    uint32_t axis = 0;
    uint32_t kernel_size = 0;
  } params;

  static std::shared_ptr<lazy_node> leaf(const storage &s,
                                         const shape_type &sh,
                                         const strides_type &st) {
    auto n = std::make_shared<lazy_node>();
    n->data = s;
    n->shape = sh;
    n->strides = st;
    n->evaluated = true;
    return n;
  }

  static std::shared_ptr<lazy_node> make(op o,
                                          std::initializer_list<std::shared_ptr<lazy_node>> ins,
                                          const shape_type &sh,
                                          const strides_type &st) {
    auto n = std::make_shared<lazy_node>();
    n->operation = o;
    n->num_inputs = static_cast<uint8_t>(ins.size());
    size_t i = 0;
    for (auto &inp : ins) n->inputs[i++] = inp;
    n->shape = sh;
    n->strides = st;
    return n;
  }
};

}  // namespace detail

using lazy_node = detail::lazy_node;

//------------------------------------------------------------------------------

template <value_type T, size_t I>
struct nested_initializer_list_ {
  using nested_type = nested_initializer_list_<T, I - 1>::type;
  using type = std::initializer_list<nested_type>;
};

template <value_type T>
struct nested_initializer_list_<T, 0> {
  using type = T;
};

template <value_type T, size_t I>
using nested_initializer_list = nested_initializer_list_<T, I>::type;

//------------------------------------------------------------------------------

// Compile-time helpers that extract the shape from a nested initializer list.
// Used by array<T>::array(nested_initializer_list<T, N>) constructors.

template <typename T>
struct depth_ {
  static constexpr size_t value = 0;
};

template <typename T>
struct depth_<std::initializer_list<T>> {
  static constexpr size_t value = 1 + depth_<T>::value;
};

template <size_t I>
struct shape_value_ {
  template <typename T>
  static constexpr size_t value(T l) {
    return l.size() == 0 ? 0 : shape_value_<I - 1>::value(*l.begin());
  }
};

template <>
struct shape_value_<0> {
  template <typename T>
  static constexpr size_t value(T l) {
    return l.size();
  }
};

template <typename T, size_t... I>
constexpr shape_type shape_(T l, std::index_sequence<I...>) {
  return {shape_type::value_type(shape_value_<I>::value(l))...};
}

template <typename T>
constexpr size_t nested_initializer_list_dimension_() {
  return depth_<T>::value;
};

template <typename T>
constexpr shape_type nested_initializer_list_shape_(T l) {
  return shape_<T>(
      l, std::make_index_sequence<nested_initializer_list_dimension_<T>()>());
}

//------------------------------------------------------------------------------

template <value_type T>
class array {
  template <value_type U> friend class array;

 public:
  array() = default;
  array(array &&rhs) = default;
  array(const array &rhs) = default;
  array &operator=(const array &rhs) = default;

  array(const shape_type &shape, T val);
  array(const shape_type &shape, std::input_iterator auto it);
  array(const shape_type &shape, std::ranges::input_range auto &&r);
  array(std::ranges::input_range auto &&r);
  array(T val);

  array(nested_initializer_list<T, 1> l);
  array(nested_initializer_list<T, 2> l);
  array(nested_initializer_list<T, 3> l);
  array(nested_initializer_list<T, 4> l);

  //----------------------------------------------------------------------------

  template <value_type U = T>
  array<U> clone() const;

  void eval() const { ensure_evaluated_(); }
  std::shared_ptr<lazy_node> lazy_node_ptr() const { return node_; }
  static void evaluate_single_node_(lazy_node *cur);

  //----------------------------------------------------------------------------

  array<bool> operator==(const array &rhs) const;
  array<bool> operator!=(const array &rhs) const;
  array<bool> operator>(const array &rhs) const;
  array<bool> operator<(const array &rhs) const;
  array<bool> operator>=(const array &rhs) const;
  array<bool> operator<=(const array &rhs) const;

  //----------------------------------------------------------------------------

  size_t buffer_element_count() const;
  size_t buffer_bytes() const;

  auto *buffer_data(this auto &&self);
  auto buffer_span(this auto &&self);

  //----------------------------------------------------------------------------

  size_t element_count() const;
  size_t length() const;

  size_t dimension() const;
  const shape_type &shape() const;
  const strides_type &strides() const;

  void reshape(const shape_type &shape);

  const auto broadcast(const shape_type &target_shape) const;

  array transpose() const;

  //----------------------------------------------------------------------------

  auto &at(this auto &&self);
  auto &at(this auto &&self, size_t i);

  auto &operator[](this auto &&self, size_t x, size_t y);
  auto &operator[](this auto &&self, size_t x, size_t y, size_t z);

  auto &at(this auto &&self, const std::vector<size_t> &position);

  template <size_t I>
  auto take() const;

  //----------------------------------------------------------------------------

  array operator[](size_t row) const;

  // Zero-copy slice: returns rows [start, start+count) sharing the same buffer
  array slice(size_t start, size_t count) const;

  //----------------------------------------------------------------------------

  auto element_begin();
  auto element_end();

  auto element_cbegin() const;
  auto element_cend() const;

  auto elements();
  auto elements() const;

  //----------------------------------------------------------------------------

  auto begin();
  auto end();

  auto begin() const;
  auto end() const;

  auto cbegin() const;
  auto cend() const;

  template <size_t N = 0>
  auto rows();

  template <size_t N = 0>
  auto rows() const;

  //----------------------------------------------------------------------------

  void set(std::input_iterator auto it);
  void set(std::initializer_list<T> l);

  //----------------------------------------------------------------------------

  void constants(T val);
  void zeros();
  void ones();
  void random();

  //----------------------------------------------------------------------------

  array operator+(const array &rhs) const;
  array operator-(const array &rhs) const;
  array operator*(const array &rhs) const;
  array operator/(const array &rhs) const;

  array pow(const array &rhs) const;

  void operator+=(const array &rhs);
  void operator-=(const array &rhs);
  void operator*=(const array &rhs);
  void operator/=(const array &rhs);

  //----------------------------------------------------------------------------

  array dot(const array &rhs) const;

  // Batched matmul: {B,M,K} × {B,K,N} → {B,M,N}. Single GPU dispatch.
  array batched_dot(const array &rhs) const;

  array linear(const array &W, const array &b) const;

  //----------------------------------------------------------------------------

  array<float> sigmoid() const;
  array<float> sigmoid_backward(const array<float> &dout) const;
  array<float> linear_sigmoid(const array &W, const array &b) const;
  array<float> relu() const;
  array<float> layer_norm(const array<float> &gamma, const array<float> &beta,
                          float eps = 1e-5f) const;

  array<float> conv2d(const array<float> &weight, size_t kernel_size) const;

  //----------------------------------------------------------------------------

  T sum() const;
  array sum(size_t axis) const;

  float mean() const;
  array<float> mean(size_t axis) const;

  T min() const;
  T max() const;

  size_t count() const;

  bool all(arithmetic auto val) const;
  template <typename U>
  bool all(U fn) const;

  array<float> softmax() const;
  auto argmax() const;

  float mean_square_error(const array &rhs) const;

  template <value_type U = T>
  array<U> one_hot(size_t class_count) const;

  //----------------------------------------------------------------------------

  std::string print_shape_type(const shape_type &shape) const;
  std::string print_shape() const;
  std::string print_strides() const;
  std::string print_data_type() const;
  std::string print_info() const;
  std::string print_array() const;

 private:
  shape_type shape_;
  strides_type strides_;
  storage storage_;
  std::shared_ptr<lazy_node> node_;

  //----------------------------------------------------------------------------

  void ensure_evaluated_() const;
  static void evaluate_node_(const std::shared_ptr<lazy_node> &node);
  static void eval_float_node_(lazy_node *node, auto store_result);
  static std::vector<lazy_node *> topo_sort_(std::initializer_list<lazy_node *> roots);
  static array make_lazy_op_(lazy_node::op o, const array &lhs, const array &rhs);
  static std::shared_ptr<lazy_node> to_node_(const array &a);
  static array from_lazy_node_(std::shared_ptr<lazy_node> node);

  static array from_node_(const std::shared_ptr<lazy_node> &n);

  void allocate_buffer_();

  static array make_uninit_(const shape_type &shape);

  array materialize_() const;

  //----------------------------------------------------------------------------

  void copy_initializer_list_(const auto &l);

  //----------------------------------------------------------------------------

  void bounds_check_(size_t i) const;
  void bounds_check_(size_t x, size_t y) const;
  void bounds_check_(size_t x, size_t y, size_t z) const;

  //----------------------------------------------------------------------------

  template <typename U>
  void enumerate_position_(size_t shape_index, std::vector<size_t> &position,
                           U fn) const;
  template <typename U>
  void enumerate_position_(U fn) const;

  //----------------------------------------------------------------------------

  static auto broadcast_(const array &lhs, const array &rhs, auto cb);

  template <value_type U = T>
  array<U> apply_binary_operation_(const array &rhs, auto ope) const;

  //----------------------------------------------------------------------------

  enum class ArithmeticOperation {
    Add = 0,
    Sub,
    Mul,
    Div,
    Pow,
  };

  static array lazy_or_eager_(lazy_node::op lo, ArithmeticOperation ao,
                              const array &lhs, const array &rhs);

  static void cpu_arithmetic_dispatch_(const storage &lhs, const storage &rhs,
                                       storage &dst, ArithmeticOperation ope);

  static void msl_arithmetic_dispatch_(const storage &lhs, const storage &rhs,
                                       storage &dst, ArithmeticOperation ope);

  static auto cpu_arithmetic_operation_(const array &lhs, const array &rhs,
                                        ArithmeticOperation ope);

  static auto msl_arithmetic_operation_(const array &lhs, const array &rhs,
                                        ArithmeticOperation ope);

  static auto arithmetic_operation_(const array &lhs, const array &rhs,
                                    ArithmeticOperation ope);

  void arithmetic_inplace_(const array &rhs, ArithmeticOperation ope);

  void cpu_arithmetic_inplace_(const array &rhs, ArithmeticOperation ope);

  //----------------------------------------------------------------------------

  static array cpu_dot_operation_(const array &lhs, const array &rhs);
  static array mps_dot_operation_(const array &lhs, const array &rhs);
  static bool steel_eligible_(size_t M, size_t N, size_t K, bool tL, bool tR);
  template <typename U>
  array dot_operation_(const array &rhs, U fn) const;
  template <typename U>
  static array dot_operation_helper_(const array &lhs, const array &rhs, U fn);

  template <typename CpuFn, typename GpuFn>
  array<float> unary_float_dispatch_(uint32_t op_id, CpuFn cpu_fn,
                                     GpuFn gpu_fn) const;
};

//----------------------------------------------------------------------------
// Iterators / range adapters
//
// These types are referenced from array<T>::element_begin(), rows(), etc.
// The array<T> member function implementations live in the Implementation
// section below, so these classes must be visible before then.
//----------------------------------------------------------------------------

template <value_type T, bool IsConst>
class element_iterator_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;

 public:
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::forward_iterator_tag;

  element_iterator_(arr_ptr arr, size_t i) : arr_(arr), i_(i) {}

  element_iterator_ &operator++() {
    ++i_;
    return *this;
  }

  element_iterator_ operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  decltype(auto) operator*() const { return arr_->at(i_); }

  bool operator==(const element_iterator_ &) const = default;

 private:
  arr_ptr arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T>
using element_iterator = element_iterator_<T, false>;
template <value_type T>
using const_element_iterator = element_iterator_<T, true>;

template <value_type T, bool IsConst>
struct element_range_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;
  element_range_(arr_ptr arr) : arr_(arr) {}
  auto begin() const { return element_iterator_<T, IsConst>(arr_, 0); }
  auto end() const {
    return element_iterator_<T, IsConst>(arr_, arr_->element_count());
  }
  arr_ptr arr_ = nullptr;
};

template <value_type T>
using element_range = element_range_<T, false>;
template <value_type T>
using const_element_range = element_range_<T, true>;

template <value_type T, bool IsConst>
class row_iterator_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;

 public:
  using difference_type = std::ptrdiff_t;
  using value_type = array<T>;
  using iterator_concept = std::forward_iterator_tag;

  row_iterator_(arr_ptr arr, size_t i) : arr_(arr), i_(i) {}

  row_iterator_ &operator++() {
    ++i_;
    return *this;
  }

  row_iterator_ operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  value_type operator*() const { return (*arr_)[i_]; }

  bool operator==(const row_iterator_ &) const = default;

 private:
  arr_ptr arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T>
using row_iterator = row_iterator_<T, false>;
template <value_type T>
using const_row_iterator = row_iterator_<T, true>;

template <value_type T, size_t N, bool IsConst>
class row_tuple_iterator_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;

 public:
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::forward_iterator_tag;

  row_tuple_iterator_(arr_ptr arr, size_t i) : arr_(arr), i_(i) {}

  row_tuple_iterator_ &operator++() {
    ++i_;
    return *this;
  }

  row_tuple_iterator_ operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  auto operator*() const { return (*arr_)[i_].template take<N>(); }

  bool operator==(const row_tuple_iterator_ &) const = default;

 private:
  arr_ptr arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T, size_t N>
using row_tuple_iterator = row_tuple_iterator_<T, N, false>;
template <value_type T, size_t N>
using const_row_tuple_iterator = row_tuple_iterator_<T, N, true>;

template <value_type T, bool IsConst>
struct row_range_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;
  row_range_(arr_ptr arr) : arr_(arr) {}
  auto begin() const { return row_iterator_<T, IsConst>(arr_, 0); }
  auto end() const {
    return row_iterator_<T, IsConst>(arr_, arr_->shape()[0]);
  }
  arr_ptr arr_ = nullptr;
};

template <value_type T>
using row_range = row_range_<T, false>;
template <value_type T>
using const_row_range = row_range_<T, true>;

template <value_type T, size_t N, bool IsConst>
struct row_tuple_range_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;
  row_tuple_range_(arr_ptr arr) : arr_(arr) {}
  auto begin() const { return row_tuple_iterator_<T, N, IsConst>(arr_, 0); }
  auto end() const {
    return row_tuple_iterator_<T, N, IsConst>(arr_, arr_->shape()[0]);
  }
  arr_ptr arr_ = nullptr;
};

template <value_type T, size_t N>
using row_tuple_range = row_tuple_range_<T, N, false>;
template <value_type T, size_t N>
using const_row_tuple_range = row_tuple_range_<T, N, true>;

//----------------------------------------------------------------------------

template <value_type T>
std::ostream &operator<<(std::ostream &os, const array<T> &arr);

//----------------------------------------------------------------------------

template <value_type T>
array<T> operator+(auto lhs, const array<T> &rhs);

template <value_type T>
array<T> operator-(auto lhs, const array<T> &rhs);

template <value_type T>
array<T> operator*(auto lhs, const array<T> &rhs);

template <value_type T>
array<T> operator/(auto lhs, const array<T> &rhs);

//----------------------------------------------------------------------------

template <value_type T, value_type U>
array<T> where(const array<U> &cond, T x, T y);

//----------------------------------------------------------------------------

template <value_type T>
bool array_equal(const array<T> &a, const array<T> &b);

template <std::floating_point T, std::floating_point U>
bool is_close(T a, U b, float tolerance = 1e-3);

template <std::integral T, std::integral U>
bool is_close(T a, U b);

template <value_type T>
bool allclose(const array<T> &a, const array<T> &b, float tolerance = 1e-3);

//----------------------------------------------------------------------------

template <value_type T>
auto empty(const shape_type &shape);

template <value_type T>
auto zeros(const shape_type &shape);

template <value_type T>
auto ones(const shape_type &shape);

auto random(const shape_type &shape);

//-----------------------------------------------------------------------------
// Implementation
//-----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::from_node_(const std::shared_ptr<lazy_node> &n) {
  array a;
  a.shape_ = n->shape;
  a.strides_ = n->strides;
  a.storage_ = n->data;
  return a;
}

template <value_type T>
inline bool array<T>::steel_eligible_(size_t M, size_t N, size_t K, bool tL, bool tR) {
  // STEEL supports edge tiles and K remainder. NN only.
  // Lowered N threshold from 64 to 16 — STEEL with edge tiles is still
  // faster than sgemm_32 for narrow matrices in practice.
  return !tL && !tR && M >= 16 && N >= 16 && K >= 16;
}

template <value_type T>
inline array<T>::array(const shape_type &shape, T val) {
  reshape(shape);
  allocate_buffer_();
  constants(val);
}

template <value_type T>
inline array<T>::array(const shape_type &shape, std::input_iterator auto it) {
  reshape(shape);
  allocate_buffer_();
  set(it);
}

template <value_type T>
inline array<T>::array(const shape_type &shape,
                       std::ranges::input_range auto &&r) {
  reshape(shape);
  allocate_buffer_();
  set(r.begin());
}

template <value_type T>
inline array<T>::array(std::ranges::input_range auto &&r) {
  size_t element_count = std::ranges::distance(r);
  reshape({element_count});
  allocate_buffer_();
  set(r.begin());
}

template <value_type T>
inline array<T>::array(T val) : array(shape_type({}), T{}) {
  *buffer_data() = val;
}

template <value_type T>
inline array<T>::array(nested_initializer_list<T, 1> l)
    : array(nested_initializer_list_shape_(l), T{}) {
  copy_initializer_list_(l);
}

template <value_type T>
inline array<T>::array(nested_initializer_list<T, 2> l)
    : array(nested_initializer_list_shape_(l), T{}) {
  copy_initializer_list_(l);
}

template <value_type T>
inline array<T>::array(nested_initializer_list<T, 3> l)
    : array(nested_initializer_list_shape_(l), T{}) {
  copy_initializer_list_(l);
}

template <value_type T>
inline array<T>::array(nested_initializer_list<T, 4> l)
    : array(nested_initializer_list_shape_(l), T{}) {
  copy_initializer_list_(l);
}

//----------------------------------------------------------------------------

template <value_type T>
template <value_type U>
inline array<U> array<T>::clone() const {
  auto tmp = array<U>(shape_, U{});
  for (size_t i = 0; i < element_count(); i++) {
    tmp.at(i) = static_cast<U>(at(i));
  }
  return tmp;
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<bool> array<T>::operator==(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs == rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator!=(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs != rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator>(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs > rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator>=(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs >= rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator<(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs < rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator<=(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs <= rhs; });
}

//----------------------------------------------------------------------------

template <value_type T>
inline size_t array<T>::buffer_element_count() const {
  ensure_evaluated_();
  return storage_.len;
}

template <value_type T>
inline size_t array<T>::buffer_bytes() const {
  ensure_evaluated_();
  return storage_.len * sizeof(T);
}

template <value_type T>
inline auto *array<T>::buffer_data(this auto &&self) {
  self.ensure_evaluated_();
  if (gpu_pending_) gpu_context::instance().flush();
  constexpr bool is_const =
      std::is_const_v<std::remove_reference_t<decltype(self)>>;
  using ptr_type = std::conditional_t<is_const, const T *, T *>;
  return static_cast<ptr_type>(self.storage_.data) + self.storage_.off;
}

template <value_type T>
inline auto array<T>::buffer_span(this auto &&self) {
  return std::span(self.buffer_data(), self.buffer_element_count());
}

//----------------------------------------------------------------------------

template <value_type T>
inline size_t array<T>::element_count() const {
  size_t count = 1;
  for (auto n : shape_) {
    count *= n;
  }
  return count;
}

template <value_type T>
inline size_t array<T>::length() const {
  if (shape_.empty()) {
    throw std::runtime_error("array: cannot call with a scalar value.");
  }
  return shape_[0];
}

template <value_type T>
inline size_t array<T>::dimension() const {
  return shape_.size();
}

template <value_type T>
inline const shape_type &array<T>::shape() const {
  return shape_;
}

template <value_type T>
inline const strides_type &array<T>::strides() const {
  return strides_;
}

template <value_type T>
inline void array<T>::reshape(const shape_type &shape) {
  ensure_evaluated_();
  shape_ = shape;
  strides_ = contiguous_strides(shape);
}

template <value_type T>
inline const auto array<T>::broadcast(const shape_type &target_shape) const {
  ensure_evaluated_();

  if (target_shape.size() < dimension()) {
    throw std::runtime_error("array: invalid shape for broadcast.");
  }

  auto diff = target_shape.size() - dimension();
  for (size_t i = 0; i < dimension(); i++) {
    if (shape_[i] != target_shape[i + diff] && shape_[i] != 1) {
      throw std::runtime_error("array: invalid shape for broadcast.");
    }
  }

  if (target_shape == shape_) {
    return *this;
  }

  array tmp = *this;
  tmp.node_.reset();
  tmp.shape_ = target_shape;

  // Build strides: start from source strides, prepend zeros for new dims,
  // and zero out dims where source size is 1 (broadcast).
  tmp.strides_.resize(target_shape.size(), 0);
  for (size_t i = 0; i < target_shape.size(); i++) {
    int src_axis = static_cast<int>(i) - static_cast<int>(diff);
    if (src_axis < 0) {
      // New dimension added by higher-dim broadcast
      tmp.strides_[i] = 0;
    } else if (src_axis < static_cast<int>(dimension()) &&
               shape_[src_axis] == 1 && target_shape[i] != 1) {
      // Source dim is 1, broadcast to target size
      tmp.strides_[i] = 0;
    } else if (src_axis < static_cast<int>(dimension())) {
      // Keep original stride
      tmp.strides_[i] = strides_[src_axis];
    }
  }
  return tmp;
}

template <value_type T>
inline array<T> array<T>::transpose() const {
  ensure_evaluated_();

  if (dimension() == 1) {
    auto tmp = *this;
    tmp.node_.reset();
    tmp.shape_ = {1, element_count()};
    tmp.strides_ = {static_cast<size_t>(element_count()), 1};
    return tmp;
  }

  if (dimension() == 2) {
    if (shape_[0] == 1) {
      auto tmp = *this;
      tmp.node_.reset();
      tmp.shape_ = {shape_[1]};
      tmp.strides_ = {1};
      return tmp;
    }
    // Zero-copy: cblas_sgemm / MPS handle transposed strides natively
    auto tmp = *this;
    tmp.node_.reset();
    tmp.shape_ = {shape_[1], shape_[0]};
    tmp.strides_ = {strides_[1], strides_[0]};
    return tmp;
  }

  // 3D+: copy fallback (stride-only view not worth the complexity)
  if (dimension() == 3) {
    auto shape = shape_;
    std::ranges::reverse(shape);

    auto tmp = make_uninit_(shape);

    auto it = element_cbegin();
    for (size_t z = 0; z < shape[2]; z++) {
      for (size_t y = 0; y < shape[1]; y++) {
        for (size_t x = 0; x < shape[0]; x++) {
          tmp[x, y, z] = *it;
          ++it;
        }
      }
    }
    return tmp;
  }

  throw std::runtime_error("array: can't do `transpose` operation.");
}

//----------------------------------------------------------------------------

template <value_type T>
inline auto &array<T>::at(this auto &&self) {
  return *self.buffer_data();
}

template <value_type T>
inline auto &array<T>::at(this auto &&self, size_t i) {
  self.bounds_check_(i);
  // Transposed 2D views have non-contiguous strides
  if (self.dimension() == 2 && self.strides_[0] != self.shape_[1]) {
    auto row = i / self.shape_[1];
    auto col = i % self.shape_[1];
    return self.buffer_data()[row * self.strides_[0] + col * self.strides_[1]];
  }
  return self.buffer_data()[i % self.buffer_element_count()];
}

template <value_type T>
inline auto &array<T>::operator[](this auto &&self, size_t x, size_t y) {
  self.bounds_check_(x, y);
  return self.buffer_data()[self.strides_[0] * x + self.strides_[1] * y];
}

template <value_type T>
inline auto &
array<T>::operator[](this auto &&self, size_t x, size_t y, size_t z) {
  self.bounds_check_(x, y, z);
  return self.buffer_data()[(self.strides_[0] * x) + (self.strides_[1] * y) +
                            z];
}

template <value_type T>
inline auto &array<T>::at(this auto &&self,
                          const std::vector<size_t> &position) {
  size_t buffer_index = 0;
  for (size_t i = 0; i < position.size(); i++) {
    buffer_index += self.strides_[i] * position[i];
  }
  return self.buffer_data()[buffer_index];
}

template <value_type T>
template <size_t I>
inline auto array<T>::take() const {
  if constexpr (I == 0) {
    return std::tuple<>();
  } else {
    auto t = take<I - 1>();
    return std::tuple_cat(t, std::tuple<T>(at(I - 1)));
  }
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::operator[](size_t row) const {
  if (dimension() == 0 || row >= shape_[0]) {
    throw std::runtime_error("array: row is out of bounds.");
  }

  ensure_evaluated_();
  array tmp(*this);
  tmp.node_.reset();

  auto s = shape();
  s.erase(s.begin());
  tmp.reshape(s);

  auto stride = strides_[0];
  tmp.storage_.off = storage_.off + stride * row;
  tmp.storage_.len = stride;
  return tmp;
}

template <value_type T>
inline array<T> array<T>::slice(size_t start, size_t count) const {
  ensure_evaluated_();
  array tmp(*this);
  tmp.node_.reset();
  tmp.shape_[0] = count;
  auto stride = strides_[0];
  tmp.storage_.off = storage_.off + stride * start;
  tmp.storage_.len = count * stride;
  return tmp;
}

template <value_type T>
inline auto array<T>::element_begin() {
  return element_iterator_<T, false>(this, 0);
}

template <value_type T>
inline auto array<T>::element_end() {
  return element_iterator_<T, false>(this, element_count());
}

template <value_type T>
inline auto array<T>::element_cbegin() const {
  return element_iterator_<T, true>(this, 0);
}

template <value_type T>
inline auto array<T>::element_cend() const {
  return element_iterator_<T, true>(this, element_count());
}

template <value_type T>
inline auto array<T>::elements() {
  return element_range_<T, false>(this);
}

template <value_type T>
inline auto array<T>::elements() const {
  return element_range_<T, true>(this);
}

template <value_type T>
inline auto array<T>::begin() {
  return row_iterator_<T, false>(this, 0);
}

template <value_type T>
inline auto array<T>::end() {
  return row_iterator_<T, false>(this, shape_[0]);
}

template <value_type T>
inline auto array<T>::begin() const {
  return row_iterator_<T, true>(this, 0);
}

template <value_type T>
inline auto array<T>::end() const {
  return row_iterator_<T, true>(this, shape_[0]);
}

template <value_type T>
inline auto array<T>::cbegin() const {
  return begin();
}
template <value_type T>
inline auto array<T>::cend() const {
  return end();
}

template <value_type T>
template <size_t N>
inline auto array<T>::rows() {
  if constexpr (N == 0) {
    return row_range_<T, false>(this);
  } else {
    return row_tuple_range_<T, N, false>(this);
  }
}

template <value_type T>
template <size_t N>
inline auto array<T>::rows() const {
  if constexpr (N == 0) {
    return row_range_<T, true>(this);
  } else {
    return row_tuple_range_<T, N, true>(this);
  }
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::set(std::input_iterator auto it) {
  for (size_t i = 0; i < element_count(); i++) {
    at(i) = *it++;
  }
  node_.reset();
}

template <value_type T>
inline void array<T>::set(std::initializer_list<T> l) {
  std::ranges::copy(l, element_begin());
  node_.reset();
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::constants(T val) {
  std::ranges::fill(buffer_span(), val);
  node_.reset();
}

template <value_type T>
inline void array<T>::zeros() {
  constants(0);
};

template <value_type T>
inline void array<T>::ones() {
  constants(1);
};

template <value_type T>
inline void array<T>::random() {
  thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::ranges::generate(buffer_span(), [&]() { return dist(gen); });
  node_.reset();
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::operator+(const array &rhs) const {
  return lazy_or_eager_(lazy_node::op::add, ArithmeticOperation::Add, *this, rhs);
}

template <value_type T>
inline array<T> array<T>::operator-(const array &rhs) const {
  return lazy_or_eager_(lazy_node::op::sub, ArithmeticOperation::Sub, *this, rhs);
}

template <value_type T>
inline array<T> array<T>::operator*(const array &rhs) const {
  return lazy_or_eager_(lazy_node::op::mul, ArithmeticOperation::Mul, *this, rhs);
}

template <value_type T>
inline array<T> array<T>::operator/(const array &rhs) const {
  return lazy_or_eager_(lazy_node::op::div, ArithmeticOperation::Div, *this, rhs);
}

template <value_type T>
inline array<T> array<T>::pow(const array &rhs) const {
  return lazy_or_eager_(lazy_node::op::pow, ArithmeticOperation::Pow, *this, rhs);
}

template <value_type T>
inline void array<T>::operator+=(const array &rhs) {
  arithmetic_inplace_(rhs, ArithmeticOperation::Add);
}

template <value_type T>
inline void array<T>::operator-=(const array &rhs) {
  arithmetic_inplace_(rhs, ArithmeticOperation::Sub);
}

template <value_type T>
inline void array<T>::operator*=(const array &rhs) {
  arithmetic_inplace_(rhs, ArithmeticOperation::Mul);
}

template <value_type T>
inline void array<T>::operator/=(const array &rhs) {
  arithmetic_inplace_(rhs, ArithmeticOperation::Div);
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::batched_dot(const array &rhs) const {
  static_assert(std::same_as<T, float>, "batched_dot requires float");

  if (dimension() != 3 || rhs.dimension() != 3)
    throw std::runtime_error("batched_dot: requires 3D arrays");
  if (shape_[0] != rhs.shape_[0] || shape_[2] != rhs.shape_[1])
    throw std::runtime_error("batched_dot: shape mismatch");

  auto out_shape = shape_type{shape_[0], shape_[1], rhs.shape_[2]};
  auto node = lazy_node::make(lazy_node::op::batched_dot,
                              {to_node_(*this), to_node_(rhs)},
                              out_shape, contiguous_strides(out_shape));
  return from_lazy_node_(std::move(node));
}

template <value_type T>
inline array<T> array<T>::dot(const array &rhs) const {
  if constexpr (std::same_as<T, float>) {
    // Compute output shape at build time (with validation)
    shape_type out_shape;
    if (dimension() == 2 && rhs.dimension() == 2 && shape_[1] == rhs.shape_[0]) {
      out_shape = {shape_[0], rhs.shape_[1]};
    } else if (dimension() == 1 && rhs.dimension() == 2 && shape_[0] == rhs.shape_[0]) {
      out_shape = {rhs.shape_[1]};
    } else if (dimension() == 2 && rhs.dimension() == 1 && shape_[1] == rhs.shape_[0]) {
      out_shape = {shape_[0]};
    } else if (dimension() == 1 && rhs.dimension() == 1 && shape_[0] == rhs.shape_[0]) {
      out_shape = {};
    } else {
      throw std::runtime_error("array: can't do `dot` operation.");
    }
    auto node = lazy_node::make(lazy_node::op::dot,
                                {to_node_(*this), to_node_(rhs)},
                                out_shape, contiguous_strides(out_shape));
    return from_lazy_node_(std::move(node));
  } else {
    switch (device_) {
      case Device::MPS:
        return dot_operation_(rhs, mps_dot_operation_);
      case Device::CPU:
        return dot_operation_(rhs, cpu_dot_operation_);
    }
  }
}

template <value_type T>
inline array<T> array<T>::linear(const array &W, const array &b) const {
  if constexpr (std::same_as<T, float>) {
    auto out_shape = shape_type{shape_[0], W.shape_[1]};
    auto node = lazy_node::make(lazy_node::op::linear,
                                {to_node_(*this), to_node_(W), to_node_(b)},
                                out_shape, contiguous_strides(out_shape));
    return from_lazy_node_(std::move(node));
  } else {
    return dot(W) + b;
  }
}

//----------------------------------------------------------------------------

template <value_type T>
template <typename CpuFn, typename GpuFn>
inline array<float> array<T>::unary_float_dispatch_(uint32_t op_id,
                                                    CpuFn cpu_fn,
                                                    GpuFn gpu_fn) const {
  ensure_evaluated_();
  if constexpr (!std::same_as<T, float>) {
    return cpu_fn();
  } else {
    switch (device_) {
      case Device::CPU: return cpu_fn();
      case Device::MPS: return gpu_fn();
    }
  }
}

template <value_type T>
inline array<float> array<T>::sigmoid() const {
  if constexpr (std::same_as<T, float>) {
    auto node = lazy_node::make(lazy_node::op::sigmoid, {to_node_(*this)},
                                shape_, contiguous_strides(shape_));
    return from_lazy_node_(std::move(node));
  } else {
    auto src = this->template clone<float>();
    return src.sigmoid();
  }
}

template <value_type T>
inline array<float> array<T>::sigmoid_backward(const array<float> &dout) const {
  if constexpr (std::same_as<T, float>) {
    auto node = lazy_node::make(lazy_node::op::sigmoid_backward,
                                {to_node_(*this), array<float>::to_node_(dout)},
                                shape_, contiguous_strides(shape_));
    return from_lazy_node_(std::move(node));
  } else {
    auto src = this->template clone<float>();
    return src.sigmoid_backward(dout);
  }
}

template <value_type T>
inline array<float> array<T>::linear_sigmoid(const array &W, const array &b) const {
  if constexpr (std::same_as<T, float>) {
    auto out_shape = shape_type{shape_[0], W.shape_[1]};
    auto node = lazy_node::make(lazy_node::op::linear_sigmoid,
                                {to_node_(*this), to_node_(W), to_node_(b)},
                                out_shape, contiguous_strides(out_shape));
    return from_lazy_node_(std::move(node));
  } else {
    return linear(W, b).sigmoid();
  }
}

template <value_type T>
inline array<float> array<T>::relu() const {
  if constexpr (std::same_as<T, float>) {
    auto node = lazy_node::make(lazy_node::op::relu, {to_node_(*this)},
                                shape_, contiguous_strides(shape_));
    return from_lazy_node_(std::move(node));
  } else {
    auto src = this->template clone<float>();
    return src.relu();
  }
}

template <value_type T>
inline array<float> array<T>::layer_norm(const array<float> &gamma,
                                         const array<float> &beta,
                                         float eps) const {
  if (dimension() != 2)
    throw std::runtime_error("array: layer_norm requires 2D array.");

  if constexpr (std::same_as<T, float>) {
    auto node = lazy_node::make(lazy_node::op::layer_norm,
                                {to_node_(*this),
                                 array<float>::to_node_(gamma),
                                 array<float>::to_node_(beta)},
                                shape_, contiguous_strides(shape_));
    node->params.eps = eps;
    return from_lazy_node_(std::move(node));
  } else {
    auto src = this->template clone<float>();
    return src.layer_norm(gamma, beta, eps);
  }
}

template <value_type T>
inline array<float> array<T>::conv2d(const array<float> &weight, size_t kernel_size) const {
  static_assert(std::same_as<T, float>, "conv2d requires float");

  auto N = shape_[0], H = shape_[1], W = shape_[2];
  auto C_out = weight.shape_[0];
  auto out_shape = shape_type{N, H, W, C_out};
  auto node = lazy_node::make(lazy_node::op::conv2d,
                              {to_node_(*this), array<float>::to_node_(weight)},
                              out_shape, contiguous_strides(out_shape));
  node->params.kernel_size = static_cast<uint32_t>(kernel_size);
  return from_lazy_node_(std::move(node));
}

//----------------------------------------------------------------------------

template <value_type T>
inline T array<T>::sum() const {
  ensure_evaluated_();
  auto cpu_sum = [&]() -> T {
    auto sp = buffer_span();
    if (sp.size() == element_count()) {
      return cpu::sum<T>(sp.data(), sp.size());
    }
    return std::accumulate(element_cbegin(), element_cend(), T{});
  };

  if constexpr (!std::same_as<T, float>) {
    return cpu_sum();
  } else {
    auto gpu_sum = [&]() -> float {
      auto n = element_count();
      size_t num_tg = gpu::sum_f32_num_tg(n);
      auto partial = array<float>::make_uninit_({num_tg});
      size_t actual_tg = gpu::sum_f32(storage_, partial.storage_, n);
      synchronize();
      return cpu::sum<float>(partial.buffer_data(), actual_tg);
    };

    constexpr size_t kGpuSumThreshold = 5'000'000;
    switch (device_) {
      case Device::CPU: return cpu_sum();
      case Device::MPS: return gpu_sum();
    }
  }
}

template <value_type T>
inline array<T> array<T>::sum(size_t axis) const {
  auto s = shape_;
  s.erase(s.begin() + axis);

  if constexpr (std::same_as<T, float>) {
    auto node = lazy_node::make(lazy_node::op::sum_axis, {to_node_(*this)},
                                s, contiguous_strides(s));
    node->params.axis = static_cast<uint32_t>(axis);
    return from_lazy_node_(std::move(node));
  }

  ensure_evaluated_();
  auto tmp = array(s, T{});

  if (dimension() == 2 && axis == 0 &&
      buffer_element_count() == element_count()) {
    cpu::sum_axis0<T>(buffer_data(), tmp.buffer_data(),
                      shape_[0], shape_[1]);
    return tmp;
  }

  enumerate_position_([&](const auto &pos) {
    auto p = pos;
    p.erase(p.begin() + axis);
    tmp.at(p) += at(pos);
  });

  return tmp;
}

template <value_type T>
inline float array<T>::mean() const {
  return sum() / static_cast<float>(element_count());
}

template <value_type T>
inline array<float> array<T>::mean(size_t axis) const {
  auto t = sum(axis);
  if constexpr (std::is_same_v<T, float>) {
    return t / shape_[axis];
  } else {
    return t.template clone<float>() / shape_[axis];
  }
}

template <value_type T>
inline T array<T>::min() const {
  ensure_evaluated_();
  if constexpr (std::same_as<T, float>) {
    auto cpu_min = [&]() -> float {
      float result;
      vDSP_Length index;
      vDSP_minvi(buffer_data(), 1, &result, &index, buffer_element_count());
      return result;
    };
    auto gpu_min = [&]() -> float {
      auto n = buffer_element_count();
      size_t num_tg = gpu::min_f32_num_tg(n);
      auto partial = array<float>::make_uninit_({num_tg});
      size_t actual_tg = gpu::min_f32(storage_, partial.storage_, n);
      synchronize();
      float result = partial.buffer_data()[0];
      for (size_t i = 1; i < actual_tg; i++)
        result = std::min(result, partial.buffer_data()[i]);
      return result;
    };
    switch (device_) {
      case Device::CPU: return cpu_min();
      case Device::MPS: return gpu_min();
    }
  } else {
    return *std::ranges::min_element(buffer_span());
  }
}

template <value_type T>
inline T array<T>::max() const {
  ensure_evaluated_();
  if constexpr (std::same_as<T, float>) {
    auto cpu_max = [&]() -> float {
      float result;
      vDSP_Length index;
      vDSP_maxvi(buffer_data(), 1, &result, &index, buffer_element_count());
      return result;
    };
    auto gpu_max = [&]() -> float {
      auto n = buffer_element_count();
      size_t num_tg = gpu::max_f32_num_tg(n);
      auto partial = array<float>::make_uninit_({num_tg});
      size_t actual_tg = gpu::max_f32(storage_, partial.storage_, n);
      synchronize();
      float result = partial.buffer_data()[0];
      for (size_t i = 1; i < actual_tg; i++)
        result = std::max(result, partial.buffer_data()[i]);
      return result;
    };
    switch (device_) {
      case Device::CPU: return cpu_max();
      case Device::MPS: return gpu_max();
    }
  } else {
    return *std::ranges::max_element(buffer_span());
  }
}

template <value_type T>
inline size_t array<T>::count() const {
  return std::ranges::count_if(buffer_span(), [](T v) { return !!v; });
}

template <value_type T>
inline bool array<T>::all(arithmetic auto val) const {
  return std::ranges::all_of(buffer_span(), [val](T v) { return v == val; });
}

template <value_type T>
template <typename U>
inline bool array<T>::all(U fn) const {
  return std::ranges::all_of(buffer_span(), fn);
}

template <value_type T>
inline array<float> array<T>::softmax() const {
  if (dimension() == 1) {
    ensure_evaluated_();
    auto c = min();
    auto tmp = array<float>(shape_, 0.0);
    for (size_t i = 0; i < element_count(); i++) {
      tmp.at(i) = std::exp(at(i) - c);
    }
    return tmp / tmp.sum();
  } else if (dimension() == 2) {
    if constexpr (std::same_as<T, float>) {
      auto node = lazy_node::make(lazy_node::op::softmax, {to_node_(*this)},
                                  shape_, contiguous_strides(shape_));
      return from_lazy_node_(std::move(node));
    } else {
      ensure_evaluated_();
      auto tmp = array<float>(shape_, 0.0);
      for (size_t i = 0; i < shape_[0]; i++) {
        const auto row = (*this)[i];
        auto c = row.min();
        for (size_t j = 0; j < row.element_count(); j++)
          tmp[i].at(j) = std::exp(row.at(j) - c);
        auto smax = tmp[i] / tmp[i].sum();
        for (size_t j = 0; j < row.element_count(); j++)
          tmp[i].at(j) = smax.at(j);
      }
      return tmp;
    }
  }

  throw std::runtime_error(
      "array: softmax is available only for 1 or 2 dimension array.");
}

template <value_type T>
inline auto array<T>::argmax() const {
  if (dimension() == 2) {
    ensure_evaluated_();
    auto row_count = shape_[0];
    auto col_count = shape_[1];

    if constexpr (std::same_as<T, float>) {
      if (device_ == Device::MPS) {
        auto tmp = array<int>::make_uninit_({row_count});
        gpu::argmax_f32(storage_, tmp.storage_,
                        static_cast<uint32_t>(row_count),
                        static_cast<uint32_t>(col_count));
        return tmp;
      }
    }

    auto tmp = array<int>({row_count}, 0);
    for (size_t i = 0; i < row_count; i++) {
      const auto row = (*this)[i];

      size_t max_index = 0;
      {
        T max_val = std::numeric_limits<T>::min();
        for (size_t j = 0; j < row.buffer_element_count(); j++) {
          auto val = row.buffer_data()[j];
          if (val > max_val) {
            max_val = val;
            max_index = j;
          }
        }
      }

      tmp.at(i) = max_index;
    }
    return tmp;
  }

  throw std::runtime_error("array: argmax is available for 2 dimension array.");
}

template <value_type T>
inline float array<T>::mean_square_error(const array &rhs) const {
  return (*this - rhs).pow(2).mean();
}

template <value_type T>
template <value_type U>
inline array<U> array<T>::one_hot(size_t class_count) const {
  if (dimension() == 1) {
    auto tmp = array<U>({shape_[0], class_count}, U{});
    for (size_t i = 0; i < element_count(); i++) {
      tmp[i, at(i)] = 1;
    }
    return tmp;
  }

  throw std::runtime_error(
      "array: one_hot is available only for 1 dimension array.");
}

//----------------------------------------------------------------------------

template <value_type T>
inline std::string array<T>::print_shape_type(const shape_type &shape) const {
  std::string result = "{";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != 0) result += ", ";
    result += std::format("{}", shape[i]);
  }
  result += "}";
  return result;
}

template <value_type T>
inline std::string array<T>::print_shape() const {
  return print_shape_type(shape_);
}

template <value_type T>
inline std::string array<T>::print_strides() const {
  return print_shape_type(strides_);
}

template <value_type T>
inline std::string array<T>::print_data_type() const {
  if constexpr (std::is_same_v<T, float>) {
    return "float";
  } else {
    return "int";
  }
}

template <value_type T>
inline std::string array<T>::print_info() const {
  return std::format("dtype: {}, dim: {}, shape: {}, strides: {}",
                     print_data_type(), dimension(), print_shape(),
                     print_strides());
}

template <value_type T>
inline std::string array<T>::print_array() const {
  auto loop = [&](auto self, auto &os, auto dim, auto arr_index) {
    auto n = shape_[dim];
    if (dim + 1 == dimension()) {
      for (size_t i = 0; i < n; i++, arr_index++) {
        if (i > 0) {
          os << ", ";
        }
        if constexpr (std::is_same_v<T, bool>) {
          os << (at(arr_index) ? "true" : "false");
        } else {
          os << at(arr_index);
        }
      }
      return arr_index;
    }

    for (size_t i = 0; i < n; i++) {
      if (dim < dimension() && i > 0) {
        os << ",\n";
        if (dimension() >= 3 && dim == 0 && i > 0) {
          os << "\n";
        }
        for (size_t j = 0; j <= dim; j++) {
          os << " ";
        }
      }
      os << '{';
      arr_index = self(self, os, dim + 1, arr_index);
      os << '}';
    }
    return arr_index;
  };

  std::stringstream ss;
  if (dimension() == 0) {
    ss << at();
  } else {
    ss << '{';
    loop(loop, ss, 0, 0);
    ss << '}';
  }
  return ss.str();
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::make_uninit_(const shape_type &shape) {
  array a;
  a.reshape(shape);
  a.allocate_buffer_();
  return a;
}

template <value_type T>
inline void array<T>::ensure_evaluated_() const {
  if (!node_) return;
  if (!node_->evaluated) evaluate_node_(node_);
  // node_ is shared_ptr: another copy of this array may have triggered evaluation
  if (!storage_.data) {
    auto &self = const_cast<array &>(*this);
    self.shape_ = node_->shape;
    self.strides_ = node_->strides;
    self.storage_ = node_->data;
  }
}

namespace detail {

inline float scalar_val(const storage &s) {
  return static_cast<const float *>(s.data)[s.off];
}

// Reduce a chain of scalar ops on a vector to out[i] = vec[i] * scale + offset.
// Returns false if the tree contains non-scalar operands (falls back to per-node).
inline bool try_affine_reduce(const lazy_node &node,
                              const float *&vec_ptr, size_t &vec_len,
                              const storage *&vec_storage,
                              float &scale, float &offset) {
  if (node.evaluated) {
    vec_ptr = static_cast<const float *>(node.data.data) + node.data.off;
    vec_len = node.data.len;
    vec_storage = &node.data;
    scale = 1.0f;
    offset = 0.0f;
    return true;
  }

  if (node.num_inputs < 2) return false;
  auto &inp0 = node.inputs[0];
  auto &inp1 = node.inputs[1];

  // Only add/sub/mul/div can be affine-fused
  if (node.operation > lazy_node::op::div) return false;

  auto rhs_scalar = inp1->evaluated && inp1->data.len == 1;
  auto lhs_scalar = inp0->evaluated && inp0->data.len == 1;

  if (rhs_scalar) {
    if (!try_affine_reduce(*inp0, vec_ptr, vec_len, vec_storage, scale, offset))
      return false;
    auto r = scalar_val(inp1->data);
    switch (node.operation) {
      case lazy_node::op::add: offset += r; break;
      case lazy_node::op::sub: offset -= r; break;
      case lazy_node::op::mul: scale *= r; offset *= r; break;
      case lazy_node::op::div: scale /= r; offset /= r; break;
      default: return false;
    }
    return true;
  }

  if (lhs_scalar) {
    if (!try_affine_reduce(*inp1, vec_ptr, vec_len, vec_storage, scale, offset))
      return false;
    auto l = scalar_val(inp0->data);
    switch (node.operation) {
      case lazy_node::op::add: offset += l; break;
      case lazy_node::op::sub: scale = -scale; offset = l - offset; break;
      case lazy_node::op::mul: scale *= l; offset *= l; break;
      case lazy_node::op::div: return false;  // scalar / chain is not affine
      default: return false;
    }
    return true;
  }

  return false;
}

}  // namespace detail

template <value_type T>
inline std::vector<lazy_node *> array<T>::topo_sort_(
    std::initializer_list<lazy_node *> roots) {
  std::vector<lazy_node *> order;
  std::vector<std::pair<lazy_node *, size_t>> stack;
  for (auto *r : roots) {
    if (r && !r->evaluated && !r->visited) {
      r->visited = true;
      stack.push_back({r, 0});
    }
  }
  while (!stack.empty()) {
    auto &[cur, idx] = stack.back();
    if (cur->evaluated) { stack.pop_back(); continue; }
    if (idx < cur->num_inputs) {
      auto *child = cur->inputs[idx].get();
      idx++;
      if (child && !child->evaluated && !child->visited) {
        child->visited = true;
        stack.push_back({child, 0});
      }
    } else {
      order.push_back(cur);
      stack.pop_back();
    }
  }
  return order;
}

template <value_type T>
inline void array<T>::evaluate_node_(const std::shared_ptr<lazy_node> &node) {
  if (node->evaluated) return;

  auto order = topo_sort_({node.get()});

  for (auto *cur : order) {
    if (cur->evaluated) continue;
    evaluate_single_node_(cur);
  }

  for (auto *cur : order) cur->visited = false;
}

template <value_type T>
inline void array<T>::evaluate_single_node_(lazy_node *cur) {
  // Try affine fusion for float
  if constexpr (std::same_as<T, float>) {
    const float *vec_ptr = nullptr;
    auto vec_len = size_t{0};
    const storage *vec_st = nullptr;
    auto scale = 1.0f, offset = 0.0f;

    if (detail::try_affine_reduce(*cur, vec_ptr, vec_len, vec_st, scale, offset)) {
      auto result = make_uninit_(cur->shape);
      auto n = result.element_count();

      if (gpu_pending_ && vec_st->mtl_buf) {
        gpu::affine(*vec_st, result.storage_, n, scale, offset);
      } else {
        auto *out = result.buffer_data();
        constexpr size_t kNeonAffineThreshold = 5'000'000;
        if (n >= kNeonAffineThreshold) {
          cpu::affine(vec_ptr, out, n, scale, offset);
        } else if (scale == 1.0f && offset == 0.0f) {
          memcpy(out, vec_ptr, n * sizeof(float));
        } else if (offset == 0.0f) {
          vDSP_vsmul(vec_ptr, 1, &scale, out, 1, n);
        } else if (scale == 1.0f) {
          vDSP_vsadd(vec_ptr, 1, &offset, out, 1, n);
        } else {
          vDSP_vsmsa(vec_ptr, 1, &scale, &offset, out, 1, n);
        }
      }

      cur->data = result.storage_;
      cur->strides = result.strides_;
      cur->evaluated = true;
      return;
    }
  }

  // Normal dispatch
  auto store_result = [&](array result) {
    if (result.node_ && !result.node_->evaluated) result.ensure_evaluated_();
    cur->data = result.storage_;
    cur->shape = result.shape_;
    cur->strides = result.strides_;
    cur->evaluated = true;
  };

  using op = lazy_node::op;
  switch (cur->operation) {
    case op::add: case op::sub: case op::mul: case op::div: case op::pow: {
      auto lhs = from_node_(cur->inputs[0]);
      auto rhs = from_node_(cur->inputs[1]);
      auto ope = ArithmeticOperation::Add;
      switch (cur->operation) {
        case op::add: ope = ArithmeticOperation::Add; break;
        case op::sub: ope = ArithmeticOperation::Sub; break;
        case op::mul: ope = ArithmeticOperation::Mul; break;
        case op::div: ope = ArithmeticOperation::Div; break;
        case op::pow: ope = ArithmeticOperation::Pow; break;
        default: break;
      }
      store_result(arithmetic_operation_(lhs, rhs, ope));
      break;
    }
    default:
      if constexpr (std::same_as<T, float>) {
        eval_float_node_(cur, store_result);
      }
      break;
  }

  // Release input references to allow natural memory reclamation
  for (uint8_t i = 0; i < cur->num_inputs; i++) cur->inputs[i].reset();
    cur->num_inputs = 0;
}

template <value_type T>
inline void array<T>::eval_float_node_(lazy_node *node,
                                       auto store_result) {
  using op = lazy_node::op;
  switch (node->operation) {
    case op::sigmoid: {
      auto input = from_node_(node->inputs[0]);
      auto tmp = make_uninit_(node->shape);
      switch (device_) {
        case Device::CPU:
          cpu::sigmoid(input.buffer_data(), tmp.buffer_data(), input.element_count());
          break;
        case Device::MPS:
          gpu::sigmoid(input.storage_, tmp.storage_);
          break;
      }
      store_result(tmp);
      break;
    }
    case op::relu: {
      auto input = from_node_(node->inputs[0]);
      auto tmp = make_uninit_(node->shape);
      switch (device_) {
        case Device::CPU:
          cpu::relu(input.buffer_data(), tmp.buffer_data(), input.element_count());
          break;
        case Device::MPS:
          gpu::relu(input.storage_, tmp.storage_);
          break;
      }
      store_result(tmp);
      break;
    }
    case op::sigmoid_backward: {
      auto sig_out = from_node_(node->inputs[0]);
      auto dout = from_node_(node->inputs[1]);
      auto tmp = make_uninit_(node->shape);
      switch (device_) {
        case Device::CPU:
          cpu::sigmoid_backward(dout.buffer_data(), sig_out.buffer_data(),
                                tmp.buffer_data(), sig_out.element_count());
          break;
        case Device::MPS:
          gpu::sigmoid_backward(dout.storage_, sig_out.storage_,
                                tmp.storage_, sig_out.element_count());
          break;
      }
      store_result(tmp);
      break;
    }
    case op::layer_norm: {
      auto input = from_node_(node->inputs[0]);
      auto gamma = from_node_(node->inputs[1]);
      auto beta = from_node_(node->inputs[2]);
      auto rows = static_cast<uint32_t>(node->shape[0]);
      auto cols = static_cast<uint32_t>(node->shape[1]);
      auto tmp = make_uninit_(node->shape);
      switch (device_) {
        case Device::CPU:
          cpu::layer_norm(input.buffer_data(), tmp.buffer_data(),
                          gamma.buffer_data(), beta.buffer_data(),
                          rows, cols, node->params.eps);
          break;
        case Device::MPS:
          gpu::layer_norm(input.storage_, tmp.storage_,
                          gamma.storage_, beta.storage_,
                          rows, cols, node->params.eps);
          break;
      }
      store_result(tmp);
      break;
    }
    case op::softmax: {
      auto input = from_node_(node->inputs[0]);
      auto tmp = make_uninit_(node->shape);
      switch (device_) {
        case Device::CPU: {
          for (size_t i = 0; i < node->shape[0]; i++) {
            const auto row = input[i];
            auto c = row.min();
            auto row_tmp = array(row.shape(), 0.0f);
            for (size_t j = 0; j < row.element_count(); j++)
              row_tmp.at(j) = std::exp(row.at(j) - c);
            auto smax = row_tmp / row_tmp.sum();
            for (size_t j = 0; j < row.element_count(); j++)
              tmp[i].at(j) = smax.at(j);
          }
          break;
        }
        case Device::MPS:
          gpu::softmax(input.storage_, tmp.storage_,
                       static_cast<uint32_t>(node->shape[0]),
                       static_cast<uint32_t>(node->shape[1]));
          break;
      }
      store_result(tmp);
      break;
    }
    case op::dot: {
      auto lhs = from_node_(node->inputs[0]);
      auto rhs = from_node_(node->inputs[1]);
      switch (device_) {
        case Device::MPS:
          store_result(dot_operation_helper_(lhs, rhs, mps_dot_operation_));
          break;
        case Device::CPU:
          store_result(dot_operation_helper_(lhs, rhs, cpu_dot_operation_));
          break;
      }
      break;
    }
    case op::batched_dot: {
      auto lhs = from_node_(node->inputs[0]);
      auto rhs = from_node_(node->inputs[1]);
      auto B = node->shape[0], M = node->shape[1], N = node->shape[2];
      auto K = lhs.shape()[lhs.dimension() - 1];
      auto tmp = make_uninit_(node->shape);
      if (device_ == Device::MPS) {
        if (steel_eligible_(M, N, K, false, false)) {
          gpu::sgemm_steel_batched(lhs.storage_, rhs.storage_, tmp.storage_,
                                    M, N, K, K, N,
                                    B, M * K, K * N, M * N);
        } else {
          gpu::sgemm_batched(lhs.storage_, rhs.storage_, tmp.storage_,
                             M, N, K, K, N,
                             B, M * K, K * N, M * N);
        }
      } else {
        for (size_t b = 0; b < B; b++) {
          auto a_slice = lhs[b];
          auto b_slice = rhs[b];
          auto c_slice = cpu_dot_operation_(a_slice, b_slice);
          std::memcpy(tmp.buffer_data() + b * M * N,
                      c_slice.buffer_data(), M * N * sizeof(float));
        }
      }
      store_result(tmp);
      break;
    }
    case op::linear: {
      auto x = from_node_(node->inputs[0]);
      auto W = from_node_(node->inputs[1]);
      auto b = from_node_(node->inputs[2]);
      auto M = x.shape()[0], K = x.shape()[1], N = W.shape()[1];
      bool tL = x.strides()[0] < x.strides()[1];
      bool tR = W.strides()[0] < W.strides()[1];
      bool aligned = (M >= 64 && N >= 64 && K % 16 == 0);
      if (steel_eligible_(M, N, K, tL, tR) && device_ == Device::MPS && aligned) {
        auto tmp = make_uninit_(node->shape);
        auto ldA = static_cast<uint32_t>(std::max(x.strides()[0], x.strides()[1]));
        auto ldB = static_cast<uint32_t>(std::max(W.strides()[0], W.strides()[1]));
        gpu::sgemm_bias_steel(
            x.storage_, W.storage_, tmp.storage_, b.storage_,
            static_cast<uint32_t>(b.element_count()),
            M, N, K, ldA, ldB);
        store_result(tmp);
      } else {
        // Direct dispatch: dot then add (no sub-graph creation)
        auto dot_result = dot_operation_helper_(x, W,
            device_ == Device::MPS ? mps_dot_operation_ : cpu_dot_operation_);
        store_result(arithmetic_operation_(dot_result, b, ArithmeticOperation::Add));
      }
      break;
    }
    case op::linear_sigmoid: {
      auto x = from_node_(node->inputs[0]);
      auto W = from_node_(node->inputs[1]);
      auto b = from_node_(node->inputs[2]);
      auto M = x.shape()[0], K = x.shape()[1], N = W.shape()[1];
      bool tL = x.strides()[0] < x.strides()[1];
      bool tR = W.strides()[0] < W.strides()[1];
      bool aligned = (M >= 64 && N >= 64 && K % 16 == 0);
      if (steel_eligible_(M, N, K, tL, tR) && device_ == Device::MPS && aligned) {
        auto tmp = make_uninit_(node->shape);
        auto ldA = static_cast<uint32_t>(std::max(x.strides()[0], x.strides()[1]));
        auto ldB = static_cast<uint32_t>(std::max(W.strides()[0], W.strides()[1]));
        gpu::sgemm_bias_sigmoid_steel(
            x.storage_, W.storage_, tmp.storage_, b.storage_,
            static_cast<uint32_t>(b.element_count()),
            M, N, K, ldA, ldB);
        store_result(tmp);
      } else {
        // Direct dispatch: dot then bias_sigmoid (no sub-graph creation)
        auto result = dot_operation_helper_(x, W,
            device_ == Device::MPS ? mps_dot_operation_ : cpu_dot_operation_);
        auto n = result.element_count();
        auto cols = static_cast<uint32_t>(b.element_count());
        if (device_ == Device::MPS) {
          gpu::bias_sigmoid(result.storage_, b.storage_, n, cols);
        } else {
          cpu::bias_sigmoid(result.buffer_data(), b.buffer_data(), n, cols);
        }
        store_result(result);
      }
      break;
    }
    case op::conv2d: {
      auto input = from_node_(node->inputs[0]);
      auto weight = from_node_(node->inputs[1]);
      auto K = node->params.kernel_size;
      auto N = input.shape()[0], H = input.shape()[1];
      auto W = input.shape()[2], C_in = input.shape()[3];
      auto C_out = weight.shape()[0];
      auto M_dim = N * H * W;

      auto result = make_uninit_({M_dim, C_out});
      gpu::conv2d(input.storage_, weight.storage_, result.storage_,
                  N, H, W, C_in, K, C_out);
      result.reshape({N, H, W, C_out});
      store_result(result);
      break;
    }
    case op::sum_axis: {
      auto input = from_node_(node->inputs[0]);
      auto axis = node->params.axis;
      auto in_shape = input.shape();

      if (input.dimension() == 2 && axis == 0 &&
          input.buffer_element_count() == input.element_count()) {
        switch (device_) {
          case Device::CPU: {
            auto tmp = array(node->shape, 0.0f);
            cpu::sum_axis0<float>(input.buffer_data(), tmp.buffer_data(),
                                  in_shape[0], in_shape[1]);
            store_result(tmp);
            break;
          }
          case Device::MPS: {
            size_t rows = in_shape[0], cols = in_shape[1];
            constexpr size_t kBM = 1024;
            size_t num_row_blocks = (rows + kBM - 1) / kBM;
            auto partial = make_uninit_({num_row_blocks * cols});
            auto out = make_uninit_(node->shape);
            gpu::sum_axis0_f32(input.storage_, out.storage_,
                               rows, cols,
                               partial.storage_, num_row_blocks, kBM);
            store_result(out);
            break;
          }
        }
      } else {
        auto tmp = array(node->shape, 0.0f);
        if (input.dimension() == 2 && axis == 0 &&
            input.buffer_element_count() == input.element_count()) {
          cpu::sum_axis0<float>(input.buffer_data(), tmp.buffer_data(),
                                in_shape[0], in_shape[1]);
        } else {
          input.enumerate_position_([&](const auto &pos) {
            auto p = pos;
            p.erase(p.begin() + axis);
            tmp.at(p) += input.at(pos);
          });
        }
        store_result(tmp);
      }
      break;
    }
    default:
      break;
  }
}

template <value_type T>
inline std::shared_ptr<lazy_node> array<T>::to_node_(const array &a) {
  if (a.node_ && !a.node_->evaluated) return a.node_;
  // storage_ may be stale if node was evaluated via another copy of this array
  if (a.node_ && a.node_->evaluated)
    return lazy_node::leaf(a.node_->data, a.node_->shape, a.node_->strides);
  return lazy_node::leaf(a.storage_, a.shape_, a.strides_);
}

template <value_type T>
inline array<T> array<T>::from_lazy_node_(std::shared_ptr<lazy_node> node) {
  array a;
  a.shape_ = node->shape;
  a.strides_ = node->strides;
  a.node_ = std::move(node);
  return a;
}

template <value_type T>
inline array<T> array<T>::make_lazy_op_(lazy_node::op o, const array &lhs,
                                        const array &rhs) {
  auto lnode = to_node_(lhs);
  auto rnode = to_node_(rhs);

  auto out_shape = broadcast_shape(lhs.shape_, rhs.shape_);
  auto out_strides = contiguous_strides(out_shape);

  auto node = lazy_node::make(o, {std::move(lnode), std::move(rnode)},
                              out_shape, out_strides);

  return from_lazy_node_(std::move(node));
}

template <value_type T>
inline array<T> array<T>::lazy_or_eager_(lazy_node::op lo, ArithmeticOperation ao,
                                         const array &lhs, const array &rhs) {
  if constexpr (std::same_as<T, float>) {
    return make_lazy_op_(lo, lhs, rhs);
  } else {
    return arithmetic_operation_(lhs, rhs, ao);
  }
}

template <value_type T>
inline void array<T>::allocate_buffer_() {
  auto len = element_count();
  auto bytes = len * sizeof(T);

  storage_ = storage::make(bytes);
  storage_.off = 0;
  storage_.len = len;
}

template <value_type T>
inline array<T> array<T>::materialize_() const {
  if (buffer_element_count() == element_count()) {
    return *this;
  }

  auto tmp = make_uninit_(shape_);
  enumerate_position_([&](const auto &pos) { tmp.at(pos) = at(pos); });
  return tmp;
}

//----------------------------------------------------------------------------

template <typename T>
constexpr size_t nested_initializer_item_count_(const T &l) {
  return 1;
}

template <typename T>
constexpr size_t nested_initializer_item_count_(std::initializer_list<T> l) {
  size_t count = 0;
  for (auto it = l.begin(); it != l.end(); ++it) {
    count += nested_initializer_item_count_(*it);
  }
  return count;
}

template <typename T>
constexpr void nested_initializer_copy_(T &&dst, const auto &src) {
  *dst++ = src;
}

template <typename T, typename U>
constexpr void nested_initializer_copy_(T &&dst, std::initializer_list<U> src) {
  for (auto it = src.begin(); it != src.end(); ++it) {
    nested_initializer_copy_(std::forward<T>(dst), *it);
  }
}

template <value_type T>
inline void array<T>::copy_initializer_list_(const auto &l) {
  if (nested_initializer_item_count_(l) != element_count()) {
    throw std::runtime_error("array: invalid initializer list.");
  }
  nested_initializer_copy_(buffer_data(), l);
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::bounds_check_(size_t i) const {
  if (strides_.empty() || i >= element_count()) {
    throw std::runtime_error("array: index is out of bounds.");
  }
}

template <value_type T>
inline void array<T>::bounds_check_(size_t x, size_t y) const {
  if (dimension() != 2 || x >= shape_[0] || y >= shape_[1]) {
    throw std::runtime_error("array: (x, y) is out of bounds.");
  }
}

template <value_type T>
inline void array<T>::bounds_check_(size_t x, size_t y, size_t z) const {
  if (dimension() != 3 || x >= shape_[0] || y >= shape_[1] || z >= shape_[2]) {
    throw std::runtime_error("array: (x, y, z) is out of bounds.");
  }
}

//----------------------------------------------------------------------------

template <value_type T>
template <typename U>
inline void array<T>::enumerate_position_(size_t shape_index,
                                          std::vector<size_t> &position,
                                          U fn) const {
  if (shape_index == shape_.size()) {
    fn(position);
    return;
  }

  for (size_t i = 0; i < shape_[shape_index]; i++) {
    position[shape_index] = i;
    enumerate_position_(shape_index + 1, position, fn);
  }
}

template <value_type T>
template <typename U>
inline void array<T>::enumerate_position_(U fn) const {
  std::vector<size_t> position(shape_.size());
  for (size_t i = 0; i < shape_[0]; i++) {
    position[0] = i;
    enumerate_position_(1, position, fn);
  }
}

//----------------------------------------------------------------------------

template <value_type T>
inline auto array<T>::broadcast_(const array &lhs, const array &rhs, auto cb) {
  lhs.ensure_evaluated_();
  rhs.ensure_evaluated_();
  if (lhs.shape() == rhs.shape()) {
    return cb(lhs, rhs);
  } else if (lhs.dimension() < rhs.dimension()) {
    return cb(lhs.broadcast(rhs.shape()), rhs);
  } else if (lhs.dimension() > rhs.dimension()) {
    return cb(lhs, rhs.broadcast(lhs.shape()));
  } else if (lhs.dimension() == rhs.dimension()) {
    // Same dimension, check broadcast compatibility (each dim must be equal or 1)
    bool rhs_to_lhs = true, lhs_to_rhs = true;
    for (size_t i = 0; i < lhs.dimension(); i++) {
      if (lhs.shape()[i] != rhs.shape()[i]) {
        if (rhs.shape()[i] != 1) rhs_to_lhs = false;
        if (lhs.shape()[i] != 1) lhs_to_rhs = false;
      }
    }
    if (rhs_to_lhs) return cb(lhs, rhs.broadcast(lhs.shape()));
    if (lhs_to_rhs) return cb(lhs.broadcast(rhs.shape()), rhs);

    // Both sides need broadcast (e.g. {3,1} + {1,3} -> {3,3})
    shape_type target(lhs.dimension());
    bool compatible = true;
    for (size_t i = 0; i < lhs.dimension(); i++) {
      if (lhs.shape()[i] == rhs.shape()[i]) {
        target[i] = lhs.shape()[i];
      } else if (lhs.shape()[i] == 1) {
        target[i] = rhs.shape()[i];
      } else if (rhs.shape()[i] == 1) {
        target[i] = lhs.shape()[i];
      } else {
        compatible = false;
        break;
      }
    }
    if (compatible) {
      return cb(lhs.broadcast(target).materialize_(),
                rhs.broadcast(target).materialize_());
    }
  }
  throw std::runtime_error("array: invalid operation.");
}

template <value_type T>
template <value_type U>
inline array<U> array<T>::apply_binary_operation_(const array &rhs,
                                                  auto ope) const {
  return broadcast_(*this, rhs, [ope](const auto &lhs, const auto &rhs) {
  
    auto n = std::max(lhs.element_count(), rhs.element_count());
    auto shape = lhs.element_count() >= rhs.element_count() ? lhs.shape() : rhs.shape();
    auto tmp = array<U>(shape, U{});
    for (size_t i = 0; i < n; i++) {
      tmp.at(i) = ope(lhs.at(i % lhs.element_count()),
                       rhs.at(i % rhs.element_count()));
    }
    return tmp;
  });
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::cpu_arithmetic_dispatch_(const storage &lhs,
                                               const storage &rhs,
                                               storage &dst,
                                               ArithmeticOperation ope) {
  switch (ope) {
    case ArithmeticOperation::Add: cpu::add<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Sub: cpu::sub<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Mul: cpu::mul<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Div: cpu::div<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Pow: cpu::pow<T>(lhs, rhs, dst); break;
  }
}

template <value_type T>
inline auto array<T>::cpu_arithmetic_operation_(const array &lhs,
                                                const array &rhs,
                                                ArithmeticOperation ope) {
  return broadcast_(lhs, rhs, [ope](const auto &lhs, const auto &rhs) {
    auto tmp = make_uninit_(lhs.shape());
    cpu_arithmetic_dispatch_(lhs.storage_, rhs.storage_, tmp.storage_, ope);
    return tmp;
  });
}

template <value_type T>
inline void array<T>::cpu_arithmetic_inplace_(const array &rhs,
                                                    ArithmeticOperation ope) {
  if (shape() == rhs.shape() || rhs.element_count() <= element_count()) {
    cpu_arithmetic_dispatch_(storage_, rhs.storage_, storage_, ope);
  } else {
    *this = cpu_arithmetic_operation_(*this, rhs, ope);
  }
}

template <value_type T>
inline void array<T>::msl_arithmetic_dispatch_(const storage &lhs,
                                               const storage &rhs,
                                               storage &dst,
                                               ArithmeticOperation ope) {
  switch (ope) {
    case ArithmeticOperation::Add: msl::add<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Sub: msl::sub<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Mul: msl::mul<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Div: msl::div<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Pow: msl::pow<T>(lhs, rhs, dst); break;
  }
}

template <value_type T>
inline auto array<T>::msl_arithmetic_operation_(const array &lhs,
                                                const array &rhs,
                                                ArithmeticOperation ope) {
  return broadcast_(lhs, rhs, [ope](const auto &lhs, const auto &rhs) {
    auto tmp = make_uninit_(lhs.shape());
    msl_arithmetic_dispatch_(lhs.storage_, rhs.storage_, tmp.storage_, ope);
    return tmp;
  });
}

template <value_type T>
inline auto array<T>::arithmetic_operation_(const array &lhs,
                                            const array &rhs,
                                            ArithmeticOperation ope) {
  switch (device_) {
    case Device::CPU:
      return cpu_arithmetic_operation_(lhs, rhs, ope);
    case Device::MPS:
      return msl_arithmetic_operation_(lhs, rhs, ope);
  }
}

template <value_type T>
inline void array<T>::arithmetic_inplace_(const array &rhs,
                                          ArithmeticOperation ope) {
  ensure_evaluated_();
  rhs.ensure_evaluated_();
  switch (device_) {
    case Device::CPU:
      cpu_arithmetic_inplace_(rhs, ope);
      break;
    case Device::MPS:
      *this = msl_arithmetic_operation_(*this, rhs, ope);
      break;
  }
  node_.reset();
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::cpu_dot_operation_(const array &lhs,
                                             const array &rhs) {
  auto M = lhs.shape_[0], K = lhs.shape_[1], N = rhs.shape_[1];
  auto tmp = make_uninit_({M, N});

  if constexpr (std::same_as<T, float>) {
    auto tA = lhs.strides_[0] < lhs.strides_[1] ? CblasTrans : CblasNoTrans;
    auto tB = rhs.strides_[0] < rhs.strides_[1] ? CblasTrans : CblasNoTrans;
    // For NoTrans: lda = cols (K or N). For Trans: lda = physical row width.
    auto ldA = static_cast<int>(tA == CblasTrans ? lhs.strides_[1] : K);
    auto ldB = static_cast<int>(tB == CblasTrans ? rhs.strides_[1] : N);

    auto *a = static_cast<const float *>(lhs.storage_.data) + lhs.storage_.off;
    auto *b = static_cast<const float *>(rhs.storage_.data) + rhs.storage_.off;
    auto *c = static_cast<float *>(tmp.storage_.data) + tmp.storage_.off;
    cblas_sgemm(CblasRowMajor, tA, tB, M, N, K, 1.0f,
                a, ldA, b, ldB, 0.0f, c, N);
  } else {
    cpu::dot<T>(lhs.storage_, rhs.storage_, tmp.storage_, K, M, N);
  }

  return tmp;
}

template <value_type T>
inline array<T> array<T>::mps_dot_operation_(const array &lhs,
                                             const array &rhs) {
  if constexpr (std::same_as<T, float>) {
    auto M = lhs.shape_[0], K = lhs.shape_[1], N = rhs.shape_[1];
    auto tmp = make_uninit_({M, N});

    auto tL = lhs.strides_[0] < lhs.strides_[1];
    auto tR = rhs.strides_[0] < rhs.strides_[1];

    auto ldA = static_cast<uint32_t>(std::max(lhs.strides_[0], lhs.strides_[1]));
    auto ldB = static_cast<uint32_t>(std::max(rhs.strides_[0], rhs.strides_[1]));
    if (M == 1 && !tL && !tR) {
      gpu::sgemv(lhs.storage_, rhs.storage_, tmp.storage_, N, K);
    } else if (steel_eligible_(M, N, K, tL, tR)) {
      gpu::sgemm_steel(lhs.storage_, rhs.storage_, tmp.storage_,
                        M, N, K, ldA, ldB);
    } else if (!tL && !tR) {
      gpu::sgemm(lhs.storage_, rhs.storage_, tmp.storage_,
                 M, N, K, ldA, ldB, tL, tR);
    } else {
      auto phys_A_rows = tL ? K : M, phys_A_cols = tL ? M : K;
      auto phys_B_rows = tR ? N : K, phys_B_cols = tR ? K : N;
      mps::dot_f32_ex(lhs.storage_, rhs.storage_, tmp.storage_,
                      phys_A_rows, phys_A_cols, phys_B_rows, phys_B_cols,
                      M, N, K, tL, tR);
    }
    return tmp;
  } else {
    return cpu_dot_operation_(lhs, rhs);
  }
}

template <value_type T>
template <typename U>
inline array<T> array<T>::dot_operation_(const array &rhs, U fn) const {
  ensure_evaluated_();
  rhs.ensure_evaluated_();
  return dot_operation_helper_(*this, rhs, fn);
}

template <value_type T>
template <typename U>
inline array<T> array<T>::dot_operation_helper_(const array &lhs, const array &rhs, U fn) {
  if (lhs.dimension() == 2 && rhs.dimension() == 2 && lhs.shape()[1] == rhs.shape()[0]) {
    return fn(lhs, rhs);
  }
  if (lhs.dimension() == 1 && rhs.dimension() == 2 && lhs.shape()[0] == rhs.shape()[0]) {
    auto lhs2 = lhs;
    lhs2.reshape({1, lhs.shape()[0]});
    auto tmp = fn(lhs2, rhs);
    tmp.reshape({rhs.shape()[1]});
    return tmp;
  }
  if (lhs.dimension() == 2 && rhs.dimension() == 1 && lhs.shape()[1] == rhs.shape()[0]) {
    auto rhs2 = rhs;
    rhs2.reshape({rhs.shape()[0], 1});
    auto tmp = fn(lhs, rhs2);
    tmp.reshape({lhs.shape()[0]});
    return tmp;
  }
  if (lhs.dimension() == 1 && rhs.dimension() == 1 && lhs.shape()[0] == rhs.shape()[0]) {
    auto lhs2 = lhs;
    lhs2.reshape({1, lhs.shape()[0]});
    auto rhs2 = rhs;
    rhs2.reshape({rhs.shape()[0], 1});
    auto tmp = fn(lhs2, rhs2);
    tmp.reshape({});
    return tmp;
  }
  throw std::runtime_error("array: can't do `dot` operation.");
}

//----------------------------------------------------------------------------

template <value_type T>
inline std::ostream &operator<<(std::ostream &os, const array<T> &arr) {
  os << arr.print_array();
  return os;
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> operator+(auto lhs, const array<T> &rhs) {
  return array<T>(static_cast<T>(lhs)) + rhs;
}

template <value_type T>
inline array<T> operator-(auto lhs, const array<T> &rhs) {
  return array<T>(static_cast<T>(lhs)) - rhs;
}

template <value_type T>
inline array<T> operator*(auto lhs, const array<T> &rhs) {
  return array<T>(static_cast<T>(lhs)) * rhs;
}

template <value_type T>
inline array<T> operator/(auto lhs, const array<T> &rhs) {
  return array<T>(static_cast<T>(lhs)) / rhs;
}

//----------------------------------------------------------------------------

template <value_type T, value_type U>
inline array<T> where(const array<U> &cond, T x, T y) {

  auto tmp = array<T>(cond.shape(), T{});
  for (size_t i = 0; i < cond.element_count(); i++) {
    tmp.at(i) = cond.at(i) ? x : y;
  }
  return tmp;
}

//----------------------------------------------------------------------------

template <value_type T>
inline bool array_equal(const array<T> &a, const array<T> &b) {
  if (&a == &b) return true;
  if (a.shape() != b.shape()) return false;
  for (size_t i = 0; i < a.element_count(); i++) {
    if (a.at(i) != b.at(i)) return false;
  }
  return true;
}

template <std::floating_point T, std::floating_point U>
inline bool is_close(T a, U b, float tolerance) {
  return std::abs(static_cast<float>(a) - static_cast<float>(b)) <= tolerance;
}

template <std::integral T, std::integral U>
inline bool is_close(T a, U b) {
  return a == b;
}

template <value_type T>
inline bool allclose(const array<T> &a, const array<T> &b, float tolerance) {
  if (&a == &b) return true;
  if (a.shape() != b.shape()) return false;
  auto as = a.buffer_span();
  auto bs = b.buffer_span();
  for (size_t i = 0; i < as.size(); i++) {
    if constexpr (std::is_same_v<T, float>) {
      if (std::abs(as[i] - bs[i]) > tolerance) return false;
    } else {
      if (as[i] != bs[i]) return false;
    }
  }
  return true;
}

//----------------------------------------------------------------------------

template <value_type T>
inline auto empty(const shape_type &shape) {
  return array<T>(shape, T{});
}

template <value_type T>
inline auto zeros(const shape_type &shape) {
  return array<T>(shape, 0);
}

template <value_type T>
inline auto ones(const shape_type &shape) {
  return array<T>(shape, 1);
}

inline auto random(const shape_type &shape) {
  auto tmp = array<float>(shape, 0.0);
  tmp.random();
  return tmp;
}

// Batch evaluation: evaluate multiple arrays in a single topological pass.
// Equivalent to MLX's mx::eval(a, b, c, ...).
template <value_type... Ts>
inline void eval(array<Ts>&... arrays) {
  // Collect root node pointers
  std::vector<lazy_node *> root_ptrs;
  auto collect = [&](auto &arr) {
    auto n = arr.lazy_node_ptr();
    if (n && !n->evaluated) root_ptrs.push_back(n.get());
  };
  (collect(arrays), ...);

  if (root_ptrs.empty()) return;

  // Single topo sort + eval across all roots
  // Use initializer_list overload via vector→span trick
  std::vector<lazy_node *> order;
  {
    std::vector<std::pair<lazy_node *, size_t>> stack;
    for (auto *r : root_ptrs) {
      if (!r->visited) {
        r->visited = true;
        stack.push_back({r, 0});
      }
    }
    while (!stack.empty()) {
      auto &[cur, idx] = stack.back();
      if (cur->evaluated) { stack.pop_back(); continue; }
      if (idx < cur->num_inputs) {
        auto *child = cur->inputs[idx].get();
        idx++;
        if (child && !child->evaluated && !child->visited) {
          child->visited = true;
          stack.push_back({child, 0});
        }
      } else {
        order.push_back(cur);
        stack.pop_back();
      }
    }
  }

  for (auto *cur : order) {
    if (cur->evaluated) continue;
    array<float>::evaluate_single_node_(cur);
  }

  // Flush GPU commands and wait for completion (matches MLX's blocking eval)
  if (gpu_pending_) gpu_context::instance().flush();

  for (auto *cur : order) cur->visited = false;

  // Sync storage for all arrays
  auto sync = [](auto &arr) { arr.eval(); };
  (sync(arrays), ...);
}

};  // namespace sil
