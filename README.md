Silicon Array
=============

Numerical Computing Library for Apple Silicon

* Header-only C++23 library -- just `#include <silarray.h>`
* Switchable CPU/GPU backend via `sil::use_cpu()` / `sil::use_mps()` (default: GPU)
* CPU: Accelerate framework (vDSP, CBLAS, NEON)
* GPU: Metal Shading Language (MSL) — STEEL SGEMM, K-unrolled GEMV, online-softmax, right-sized reductions, and conv2d that dispatches to Apple's MPSGraph for medium/large-channel shapes
* MLX-style lazy evaluation: operations build a computation graph; `sil::eval()` evaluates multiple arrays in a single topological pass
* Affine fusion for chained scalar operations
* Zero-copy slicing via `array::slice(start, count)` for efficient batched data access
* Data types: `float`, `int`, `bool`

Requirements
------------

* macOS with Apple Silicon
* Xcode Command Line Tools (clang++ with C++23 support)
* Frameworks: Metal, Accelerate, MetalPerformanceShaders, MetalPerformanceShadersGraph, Foundation

Example
-------

```cpp
#include <silarray.h>

auto a = sil::ones<float>({1000, 1000});
auto b = sil::ones<float>({1000, 1000});

auto c = a + b;       // builds a lazy graph node (no compute yet)
auto d = a.dot(b);    // also lazy

sil::eval(c, d);       // evaluates both in a single topological pass

sil::use_cpu();        // switch to CPU backend
auto e = a + b;        // CPU-side eager
```

For training loops, prefer batch evaluation:

```cpp
W1 = W1 - dW1 * lr;
b1 = b1 - db1 * lr;
W2 = W2 - dW2 * lr;
b2 = b2 - db2 * lr;
sil::eval(W1, b1, W2, b2);   // single eval evaluates the whole graph
```

Operations
----------

### CPU/GPU switchable

| Category | Operations |
|----------|-----------|
| Arithmetic | `+` `-` `*` `/` `pow` (elementwise, with broadcasting) |
| In-place | `+=` `-=` `*=` `/=` |
| Linear algebra | `dot` (STEEL SGEMM on GPU, CBLAS on CPU) |
| Activations | `sigmoid` `relu` `softmax` `layer_norm` |
| Fused ops | `linear` (dot + bias), `linear_sigmoid` (dot + bias + sigmoid) |
| Reduction | `sum` `sum(axis)` `min` `max` `argmax` |
| Convolution | `conv2d` (NHWC; smallch3 kernel for C_in=3, MPSGraph for C_in ≥ 16, JIT implicit-GEMM fallback) |

### CPU/GPU switchable (continued)

| Category | Operations |
|----------|-----------|
| NN utilities | `sigmoid_backward` |
| Slicing | `slice(start, count)` (zero-copy view) |
| Evaluation | `eval()`, free function `sil::eval(arrays...)`, `sil::synchronize()` |

### CPU only

| Category | Operations |
|----------|-----------|
| Comparison | `==` `!=` `>` `<` `>=` `<=` |
| Shape | `clone` `transpose` `reshape` `broadcast` |
| Creation | `empty` `zeros` `ones` `random` `constants` |
| Reduction | `mean` `mean(axis)` `count` `all` |
| NN utilities | `mean_square_error` `one_hot` |
| Selection | `where(condition, x, y)` |
| Testing | `array_equal` `allclose` |

Performance
-----------

Competitive with MLX across most operations on Apple M1 Pro, measured with eval+sync per iteration.

| Category | vs MLX |
|----------|--------|
| SGEMM (square, 1024–8192) | **Same speed** |
| GEMV (1x4096x4096, single-vector inference) | **Same speed** |
| Small-batch SGEMM (32x4096x768) | **3.4x faster** |
| Elementwise (add, mul, div, pow) | **Same speed** |
| Reduction (sum, min, max) | **1.1–1.2x faster** |
| Argmax | **1.3–1.9x faster** |
| Softmax | **Same to 1.1x faster** (incl. cols > 4096) |
| Layer norm | **1.5–3.6x faster** |
| Conv2d (ResNet mid, 16×64×56×56, k=3) | **1.5x faster** (via MPSGraph) |
| Conv2d (ResNet deep, 16×128×28×28, k=3) | **Tied to 1.1x faster** (via MPSGraph) |
| Conv2d (ImageNet first, 1×3×224×224, k=3) | **1.1x faster** (smallch3 kernel) |
| BMM attention | **Same speed** |
| Transformer block | **Same to 1.1x faster** |
| MLP inference | Noisy (±5–20%, shape-dependent) |

See [bench/README.md](bench/README.md) for detailed results.

Build and Run
-------------

### Unit tests

```bash
cd test
make
```

Tests can be run in different device modes:

```bash
./test          # GPU mode (default)
./test --gpu    # explicit GPU
./test --cpu    # CPU mode
```

### MNIST

```bash
cd test
make mnist
./mnist
```

### Benchmarks

Benchmarks compare against Eigen, MLX, libtorch, and ggml.

```bash
just bench-all          # all benchmarks
just bench-micro        # micro only

# or manually:
cd bench
make run                # all benchmarks
make table              # Markdown table output
```

See [bench/README.md](bench/README.md) for setup instructions and full results.

Architecture
------------

```
include/
  silarray.h          Main header (includes all below)
  array.h             Core array class with expression templates
  cpu.h               CPU backend (Accelerate: vDSP, CBLAS, NEON)
  gpu.h               GPU backend (Metal/MSL kernels, MPSMatrixMultiplication, MPSGraph)
  device.h            Device selection (CPU/MPS switch)
  types.h             Type concepts (float, int, bool)
  objc.h              Objective-C bridge for Metal API
  unified_memory.h    GPU shared memory management
```

License
-------

MIT license (c) 2026 Yuji Hirose
