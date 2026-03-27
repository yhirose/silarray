# Benchmark Suite

Silicon Array (sil) vs Eigen vs MLX vs libtorch (PyTorch C++) vs ggml

## Requirements

- macOS with Apple Silicon
- Eigen: `brew install eigen`
- MLX: `brew install mlx`
- libtorch: `brew install pytorch`
- ggml: build from source (see below)

The Makefile auto-detects each library by probing for a known header — if one
isn't installed the corresponding benchmark rows are simply skipped. No
manual edits needed.

### Building ggml

The Homebrew `ggml` formula is CPU-only (no Metal backend), so you must
build from source with `GGML_METAL=ON`. The benchmark Makefile expects the
install to live at `~/Projects/ggml-install` (a persistent path), but you
can clone the source anywhere — it's deleted after the install completes.

```bash
git clone --depth 1 https://github.com/ggml-org/ggml.git
cd ggml
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/Projects/ggml-install \
      -DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_ACCELERATE=ON
cmake --build build -j$(sysctl -n hw.ncpu)
cmake --install build
cd .. && rm -rf ggml
```

## Build & Run

```bash
just bench-all      # all benchmarks (requires justfile)

# or manually:
cd bench
make run            # all benchmarks
make run-micro      # micro only
make run-composite  # composite only
make run-mnist      # MNIST only
```

Other output formats:

```bash
make table          # compact Markdown table (for README)
make csv            # CSV

# per-benchmark:
./micro/bench_sgemm --table
./micro/bench_sgemm --csv
```

## Benchmarks

### Micro — single operation throughput

| Name | What it measures |
|------|-----------------|
| `bench_sgemm` | Matrix multiplication (GFLOPS) at sizes 1024–8192, square and non-square |
| `bench_elementwise` | Vector add/mul/div/pow throughput at 1M–10M elements |
| `bench_broadcast` | Bias-add pattern `(N,M) + (M)` |
| `bench_reduction` | sum, min, max (1D), sum axis=0 (2D), argmax (2D) |
| `bench_nn_ops` | softmax, layer_norm, conv2d, batch matmul |

### Composite — multi-operation workloads

| Name | What it measures |
|------|-----------------|
| `bench_mlp` | 2-layer MLP (768→2048→768) inference |
| `bench_train` | 2-layer MLP (768→2048→768) training step |
| `bench_transformer` | Single transformer block (self-attention + FFN) inference |

### MNIST — end-to-end with real data

| Name | What it measures |
|------|-----------------|
| `bench_classifier` | 784→50→10 classifier, training + inference |
| `bench_autoencoder` | 784→512→256→64→256→512→784 autoencoder, training + inference |

## Results

Apple M1 Pro. All benchmarks include `sil::eval()` (or `sil::synchronize()`) inside
the measured function so sil and other libraries do equivalent work per iteration.

### Broadcast

Bias-add broadcast `(N,M) + (M)`: GPU (1024-4096) and CPU (256-1024)

| benchmark                    | sil-gpu    | ggml       | mlx        | torch      | sil-cpu    | eigen      |
|------------------------------|------------|------------|------------|------------|------------|------------|
| broadcast (1024x1024)+(1024) | 327 us (1.1x) | 553 us (1.8x) | **301 us** | 862 us (2.9x) | -          | -          |
| broadcast (4096x512)+(512)   | 412 us (1.1x) | 764 us (2.1x) | **368 us** | 896 us (2.4x) | -          | -          |
| broadcast (4096x4096)+(4096) | **1.05 ms** | 4.11 ms (4.0x) | **1.03 ms** | 5.19 ms (5.0x) | -          | -          |
| broadcast (256x256)+(256)    | -          | -          | -          | -          | 8 us (2.1x) | **4 us**   |
| broadcast (1024x256)+(256)   | -          | -          | -          | -          | 31 us (1.4x) | **22 us**  |
| broadcast (1024x1024)+(1024) | -          | -          | -          | -          | **68 us**  | 88 us (1.3x) |

### Elementwise

Per-element vector operations: GPU (1M-10M) and CPU (100K-1M)

| benchmark      | sil-gpu    | ggml       | mlx        | torch      | sil-cpu    | eigen      |
|----------------|------------|------------|------------|------------|------------|------------|
| add (1000000)  | **317 us** | 1.45 ms (4.6x) | **325 us** | **321 us** | -          | -          |
| add (10000000) | **967 us** | 8.76 ms (9.1x) | 1.03 ms (1.1x) | **973 us** | -          | -          |
| mul (1000000)  | **316 us** | 1.49 ms (4.7x) | **321 us** | **324 us** | -          | -          |
| mul (10000000) | **993 us** | 9.39 ms (9.5x) | **1.02 ms** | **994 us** | -          | -          |
| div (1000000)  | **319 us** | 1.44 ms (4.6x) | **320 us** | **315 us** | -          | -          |
| div (10000000) | 1.05 ms (1.1x) | 8.61 ms (9.1x) | **950 us** | **946 us** | -          | -          |
| pow (1000000)  | **324 us** | -          | 367 us (1.1x) | 346 us (1.1x) | -          | -          |
| pow (10000000) | **981 us** | -          | **946 us** | 1.06 ms (1.1x) | -          | -          |

### NN Ops

Neural network primitives: softmax, layer norm, conv2d, batched matmul

| benchmark                                      | sil-gpu    | ggml       | mlx        | torch      |
|------------------------------------------------|------------|------------|------------|------------|
| softmax (256x512)                              | **246 us** | 335 us (1.4x) | **242 us** | 393 us (1.6x) |
| softmax (1024x1024)                            | **307 us** | 644 us (2.1x) | **310 us** | 587 us (1.9x) |
| softmax (4096x2048)                            | **763 us** | 2.87 ms (3.8x) | **788 us** | 1.16 ms (1.5x) |
| softmax (256x8192)                             | 458 us (1.5x) | 711 us (2.4x) | **299 us** | 409 us (1.4x) |
| layer_norm (256x512)                           | **235 us** | 268 us (1.1x) | 338 us (1.4x) | **241 us** |
| layer_norm (1024x1024)                         | **277 us** | 449 us (1.6x) | 494 us (1.8x) | 321 us (1.2x) |
| layer_norm (4096x2048)                         | **611 us** | 1.92 ms (3.1x) | 3.14 ms (5.1x) | 1.06 ms (1.7x) |
| conv2d ImageNet first layer (1x3x224x224, k=3) | 564 us (1.7x) | -          | **337 us** | **353 us** |
| conv2d ResNet mid layer (16x64x56x56, k=3)     | 2.89 ms (1.4x) | -          | 2.95 ms (1.4x) | **2.04 ms** |
| conv2d ResNet deep layer (16x128x28x28, k=3)   | 2.93 ms (1.5x) | -          | **2.03 ms** | **1.94 ms** |
| bmm attention (8h, seq=128, d=64)              | **228 us** | 288 us (1.3x) | **234 us** | 252 us (1.1x) |
| bmm attention (8h, seq=512, d=64)              | **301 us** | 564 us (1.9x) | **309 us** | 342 us (1.1x) |
| bmm attention (16h, seq=256, d=128)            | **292 us** | 386 us (1.3x) | **303 us** | 337 us (1.2x) |

### Reduction

Reduction operations (sum, min, max, argmax) on 1D and 2D arrays

| benchmark             | sil-gpu    | ggml       | mlx        | torch      |
|-----------------------|------------|------------|------------|------------|
| sum (1000000)         | **230 us** | 526 us (2.3x) | 245 us (1.1x) | 282 us (1.2x) |
| sum (10000000)        | **445 us** | 3.75 ms (8.4x) | 471 us (1.1x) | 502 us (1.1x) |
| sum axis=0 (1024x256) | **213 us** | 232 us (1.1x) | 234 us (1.1x) | 248 us (1.2x) |
| sum axis=0 (4096x256) | **231 us** | 279 us (1.2x) | **232 us** | 290 us (1.3x) |
| min (1000000)         | **229 us** | -          | 244 us (1.1x) | 316 us (1.4x) |
| min (10000000)        | **455 us** | -          | **471 us** | 495 us (1.1x) |
| max (1000000)         | **229 us** | -          | 251 us (1.1x) | 275 us (1.2x) |
| max (10000000)        | **443 us** | -          | **464 us** | 502 us (1.1x) |
| argmax (1024x256)     | **240 us** | -          | **232 us** | 271 us (1.2x) |
| argmax (4096x256)     | 333 us (1.1x) | -          | **300 us** | **298 us** |

### SGEMM

Single-precision matrix multiplication: GPU (1024-8192 square + DL shapes)

| benchmark                                             | sil-gpu    | ggml       | mlx        | torch      |
|-------------------------------------------------------|------------|------------|------------|------------|
| sgemm 1024x1024                                       | **832 us** | 1.01 ms (1.3x) | **806 us** | **834 us** |
| sgemm 2048x2048                                       | **4.94 ms** | 5.43 ms (1.1x) | **4.83 ms** | **4.84 ms** |
| sgemm 4096x4096                                       | **37.46 ms** | 38.73 ms (1.1x) | **36.71 ms** | **36.28 ms** |
| sgemm 8192x8192                                       | **311.03 ms** | 540.02 ms (1.8x) | 321.76 ms (1.1x) | **299.20 ms** |
| 1x4096x4096 (single-vector inference)                 | 705 us (1.1x) | 796 us (1.3x) | 652 us (1.1x) | **619 us** |
| 32x4096x768 (small-batch embedding)                   | **187 us** | 515 us (2.8x) | 388 us (2.1x) | 334 us (1.8x) |
| 256x4096x768 (medium-batch projection)                | 732 us (1.1x) | 1.21 ms (1.8x) | **673 us** | 709 us (1.1x) |
| 1024x4096x768 (large-batch projection)                | **1.99 ms** | 2.59 ms (1.3x) | 2.39 ms (1.2x) | **1.97 ms** |
| 2048x768x4096 (FFN down-projection)                   | **3.82 ms** | 3.93 ms (1.1x) | **3.71 ms** | **3.75 ms** |

### MLP Inference

2-layer MLP (768->2048->768 with sigmoid) forward pass

| benchmark                  | sil-gpu    | ggml       | mlx        | torch      |
|----------------------------|------------|------------|------------|------------|
| mlp inference (batch=128)  | 938 us (1.7x) | 754 us (1.4x) | **537 us** | 596 us (1.1x) |
| mlp inference (batch=256)  | 853 us (1.1x) | 1.10 ms (1.4x) | **802 us** | **825 us** |
| mlp inference (batch=1024) | **2.07 ms** | 3.43 ms (1.7x) | **2.13 ms** | 2.32 ms (1.1x) |

### Training

Full training step (forward + backward + SGD) for a 2-layer MLP (768->2048->768, sigmoid)

| benchmark              | sil-gpu    | mlx        | torch      |
|------------------------|------------|------------|------------|
| train step (batch=64)  | 1.11 ms (1.1x) | **1.02 ms** | 1.33 ms (1.3x) |
| train step (batch=128) | **1.51 ms** | **1.51 ms** | 1.66 ms (1.1x) |

### Transformer

Single transformer block (multi-head self-attention + FFN) inference

| benchmark                     | sil-gpu    | ggml       | mlx        | torch      |
|-------------------------------|------------|------------|------------|------------|
| transformer (seq=256, d=512)  | **1.53 ms** | 1.88 ms (1.2x) | 1.71 ms (1.1x) | 1.85 ms (1.2x) |
| transformer (seq=256, d=768)  | 2.30 ms (1.1x) | 2.95 ms (1.4x) | **2.16 ms** | 2.99 ms (1.4x) |
| transformer (seq=256, d=1024) | **2.93 ms** | 4.36 ms (1.5x) | 3.21 ms (1.1x) | 4.33 ms (1.5x) |
| transformer (seq=512, d=1024) | **5.32 ms** | 12.06 ms (2.3x) | 6.35 ms (1.2x) | 6.34 ms (1.2x) |

### MNIST Classifier

784->50->10 (sigmoid, MSE loss, SGD). Training: 1 epoch, batch=100. Inference: 10000 images.
Pre-allocates the dataset and uses zero-copy `slice()` for batching.

| benchmark                    | sil-gpu    | mlx        | sil-cpu    | eigen      |
|------------------------------|------------|------------|------------|------------|
| train 1 epoch (60000 images) | 283.83 ms (1.2x) | **239.33 ms** | **44.28 ms** | 205.97 ms (4.7x) |
| inference (10000 images)     | 920 us (1.3x) | **730 us** | **1.58 ms** | 12.98 ms (8.2x) |

### MNIST Autoencoder

784->512->256->64->256->512->784 (sigmoid, MSE loss, SGD). Training: 1 epoch, batch=100. Inference: 10000 images.

| benchmark                    | sil-gpu    | mlx        | torch      | sil-cpu    | eigen      |
|------------------------------|------------|------------|------------|------------|------------|
| train 1 epoch (60000 images) | 1.08 s (1.7x) | **616.37 ms** | 1.70 s (2.8x) | **1.36 s** | 4.36 s (3.2x) |
| inference (10000 images)     | 8.65 ms (1.1x) | **7.98 ms** | 8.85 ms (1.1x) | **33.03 ms** | 314.72 ms (9.5x) |

## Fairness

sil uses lazy evaluation: arithmetic operations build a computation graph instead of
executing immediately. To make benchmarks fair against MLX (which also uses lazy eval
with blocking `mx::eval()`) and torch/ggml (eager), each measured iteration in the
sil benchmarks calls `sil::eval(...)` (or `sil::synchronize()`) to force GPU work to
complete. This ensures the timing reflects real end-to-end work, not just node
construction.

The full bench suite runs in roughly 90 seconds on M1 Pro (`make run`).

## Notes

- **sil-gpu** and **sil-cpu** show Silicon Array running on Metal GPU and Accelerate CPU respectively. Default is GPU (`use_mps()`); switch with `use_cpu()`.
- GPU benchmarks include automatic warmup (sgemm dispatch) at process start to stabilize GPU power state.
- libtorch is a full deep learning framework with autograd, optimizers, and data loaders. The comparison is inherently unfair for raw computation, but illustrative of lightweight vs heavyweight tradeoffs.
- ggml is optimized for LLM inference (quantized matmul, token generation). Its graph-based API adds overhead for simple elementwise ops.
- All GPU benchmarks include synchronization in timing.
- Results report median of multiple trials after warmup and calibration.
- Eigen uses its own BLAS implementation (not Apple Accelerate).
- The fastest entry in each group is highlighted in aqua in terminal output.
