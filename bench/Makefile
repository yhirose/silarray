CXX      = clang++
CXXFLAGS = -std=c++23 -O2 -I../include -DACCELERATE_NEW_LAPACK \
           -framework Foundation -framework Metal \
           -framework Accelerate -framework MetalPerformanceShaders \
           -framework MetalPerformanceShadersGraph

INC = ../include/types.h ../include/objc.h ../include/unified_memory.h ../include/device.h ../include/cpu.h ../include/gpu.h ../include/array.h ../include/silarray.h

# External libraries (auto-detected — a library is enabled only if a required
# header is actually on disk, so a partial or missing install doesn't break
# the build). Probe files were picked because they are small, stable across
# versions, and live directly under the include prefix.

EIGEN_PROBE = /opt/homebrew/include/eigen3/Eigen/Core
EIGEN_FLAGS = $(if $(wildcard $(EIGEN_PROBE)), \
              -I/opt/homebrew/include -DBENCH_HAS_EIGEN)

MLX_PROBE   = /opt/homebrew/include/mlx/mlx.h
MLX_FLAGS   = $(if $(wildcard $(MLX_PROBE)), \
              -I/opt/homebrew/include -DBENCH_HAS_MLX \
              -L/opt/homebrew/lib -lmlx -rpath /opt/homebrew/lib)

TORCH_PREFIX = /opt/homebrew/opt/pytorch
TORCH_PROBE  = $(TORCH_PREFIX)/include/torch/csrc/api/include/torch/torch.h
TORCH_FLAGS  = $(if $(wildcard $(TORCH_PROBE)), \
               -I$(TORCH_PREFIX)/include \
               -I$(TORCH_PREFIX)/include/torch/csrc/api/include \
               -DBENCH_HAS_LIBTORCH \
               -L$(TORCH_PREFIX)/lib -ltorch -ltorch_cpu -lc10 \
               -rpath $(TORCH_PREFIX)/lib)

# ggml with Metal backend enabled. The upstream Homebrew formula is CPU-only
# (no libggml-metal.dylib), so we ship a self-contained install at a
# persistent path that contains all 5 backend libs.
# Build procedure is documented in bench/README.md.
GGML_PREFIX = $(HOME)/Projects/ggml-install
GGML_PROBE  = $(GGML_PREFIX)/include/ggml-alloc.h
GGML_FLAGS  = $(if $(wildcard $(GGML_PROBE)), \
              -I$(GGML_PREFIX)/include -DBENCH_HAS_GGML \
              -L$(GGML_PREFIX)/lib -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas \
              -rpath $(GGML_PREFIX)/lib \
              -framework Foundation -framework Metal -framework MetalPerformanceShaders)

BENCH_FLAGS = $(CXXFLAGS) $(EIGEN_FLAGS) $(MLX_FLAGS) $(TORCH_FLAGS) $(GGML_FLAGS)

# Summary printed at the top of `make` — shows which libraries were detected.
$(info silarray bench detected libs:)
$(info   eigen  = $(if $(EIGEN_FLAGS),yes,no (probe: $(EIGEN_PROBE))))
$(info   mlx    = $(if $(MLX_FLAGS),yes,no (probe: $(MLX_PROBE))))
$(info   torch  = $(if $(TORCH_FLAGS),yes,no (probe: $(TORCH_PROBE))))
$(info   ggml   = $(if $(GGML_FLAGS),yes,no (probe: $(GGML_PROBE))))

# Micro benchmarks — single operation throughput
MICRO_SRCS = $(wildcard micro/*.cpp)
MICRO_TARGETS = $(patsubst micro/%.cpp,micro/%,$(MICRO_SRCS))

# Composite benchmarks — multi-operation workloads
COMPOSITE_SRCS = $(wildcard composite/*.cpp)
COMPOSITE_TARGETS = $(patsubst composite/%.cpp,composite/%,$(COMPOSITE_SRCS))

# MNIST benchmarks — end-to-end with real data
MNIST_SRCS = $(wildcard mnist/*.cpp)
MNIST_TARGETS = $(patsubst mnist/%.cpp,mnist/%,$(MNIST_SRCS))

TARGETS = $(MICRO_TARGETS) $(COMPOSITE_TARGETS) $(MNIST_TARGETS)

all: $(TARGETS)

micro: $(MICRO_TARGETS)

composite: $(COMPOSITE_TARGETS)

mnist: $(MNIST_TARGETS)

micro/%: micro/%.cpp bench_common.h $(INC) Makefile
	$(CXX) -o $@ $(BENCH_FLAGS) $<

composite/%: composite/%.cpp bench_common.h $(INC) Makefile
	$(CXX) -o $@ $(BENCH_FLAGS) $<

mnist/%: mnist/%.cpp mnist/mnist_data.h bench_common.h $(INC) Makefile
	$(CXX) -o $@ $(BENCH_FLAGS) $<

# Run all or selected benchmarks
run: $(if $(BENCH),$(BENCH),$(TARGETS))
	@for t in $^; do echo "\n=== $$t ==="; ./$$t; done

run-micro: $(MICRO_TARGETS)
	@for t in $^; do echo "\n=== $$t ==="; ./$$t; done

run-composite: $(COMPOSITE_TARGETS)
	@for t in $^; do echo "\n=== $$t ==="; ./$$t; done

run-mnist: $(MNIST_TARGETS)
	@for t in $^; do echo "\n=== $$t ==="; ./$$t; done

csv: $(if $(BENCH),$(BENCH),$(TARGETS))
	@for t in $^; do ./$$t --csv; done

table: $(if $(BENCH),$(BENCH),$(TARGETS))
	@for t in $^; do ./$$t --table; done

clean:
	rm -f $(TARGETS)

.PHONY: all micro composite mnist run run-micro run-composite run-mnist csv table clean
