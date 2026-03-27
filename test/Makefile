CXX = clang++
CXXFLAGS = -std=c++23 -O2 -I../include -DACCELERATE_NEW_LAPACK -framework Foundation -framework Metal -framework Accelerate -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph

INC = ../include/types.h ../include/objc.h ../include/unified_memory.h ../include/device.h ../include/cpu.h ../include/gpu.h ../include/array.h ../include/silarray.h

TEST_SRC = test_array.cpp test_examples.cpp test_perceptron.cpp test_2lnn.cpp test.cpp
MNIST_SRC = mnist.cpp

all : test mnist
	@echo "=== Auto mode ==="
	./test
	@echo "=== GPU mode ==="
	./test --gpu
	@echo "=== CPU mode ==="
	./test --cpu
	./mnist

test : $(INC) $(TEST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(TEST_SRC)

mnist : $(INC) $(MNIST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(MNIST_SRC)

clean:
	rm -rf test mnist
