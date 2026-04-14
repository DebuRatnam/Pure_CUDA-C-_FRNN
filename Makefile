NVCC = nvcc
# -O3 for optimization, sm_80 for Perlmutter A100s
FLAGS = -O3 -arch=sm_80 --std=c++14 -Xcompiler -fPIC
INC = -I. -Ifrnn/csrc/grid -Iexternal/prefix_sum-master -Iexternal/prefix_sum-master/parallel-scan -Iexternal/prefix_sum-master/include

# Python / PyBind11 Detection
PY_FLAGS = $(shell python3 -m pybind11 --includes)
PY_EXT   = $(shell python3-config --extension-suffix)

# These are the shared object files both apps need
CORE_OBJS = insert_points.o find_nbrs.o prefix_sum_wrapper.o scan.o kernels.o utils.o bruteforce.o

# Build everything by default (Added python_wrapper to default)
all: frnn_test frnn_bench python_wrapper

# Program 1: The Validator
frnn_test: test_frnn.o $(CORE_OBJS)
	$(NVCC) $(FLAGS) test_frnn.o $(CORE_OBJS) -o frnn_test

# Program 2: The Benchmarker
frnn_bench: benchmark_frnn.o $(CORE_OBJS)
	$(NVCC) $(FLAGS) benchmark_frnn.o $(CORE_OBJS) -o frnn_bench

# --- NEW: The Python Interface ---
# This compiles the engine manager and links it with all core CUDA objects
python_wrapper: frnn_engine.o $(CORE_OBJS)
	$(NVCC) $(FLAGS) -shared frnn_engine.o $(CORE_OBJS) -o frnn_cuda$(PY_EXT)

frnn_engine.o: python_interface/frnn_engine.cu
	$(NVCC) $(FLAGS) $(INC) $(PY_FLAGS) -c python_interface/frnn_engine.cu -o frnn_engine.o

# How to compile the main files
test_frnn.o: test_frnn.cu
	$(NVCC) $(FLAGS) $(INC) -c test_frnn.cu -o test_frnn.o

benchmark_frnn.o: benchmark_frnn.cu
	$(NVCC) $(FLAGS) $(INC) -c benchmark_frnn.cu -o benchmark_frnn.o

# How to compile the Engine files
insert_points.o: frnn/csrc/grid/insert_points.cu
	$(NVCC) $(FLAGS) $(INC) -c frnn/csrc/grid/insert_points.cu -o insert_points.o

find_nbrs.o: frnn/csrc/grid/find_nbrs.cu
	$(NVCC) $(FLAGS) $(INC) -c frnn/csrc/grid/find_nbrs.cu -o find_nbrs.o

bruteforce.o: frnn/csrc/bruteforce/bruteforce.cu
	$(NVCC) $(FLAGS) $(INC) -c frnn/csrc/bruteforce/bruteforce.cu -o bruteforce.o

prefix_sum_wrapper.o: external/prefix_sum-master/prefix_sum.cu
	$(NVCC) $(FLAGS) $(INC) -c external/prefix_sum-master/prefix_sum.cu -o prefix_sum_wrapper.o

scan.o: external/prefix_sum-master/parallel-scan/scan.cu
	$(NVCC) $(FLAGS) $(INC) -c external/prefix_sum-master/parallel-scan/scan.cu -o scan.o

kernels.o: external/prefix_sum-master/parallel-scan/kernels.cu
	$(NVCC) $(FLAGS) $(INC) -c external/prefix_sum-master/parallel-scan/kernels.cu -o kernels.o

utils.o: external/prefix_sum-master/parallel-scan/utils.cpp
	$(NVCC) $(FLAGS) $(INC) -c external/prefix_sum-master/parallel-scan/utils.cpp -o utils.o

# Updated Clean Rule
clean:
	rm -f *.o frnn_test frnn_bench frnn_cuda*.so