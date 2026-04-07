# --- Compiler Settings ---
NVCC = nvcc
CUDA_COMPUTE = -gencode=arch=compute_80,code=sm_80
FLAGS = -O3 $(CUDA_COMPUTE) --std=c++14

# --- Project Paths ---
GRID_DIR   = frnn/csrc/grid
PREFIX_DIR = external/prefix_sum-master
SCAN_DIR   = $(PREFIX_DIR)/parallel-scan
INC        = -I$(GRID_DIR) -I$(PREFIX_DIR) -I$(SCAN_DIR) -I$(PREFIX_DIR)/include

# --- Target ---
TARGET = frnn_standalone

# --- Object Files (ADDED utils.o HERE) ---
OBJS = main.o \
       insert_points.o \
       find_nbrs.o \
       prefix_sum_wrapper.o \
       scan.o \
       kernels.o \
       utils.o

# --- Build Rules ---

all: $(TARGET)

$(TARGET): $(OBJS)
	$(info [Linking]: Creating $(TARGET) executable)
	$(NVCC) $(FLAGS) $(OBJS) -o $(TARGET)

main.o: main.cu
	$(NVCC) $(FLAGS) $(INC) -c main.cu -o main.o

insert_points.o: $(GRID_DIR)/insert_points.cu
	$(NVCC) $(FLAGS) $(INC) -c $(GRID_DIR)/insert_points.cu -o insert_points.o

find_nbrs.o: $(GRID_DIR)/find_nbrs.cu
	$(NVCC) $(FLAGS) $(INC) -c $(GRID_DIR)/find_nbrs.cu -o find_nbrs.o

prefix_sum_wrapper.o: $(PREFIX_DIR)/prefix_sum.cu
	$(NVCC) $(FLAGS) $(INC) -c $(PREFIX_DIR)/prefix_sum.cu -o prefix_sum_wrapper.o

scan.o: $(SCAN_DIR)/scan.cu
	$(NVCC) $(FLAGS) $(INC) -c $(SCAN_DIR)/scan.cu -o scan.o

kernels.o: $(SCAN_DIR)/kernels.cu
	$(NVCC) $(FLAGS) $(INC) -c $(SCAN_DIR)/kernels.cu -o kernels.o

# --- NEW RULE FOR UTILS ---
utils.o: $(SCAN_DIR)/utils.cpp
	$(NVCC) $(FLAGS) $(INC) -c $(SCAN_DIR)/utils.cpp -o utils.o

clean:
	rm -f *.o $(TARGET)