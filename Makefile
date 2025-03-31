CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O3
NVCCFLAGS = -std=c++17 -O3
LDFLAGS = 

# CUDA paths - adjust these for your system
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIBDIR = /usr/local/cuda/lib64
CUDA_LIBS = -lcudart

# Google Test flags
GTEST_CFLAGS = -I/usr/src/googletest/googletest/include
GTEST_LIBS = -lgtest -lgtest_main -pthread

SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

# Source files
CPU_SRC = $(SRC_DIR)/cpu_game_of_life.cpp
GPU_SRC = $(SRC_DIR)/gpu_game_of_life.cu
CPU_MAIN_SRC = $(SRC_DIR)/cpu_main.cpp
GPU_MAIN_SRC = $(SRC_DIR)/gpu_main.cu
CPU_TEST_SRC = $(TEST_DIR)/cpu_game_of_life_test.cpp

# Object files
CPU_MAIN_OBJ = $(BUILD_DIR)/cpu_main.o
CPU_OBJ = $(BUILD_DIR)/cpu_game_of_life.o
GPU_OBJ = $(BUILD_DIR)/gpu_game_of_life.o
GPU_MAIN_OBJ = $(BUILD_DIR)/gpu_main.o
CPU_TEST_OBJ = $(BUILD_DIR)/cpu_game_of_life_test.o

# Executables
CPU_TARGET = cpu_game_of_life
GPU_TARGET = gpu_game_of_life
CPU_TEST_TARGET = cpu_game_of_life_test

.PHONY: all clean test

all: $(CPU_TARGET) $(GPU_TARGET) $(CPU_TEST_TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CPU game of life library
$(CPU_OBJ): $(CPU_SRC) $(SRC_DIR)/cpu_game_of_life.hpp $(SRC_DIR)/game_of_life.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile GPU game of life library
$(GPU_OBJ): $(GPU_SRC) $(SRC_DIR)/gpu_game_of_life.cuh $(SRC_DIR)/game_of_life.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INCLUDE) -c $< -o $@

# Compile CPU main program
$(CPU_MAIN_OBJ): $(CPU_MAIN_SRC) $(SRC_DIR)/cpu_game_of_life.hpp $(SRC_DIR)/utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile GPU main program
$(GPU_MAIN_OBJ): $(GPU_MAIN_SRC) $(SRC_DIR)/gpu_game_of_life.cuh $(SRC_DIR)/utils.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INCLUDE) -c $< -o $@

# Compile test program
$(CPU_TEST_OBJ): $(CPU_TEST_SRC) $(SRC_DIR)/cpu_game_of_life.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(GTEST_CFLAGS) -c $< -o $@

# Link CPU main program
$(CPU_TARGET): $(CPU_MAIN_OBJ) $(CPU_OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@

# Link GPU main program
$(GPU_TARGET): $(GPU_MAIN_OBJ) $(GPU_OBJ)
	$(NVCC) $(LDFLAGS) $^ -L$(CUDA_LIBDIR) $(CUDA_LIBS) -o $@

# Link test program
$(CPU_TEST_TARGET): $(CPU_TEST_OBJ) $(CPU_OBJ)
	$(CXX) $(LDFLAGS) $^ $(GTEST_LIBS) -o $@

test: $(CPU_TEST_TARGET)
	./$(CPU_TEST_TARGET)

# Clean up
clean:
	rm -rf $(BUILD_DIR) $(CPU_TARGET) $(GPU_TARGET) $(CPU_TEST_TARGET) 