#include "gpu_game_of_life.cuh"

#include <iostream>
#include <random>
#include <stdexcept>

#include <cuda_runtime.h>

// CUDA kernel to compute the next generation
__global__ void gameOfLifeKernel(const unsigned char* current_grid,
                                 unsigned char* next_grid, int width,
                                 int height) {
  // Calculate global thread indices
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Skip threads outside grid boundaries
  if (x >= width || y >= height) return;

  int idx = y * width + x;

  // Count live neighbors with wrap-around
  int live_neighbors = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      // Skip the cell itself
      if (dx == 0 && dy == 0) continue;

      // Calculate neighbor coordinates with wrap-around
      int nx = (x + dx + width) % width;
      int ny = (y + dy + height) % height;

      // Add 1 if the neighbor is alive
      live_neighbors += current_grid[ny * width + nx];
    }
  }

  // Apply Conway's Game of Life rules
  bool current_state = current_grid[idx] != 0;
  bool next_state;

  if (current_state) {
    // Any live cell with 2 or 3 live neighbors survives
    next_state = (live_neighbors == 2 || live_neighbors == 3);
  } else {
    // Any dead cell with exactly 3 live neighbors becomes alive
    next_state = (live_neighbors == 3);
  }

  // Update the next grid
  next_grid[idx] = next_state ? 1 : 0;
}

// Constructor
GPUGameOfLife::GPUGameOfLife(size_t width, size_t height)
    : GameOfLife(width, height) {
  // Calculate appropriate CUDA block and grid dimensions
  // Block dimensions - aim for 16x16 threads per block (common for 2D problems)
  block_dim_ = dim3(16, 16, 1);

  // Grid dimensions - ensure we have enough blocks to cover the entire grid
  grid_dim_ = dim3((width + block_dim_.x - 1) / block_dim_.x,
                   (height + block_dim_.y - 1) / block_dim_.y, 1);

  // Initialize CUDA memory
  InitializeCudaMemory();
}

// Destructor
GPUGameOfLife::~GPUGameOfLife() { ReleaseCudaMemory(); }

// Update to next generation
void GPUGameOfLife::NextGeneration() {
  // Launch the CUDA kernel to calculate the next generation
  gameOfLifeKernel<<<grid_dim_, block_dim_>>>(d_current_grid_, d_next_grid_,
                                              width_, height_);

  // Check for kernel launch errors
  CheckCudaError(cudaGetLastError(), "Failed to launch Game of Life kernel");

  // Wait for the kernel to complete
  CheckCudaError(cudaDeviceSynchronize(), "Failed to synchronize CUDA device");

  // Swap current and next grid pointers
  unsigned char* temp = d_current_grid_;
  d_current_grid_ = d_next_grid_;
  d_next_grid_ = temp;

  // Update host memory with the new state from device memory
  CopyToHost();
}

// Error handling for CUDA operations
void GPUGameOfLife::CheckCudaError(cudaError_t error, const char* message) {
  if (error != cudaSuccess) {
    std::string error_message =
        std::string(message) + ": " + std::string(cudaGetErrorString(error));
    throw std::runtime_error(error_message);
  }
}

// Initialize CUDA memory
void GPUGameOfLife::InitializeCudaMemory() {
  // Allocate device memory for current and next grids
  size_t size = width_ * height_ * sizeof(unsigned char);

  CheckCudaError(cudaMalloc(&d_current_grid_, size),
                 "Failed to allocate device memory for current grid");

  CheckCudaError(cudaMalloc(&d_next_grid_, size),
                 "Failed to allocate device memory for next grid");

  // Initialize device memory to zeros
  CheckCudaError(cudaMemset(d_current_grid_, 0, size),
                 "Failed to initialize current grid");

  CheckCudaError(cudaMemset(d_next_grid_, 0, size),
                 "Failed to initialize next grid");
}

// Release CUDA memory
void GPUGameOfLife::ReleaseCudaMemory() {
  if (d_current_grid_ != nullptr) {
    cudaFree(d_current_grid_);
    d_current_grid_ = nullptr;
  }

  if (d_next_grid_ != nullptr) {
    cudaFree(d_next_grid_);
    d_next_grid_ = nullptr;
  }
}

// Copy host to device
void GPUGameOfLife::CopyToDevice() {
  // Convert std::vector<bool> to unsigned char array for CUDA
  std::vector<unsigned char> temp_grid(width_ * height_);
  for (int i = 0; i < width_ * height_; ++i) {
    temp_grid[i] = grid_[i] ? 1 : 0;
  }

  // Copy to device
  size_t size = width_ * height_ * sizeof(unsigned char);
  CheckCudaError(cudaMemcpy(d_current_grid_, temp_grid.data(), size,
                            cudaMemcpyHostToDevice),
                 "Failed to copy grid from host to device");
}

// Copy device to host
void GPUGameOfLife::CopyToHost() {
  // Create a temporary buffer for device data
  std::vector<unsigned char> temp_grid(width_ * height_);

  // Copy from device to temp buffer
  size_t size = width_ * height_ * sizeof(unsigned char);
  CheckCudaError(cudaMemcpy(temp_grid.data(), d_current_grid_, size,
                            cudaMemcpyDeviceToHost),
                 "Failed to copy grid from device to host");

  // Convert unsigned char to bool for grid_
  for (int i = 0; i < width_ * height_; ++i) {
    grid_[i] = temp_grid[i] != 0;
  }
}