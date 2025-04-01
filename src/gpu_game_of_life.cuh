#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "game_of_life.hpp"

// GPU implementation of Game of Life
class GPUGameOfLife : public GameOfLife {
 public:
  // Constructor that initializes a grid with given dimensions
  GPUGameOfLife(size_t width, size_t height);

  // Destructor to release CUDA resources
  ~GPUGameOfLife() override;

  // Update the grid to the next generation according to Game of Life rules
  void NextGeneration() override;

  // Copy grid from host to device
  void CopyToDevice();

 private:
  // Device memory pointers
  unsigned char* d_current_grid_ = nullptr;
  unsigned char* d_next_grid_ = nullptr;

  // Block and grid dimensions for CUDA kernels
  dim3 block_dim_;
  dim3 grid_dim_;

  // CUDA error handling
  void CheckCudaError(cudaError_t error, const char* message);

  // Initialize CUDA memory
  void InitializeCudaMemory();

  // Release CUDA memory
  void ReleaseCudaMemory();

  // Copy grid from device to host
  void CopyToHost();
};