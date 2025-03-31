#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "game_of_life.hpp"

// GPU implementation of Game of Life
class GPUGameOfLife : public GameOfLife {
 public:
  // Constructor that initializes a grid with given dimensions
  GPUGameOfLife(int width, int height);

  // Destructor to release CUDA resources
  ~GPUGameOfLife() override;

  // Initialize with random state (alive cells with the given probability)
  void Randomize(double alive_probability = 0.3) override;

  // Set a specific cell state (x, y coordinates and state)
  void SetCellState(int x, int y, bool alive) override;

  // Get a specific cell state
  [[nodiscard]] bool GetCellState(int x, int y) const override;

  // Update the grid to the next generation according to Game of Life rules
  void NextGeneration() override;

  // Print the current state of the grid to the console
  void Print(char alive_char = 'O', char dead_char = '.') const override;

  // Get a copy of the current grid
  [[nodiscard]] std::vector<bool> grid() const override;

  // Synchronize device memory with host
  void SynchronizeMemory();

 private:
  std::vector<bool> host_grid_;  // Host-side grid for I/O operations

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

  // Copy grid from host to device
  void CopyToDevice();

  // Copy grid from device to host
  void CopyToHost();
};