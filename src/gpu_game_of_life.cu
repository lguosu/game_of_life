#include "gpu_game_of_life.cuh"

#include <iostream>
#include <random>
#include <stdexcept>

// Constructor
GPUGameOfLife::GPUGameOfLife(int width, int height)
    : GameOfLife(width, height), host_grid_(width * height, false) {
  // Implementation will be added in the next step
  std::cout << "GPUGameOfLife constructor called, but implementation is not "
               "complete yet."
            << std::endl;
}

// Destructor
GPUGameOfLife::~GPUGameOfLife() {
  // Implementation will be added in the next step
}

// Initialize with random state
void GPUGameOfLife::Randomize(double alive_probability) {
  // Just initialize the host grid for now
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = 0; i < width_ * height_; ++i) {
    host_grid_[i] = dis(gen) < alive_probability;
  }
}

// Set a specific cell state
void GPUGameOfLife::SetCellState(int x, int y, bool alive) {
  if (x >= 0 && x < width_ && y >= 0 && y < height_) {
    host_grid_[y * width_ + x] = alive;
  }
}

// Get a specific cell state
bool GPUGameOfLife::GetCellState(int x, int y) const {
  if (x >= 0 && x < width_ && y >= 0 && y < height_) {
    return host_grid_[y * width_ + x];
  }
  return false;  // Out of bounds cells are considered dead
}

// Update to next generation
void GPUGameOfLife::NextGeneration() {
  // Stub implementation - no actual GPU computation yet
  std::cout << "GPUGameOfLife::NextGeneration called, but not yet implemented."
            << std::endl;
}

// Print the grid
void GPUGameOfLife::Print(char alive_char, char dead_char) const {
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      std::cout << (GetCellState(x, y) ? alive_char : dead_char);
    }
    std::cout << std::endl;
  }
}

// Get a copy of the current grid
std::vector<bool> GPUGameOfLife::grid() const { return host_grid_; }

// Synchronize memory between host and device (stub implementation)
void GPUGameOfLife::SynchronizeMemory() {
  // Will be implemented in the next step
}

// Error handling (stub)
void GPUGameOfLife::CheckCudaError(cudaError_t error, const char* message) {
  // Will be implemented in the next step
}

// Initialize CUDA memory (stub)
void GPUGameOfLife::InitializeCudaMemory() {
  // Will be implemented in the next step
}

// Release CUDA memory (stub)
void GPUGameOfLife::ReleaseCudaMemory() {
  // Will be implemented in the next step
}

// Copy host to device (stub)
void GPUGameOfLife::CopyToDevice() {
  // Will be implemented in the next step
}

// Copy device to host (stub)
void GPUGameOfLife::CopyToHost() {
  // Will be implemented in the next step
}