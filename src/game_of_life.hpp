#pragma once

#include <iostream>
#include <random>
#include <vector>

// Abstract base class for Game of Life
class GameOfLife {
 public:
  // Constructor
  GameOfLife(size_t width, size_t height)
      : width_(width), height_(height), grid_(width * height, false) {}

  // Virtual destructor
  virtual ~GameOfLife() = default;

  // Initialize with random state (alive cells with the given probability)
  // Optional seed parameter for reproducible randomization
  void Randomize(double alive_probability = 0.3, unsigned seed = 0) {
    std::random_device rd;
    std::mt19937 gen(seed != 0 ? seed : rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < width_ * height_; ++i) {
      grid_[i] = dis(gen) < alive_probability;
    }
  }

  // Set a specific cell state (x, y coordinates and state)
  void SetCellState(size_t x, size_t y, bool alive) {
    if (x < width_ && y < height_) {
      grid_[(y * width_) + x] = alive;
    }
  }

  // Get a specific cell state
  [[nodiscard]] bool GetCellState(size_t x, size_t y) const {
    if (x < width_ && y < height_) {
      return grid_[(y * width_) + x];
    }
    return false;  // Out of bounds cells are considered dead
  }

  // Update the grid to the next generation according to Game of Life rules
  virtual void NextGeneration() = 0;

  // Print the current state of the grid to the console
  void Print(char alive_char = 'X', char dead_char = ' ') const {
    for (size_t y = 0; y < height_; ++y) {
      for (size_t x = 0; x < width_; ++x) {
        std::cout << (GetCellState(x, y) ? alive_char : dead_char);
      }
      std::cout << "\n";
    }
  }

  // Get the width of the grid
  [[nodiscard]] size_t width() const { return width_; }

  // Get the height of the grid
  [[nodiscard]] size_t height() const { return height_; }

 protected:
  // Get a modifiable reference to the current grid.
  [[nodiscard]] virtual std::vector<bool>& grid() { return grid_; }

  /// Set the internal grid by moving the provided vector into the member
  /// variable.
  /// @param grid The grid to be moved into the member variable.
  virtual void set_grid(std::vector<bool>&& grid) { grid_ = std::move(grid); }

 private:
  size_t width_;
  size_t height_;
  std::vector<bool> grid_;  // The grid for storing cell states
};