#pragma once

#include <iostream>
#include <random>
#include <vector>

// Abstract base class for Game of Life
class GameOfLife {
 public:
  // Constructor
  GameOfLife(int width, int height)
      : width_(width), height_(height), grid_(width * height, false) {}

  // Virtual destructor
  virtual ~GameOfLife() = default;

  // Initialize with random state (alive cells with the given probability)
  // Optional seed parameter for reproducible randomization
  virtual void Randomize(double alive_probability = 0.3, unsigned seed = 0) {
    std::random_device rd;
    std::mt19937 gen(seed != 0 ? seed : rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < width_ * height_; ++i) {
      grid_[i] = dis(gen) < alive_probability;
    }
  }

  // Set a specific cell state (x, y coordinates and state)
  void SetCellState(int x, int y, bool alive) {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
      grid_[y * width_ + x] = alive;
    }
  }

  // Get a specific cell state
  [[nodiscard]] bool GetCellState(int x, int y) const {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
      return grid_[y * width_ + x];
    }
    return false;  // Out of bounds cells are considered dead
  }

  // Update the grid to the next generation according to Game of Life rules
  virtual void NextGeneration() = 0;

  // Print the current state of the grid to the console
  virtual void Print(char alive_char = 'O', char dead_char = '.') const {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        std::cout << (GetCellState(x, y) ? alive_char : dead_char);
      }
      std::cout << std::endl;
    }
  }

  // Get the width of the grid
  [[nodiscard]] int width() const { return width_; }

  // Get the height of the grid
  [[nodiscard]] int height() const { return height_; }

  // Get a copy of the current grid
  [[nodiscard]] virtual std::vector<bool> grid() const { return grid_; }

 protected:
  int width_;
  int height_;
  std::vector<bool> grid_;  // The grid for storing cell states
};