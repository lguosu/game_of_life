#pragma once

#include <vector>

// Abstract base class for Game of Life
class GameOfLife {
 public:
  // Constructor
  GameOfLife(int width, int height) : width_(width), height_(height) {}

  // Virtual destructor
  virtual ~GameOfLife() = default;

  // Initialize with random state (alive cells with the given probability)
  virtual void Randomize(double alive_probability = 0.3) = 0;

  // Set a specific cell state (x, y coordinates and state)
  virtual void SetCellState(int x, int y, bool alive) = 0;

  // Get a specific cell state
  [[nodiscard]] virtual bool GetCellState(int x, int y) const = 0;

  // Update the grid to the next generation according to Game of Life rules
  virtual void NextGeneration() = 0;

  // Print the current state of the grid to the console
  virtual void Print(char alive_char = 'O', char dead_char = '.') const = 0;

  // Get the width of the grid
  [[nodiscard]] int width() const { return width_; }

  // Get the height of the grid
  [[nodiscard]] int height() const { return height_; }

  // Get a copy of the current grid
  [[nodiscard]] virtual std::vector<bool> grid() const = 0;

 protected:
  int width_;
  int height_;
};