#pragma once

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

class GameOfLife {
 public:
  // Constructor that initializes a grid with given dimensions
  GameOfLife(int width, int height);

  // Initialize with random state (alive cells with the given probability)
  void Randomize(double alive_probability = 0.3);

  // Set a specific cell state (x, y coordinates and state)
  void SetCellState(int x, int y, bool alive);

  // Get a specific cell state
  [[nodiscard]] bool GetCellState(int x, int y) const;

  // Update the grid to the next generation according to Game of Life rules
  void NextGeneration();

  // Print the current state of the grid to the console
  void Print(char alive_char = 'O', char dead_char = '.') const;

  // Get the width of the grid
  [[nodiscard]] int width() const { return width_; }

  // Get the height of the grid
  [[nodiscard]] int height() const { return height_; }

  // Get a copy of the current grid
  [[nodiscard]] std::vector<bool> grid() const { return grid_; }

 private:
  int width_;
  int height_;
  std::vector<bool> grid_;

  // Helper method to count live neighbors of a cell
  [[nodiscard]] int CountLiveNeighbors(int x, int y) const;

  // Helper method to get index from x, y coordinates
  [[nodiscard]] int GetIndex(int x, int y) const;
};