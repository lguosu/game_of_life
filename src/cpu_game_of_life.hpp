#pragma once

#include <vector>

#include "game_of_life.hpp"

// CPU implementation of Game of Life
class CPUGameOfLife : public GameOfLife {
 public:
  // Constructor that initializes a grid with given dimensions
  CPUGameOfLife(int width, int height);

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
  [[nodiscard]] std::vector<bool> grid() const override { return grid_; }

 private:
  std::vector<bool> grid_;

  // Helper method to count live neighbors of a cell
  [[nodiscard]] int CountLiveNeighbors(int x, int y) const;

  // Helper method to get index from x, y coordinates
  [[nodiscard]] int GetIndex(int x, int y) const;
};