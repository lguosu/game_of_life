#pragma once

#include <vector>

#include "game_of_life.hpp"

// CPU implementation of Game of Life
class CPUGameOfLife : public GameOfLife {
 public:
  // Constructor that initializes a grid with given dimensions
  CPUGameOfLife(int width, int height);

  // Update the grid to the next generation according to Game of Life rules
  void NextGeneration() override;

 private:
  // Helper method to count live neighbors of a cell
  [[nodiscard]] int CountLiveNeighbors(int x, int y) const;

  // Helper method to get index from x, y coordinates
  [[nodiscard]] int GetIndex(int x, int y) const;
};