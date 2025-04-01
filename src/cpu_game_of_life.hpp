#pragma once

#include <vector>

#include "game_of_life.hpp"

// CPU implementation of Game of Life
class CPUGameOfLife : public GameOfLife {
 public:
  // Constructor that initializes a grid with given dimensions
  CPUGameOfLife(size_t width, size_t height);

  // Update the grid to the next generation according to Game of Life rules
  void NextGeneration() override;

 private:
  // Helper method to count live neighbors of a cell
  [[nodiscard]] size_t CountLiveNeighbors(size_t x, size_t y) const;

  // Helper method to get index from x, y coordinates
  [[nodiscard]] size_t GetIndex(size_t x, size_t y) const {
    return (y * width()) + x;
  }
};