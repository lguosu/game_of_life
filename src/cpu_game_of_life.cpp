#include "cpu_game_of_life.hpp"

#include <iostream>

CPUGameOfLife::CPUGameOfLife(int width, int height)
    : GameOfLife(width, height) {}

void CPUGameOfLife::NextGeneration() {
  std::vector<bool> new_grid(width_ * height_);

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int live_neighbors = CountLiveNeighbors(x, y);
      bool current_state = GetCellState(x, y);

      // Apply Conway's Game of Life rules
      if (current_state) {
        // Any live cell with fewer than two live neighbors dies
        // (underpopulation) Any live cell with two or three live neighbors
        // lives on Any live cell with more than three live neighbors dies
        // (overpopulation)
        new_grid[GetIndex(x, y)] = (live_neighbors == 2 || live_neighbors == 3);
      } else {
        // Any dead cell with exactly three live neighbors becomes a live cell
        // (reproduction)
        new_grid[GetIndex(x, y)] = (live_neighbors == 3);
      }
    }
  }

  grid_ = std::move(new_grid);
}

int CPUGameOfLife::CountLiveNeighbors(int x, int y) const {
  int count = 0;

  // Check all 8 neighbors
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      // Skip the cell itself
      if (dx == 0 && dy == 0) {
        continue;
      }

      // Wrap around the edges (toroidal grid)
      int nx = (x + dx + width_) % width_;
      int ny = (y + dy + height_) % height_;

      if (GetCellState(nx, ny)) {
        count++;
      }
    }
  }

  return count;
}

int CPUGameOfLife::GetIndex(int x, int y) const { return y * width_ + x; }