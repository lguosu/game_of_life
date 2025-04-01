#include "cpu_game_of_life.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "game_of_life.hpp"

CPUGameOfLife::CPUGameOfLife(size_t width, size_t height)
    : GameOfLife(width, height) {}

void CPUGameOfLife::NextGeneration() {
  std::vector<bool> new_grid(width() * height());

  for (size_t y = 0; y < height(); ++y) {
    for (size_t x = 0; x < width(); ++x) {
      const size_t live_neighbors = CountLiveNeighbors(x, y);
      const bool current_state = GetCellState(x, y);

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

  set_grid(std::move(new_grid));
}

size_t CPUGameOfLife::CountLiveNeighbors(size_t x, size_t y) const {
  size_t count = 0;

  // Check all 8 neighbors
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      // Skip the cell itself
      if (dx == 0 && dy == 0) {
        continue;
      }

      // Wrap around the edges (toroidal grid)
      const size_t nx = (x + dx + width()) % width();
      const size_t ny = (y + dy + height()) % height();

      if (GetCellState(nx, ny)) {
        count++;
      }
    }
  }

  return count;
}
