#include "game_of_life.hpp"

GameOfLife::GameOfLife(int width, int height)
    : width_(width), height_(height), grid_(width * height, false) {}

void GameOfLife::Randomize(double alive_probability) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = 0; i < width_ * height_; ++i) {
    grid_[i] = dis(gen) < alive_probability;
  }
}

void GameOfLife::SetCellState(int x, int y, bool alive) {
  if (x >= 0 && x < width_ && y >= 0 && y < height_) {
    grid_[GetIndex(x, y)] = alive;
  }
}

bool GameOfLife::GetCellState(int x, int y) const {
  if (x >= 0 && x < width_ && y >= 0 && y < height_) {
    return grid_[GetIndex(x, y)];
  }
  return false;  // Out of bounds cells are considered dead
}

void GameOfLife::NextGeneration() {
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

void GameOfLife::Print(char alive_char, char dead_char) const {
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      std::cout << (GetCellState(x, y) ? alive_char : dead_char);
    }
    std::cout << std::endl;
  }
}

int GameOfLife::CountLiveNeighbors(int x, int y) const {
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

int GameOfLife::GetIndex(int x, int y) const { return y * width_ + x; }