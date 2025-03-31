#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "cpu_game_of_life.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
  // Default values
  int width = 40;
  int height = 20;
  int generations = 100;
  double initial_density = 0.3;
  int delay = 100;  // milliseconds

  // Parse command line arguments if provided
  if (argc > 1) {
    width = std::atoi(argv[1]);
  }
  if (argc > 2) {
    height = std::atoi(argv[2]);
  }
  if (argc > 3) {
    generations = std::atoi(argv[3]);
  }
  if (argc > 4) {
    initial_density = std::atof(argv[4]);
  }
  if (argc > 5) {
    delay = std::atoi(argv[5]);
  }

  // Create the game with specified dimensions
  CPUGameOfLife game(width, height);

  // Initialize with random state
  game.Randomize(initial_density);

  // Run the simulation for the specified number of generations
  for (int gen = 0; gen < generations; ++gen) {
    ClearScreen();
    std::cout << "CPU Game of Life - Generation: " << gen << std::endl;
    game.Print();

    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    game.NextGeneration();
  }

  return 0;
}