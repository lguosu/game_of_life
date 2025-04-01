#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <thread>

#include "gpu_game_of_life.cuh"
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

  try {
    // Create the GPU game with specified dimensions
    GPUGameOfLife game(width, height);

    // Initialize with random state
    game.Randomize(initial_density);
    game.CopyToDevice();

    ClearScreen();
    std::cout << "GPU Game of Life - Generation: 0"
              << "\n";
    game.Print();

    // Run the simulation for the specified number of generations
    for (int gen = 1; gen < generations; ++gen) {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
      // Measure performance for NextGeneration
      auto start_time = std::chrono::high_resolution_clock::now();
      game.NextGeneration();
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_time - start_time)
                          .count();
      ClearScreen();
      std::cout << "GPU Game of Life - Generation: " << gen << "\n";
      game.Print();
      std::cout << "Generation time: " << duration << " microseconds"
                << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}