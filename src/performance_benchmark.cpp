#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cpu_game_of_life.hpp"
#include "gpu_game_of_life.cuh"

namespace {
// Benchmark a single run
template <typename GameType>
double benchmark_single_run(GameType& game, int generations) {
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int gen = 0; gen < generations; ++gen) {
    game.NextGeneration();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();

  return static_cast<double>(duration) / generations;
}

// Run benchmarks for various grid sizes
void run_benchmarks() {
  const int generations = 100;
  const double initial_density = 0.3;
  const std::vector<size_t> grid_sizes = {32, 64, 128, 256, 512, 1024, 2048};

  std::cout << "======= Performance Benchmark: CPU vs GPU ======="
            << "\n";
  std::cout << "Running " << generations << " generations for each test"
            << "\n";
  std::cout << "\n";

  std::cout << std::setw(10) << "Grid Size" << std::setw(15) << "CPU (μs/gen)"
            << std::setw(15) << "GPU (μs/gen)" << std::setw(15) << "Speedup (x)"
            << "\n";
  std::cout << std::string(55, '-') << "\n";

  for (const auto& size : grid_sizes) {
    try {
      // Create square grids of the current size
      CPUGameOfLife cpu_game(size, size);
      GPUGameOfLife gpu_game(size, size);

      // Initialize both games with the same random seed for fairness
      const unsigned seed = static_cast<unsigned>(
          std::chrono::system_clock::now().time_since_epoch().count());
      cpu_game.Randomize(initial_density, seed);
      gpu_game.Randomize(initial_density, seed);

      // Copy initial state to GPU
      gpu_game.CopyToDevice();

      // Benchmark CPU version
      const double cpu_time = benchmark_single_run(cpu_game, generations);

      // Benchmark GPU version
      const double gpu_time = benchmark_single_run(gpu_game, generations);

      // Calculate speedup
      const double speedup = cpu_time / gpu_time;

      // Print results
      std::cout << std::setw(5) << size << "x" << std::setw(4) << size
                << std::setw(15) << std::fixed << std::setprecision(2)
                << cpu_time << std::setw(15) << std::fixed
                << std::setprecision(2) << gpu_time << std::setw(15)
                << std::fixed << std::setprecision(2) << speedup << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Error with grid size " << size << ": " << e.what() << "\n";
    }
  }
}
}  // namespace

int main() {
  try {
    run_benchmarks();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}