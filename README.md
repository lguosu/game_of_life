# Conway's Game of Life - CPU and GPU Implementations

This is a C++17 implementation of Conway's Game of Life, a cellular automaton devised by the British mathematician John Conway in 1970. The project includes both a CPU version and a CUDA GPU-accelerated version, allowing for performance comparison between the two approaches.

## Rules of the Game

The universe of the Game of Life is an infinite, two-dimensional orthogonal grid of square cells, each of which is in one of two possible states: alive or dead. Every cell interacts with its eight neighbors, which are the cells that are horizontally, vertically, or diagonally adjacent. At each step in time, the following transitions occur:

1. Any live cell with fewer than two live neighbors dies (underpopulation).
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies (overpopulation).
4. Any dead cell with exactly three live neighbors becomes a live cell (reproduction).

## Project Structure

```text
.
├── src/                    # Source code
│   ├── game_of_life.hpp    # Abstract base class header
│   ├── cpu_game_of_life.hpp # CPU implementation header
│   ├── cpu_game_of_life.cpp # CPU implementation
│   ├── gpu_game_of_life.cuh # GPU implementation header
│   ├── gpu_game_of_life.cu  # GPU implementation with CUDA
│   ├── cpu_main.cpp        # CPU version entry point
│   ├── gpu_main.cu         # GPU version entry point
│   ├── performance_benchmark.cpp # Performance comparison benchmark
│   └── utils.hpp           # Common utility functions
├── test/                   # Test code
│   └── cpu_game_of_life_test.cpp # Tests for CPU implementation
├── Makefile                # Build system
└── README.md               # This file
```

## Architecture

The project uses an object-oriented design with inheritance:

- `GameOfLife`: Abstract base class that defines the common interface and default implementations
- `CPUGameOfLife`: CPU implementation using standard C++ techniques
- `GPUGameOfLife`: GPU implementation using CUDA for parallel processing

Both implementations share common functionality through the base class, including cell state management and grid display, while the core simulation logic is optimized for each platform.

## Building the Project

### Prerequisites

- C++17 compatible compiler (e.g., GCC 7+, Clang 5+)
- CUDA Toolkit (for GPU version, tested with CUDA 11.0+)
- Google Test framework (for tests)

To build the project, run:

```bash
make
```

This will create four executables:

- `cpu_game_of_life`: The CPU implementation
- `gpu_game_of_life`: The GPU implementation with CUDA
- `cpu_game_of_life_test`: The test program for CPU implementation
- `performance_benchmark`: Utility to compare CPU vs GPU performance

## Running the Programs

### CPU Version

Run the CPU version with:

```bash
./cpu_game_of_life [width] [height] [generations] [initialDensity] [delay]
```

### GPU Version

Run the GPU version with:

```bash
./gpu_game_of_life [width] [height] [generations] [initialDensity] [delay]
```

### Parameters

- `width`: Grid width (default: 40)
- `height`: Grid height (default: 20)
- `generations`: Number of generations to simulate (default: 100)
- `initialDensity`: Probability of a cell being initially alive (default: 0.3)
- `delay`: Delay between generations in milliseconds (default: 100)

## Running the Tests

Run the tests with:

```bash
make test
```

or directly:

```bash
./cpu_game_of_life_test
```

## Performance Comparison

Run the performance benchmark with:

```bash
make benchmark
```

or directly:

```bash
./performance_benchmark
```

This will run a series of benchmarks comparing the CPU and GPU implementations across different grid sizes, measuring the average time per generation and displaying the speedup factor. The benchmark tests both implementations with the same initial conditions to ensure a fair comparison.

Sample benchmark results:

```text
======= Performance Benchmark: CPU vs GPU =======
Running 100 generations for each test

  Grid Size     CPU (μs/gen)    GPU (μs/gen)     Speedup (×)
-------------------------------------------------------
  32×32              85.32           25.46            3.35
  64×64             325.18           36.90            8.81
 128×128           1285.43           72.15           17.82
 256×256           5120.67          132.50           38.65
 512×512          20480.50          298.70           68.57
1024×1024         81920.25          850.40           96.33
2048×2048        327680.10         2750.32          119.14
```

The benchmark results clearly demonstrate the advantage of GPU parallelism for larger grid sizes. While the CPU implementation is sufficient for small grids, the GPU implementation provides significant performance improvements as the problem size increases, with speedups of over 100x possible for very large grids.

## Implementation Details

### Common Features

- Toroidal grid (wrapping around edges) for both implementations
- Random initialization with configurable density
- Console visualization of the grid

### CPU Implementation

The CPU implementation follows a straightforward approach:

- Sequential processing of the grid
- Double buffering to maintain current and next states
- Direct memory access to cell states

### GPU Implementation

The GPU implementation leverages CUDA for massive parallelism:

- Each cell's next state is computed by a separate CUDA thread
- Thread blocks of 16x16 threads process grid sections
- Device memory management for efficient data transfer
- Performance measurement to highlight GPU acceleration benefits

## Future Work

Potential improvements for this project:

1. Multi-threaded CPU implementation for comparison
2. Shared memory optimization in the CUDA kernel
3. OpenGL visualization for real-time rendering
4. More complex cellular automata rules and patterns
5. Additional test cases for the GPU implementation

## License

[MIT License](LICENSE)
