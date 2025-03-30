# Conway's Game of Life

This is a C++17 implementation of Conway's Game of Life, a cellular automaton devised by the British mathematician John Conway in 1970. This implementation serves as a reference CPU version before proceeding with a CUDA GPU-accelerated version.

## Rules of the Game

The universe of the Game of Life is an infinite, two-dimensional orthogonal grid of square cells, each of which is in one of two possible states: alive or dead. Every cell interacts with its eight neighbors, which are the cells that are horizontally, vertically, or diagonally adjacent. At each step in time, the following transitions occur:

1. Any live cell with fewer than two live neighbors dies (underpopulation).
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies (overpopulation).
4. Any dead cell with exactly three live neighbors becomes a live cell (reproduction).

## Project Structure

```text
.
├── src/                 # Source code
│   ├── game_of_life.hpp # Main class header
│   ├── game_of_life.cpp # Main class implementation
│   └── main.cpp         # Program entry point
├── test/                # Test code
│   └── game_of_life_test.cpp # Tests for GameOfLife class
├── Makefile             # Build system
└── README.md            # This file
```

## Building the Project

To build the project, run:

```bash
make
```

This will create two executables:

- `game_of_life`: The main program
- `game_of_life_test`: The test program

## Running the Program

Run the main program with:

```bash
./game_of_life [width] [height] [generations] [initialDensity] [delay]
```

Parameters:

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
./game_of_life_test
```

## Implementation Details

The implementation uses a toroidal grid (wrapping around edges) for the simulation. The `GameOfLife` class provides methods to:

- Initialize the grid
- Randomize the state
- Set and get cell states
- Compute the next generation
- Display the grid

The game rules are implemented in the `nextGeneration()` method, which updates all cells simultaneously according to Conway's rules.

## Future Work

This implementation will serve as a reference for a CUDA-accelerated version to explore the speedup potential of GPU parallel computing for this type of cellular automaton simulation.
