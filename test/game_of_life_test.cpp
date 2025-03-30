#include "../src/game_of_life.hpp"

#include <gtest/gtest.h>

// Test basic grid initialization
TEST(GameOfLifeTest, Initialization) {
  GameOfLife game(10, 10);

  // Check that the grid is initially empty
  for (int y = 0; y < game.height(); ++y) {
    for (int x = 0; x < game.width(); ++x) {
      EXPECT_FALSE(game.GetCellState(x, y));
    }
  }

  // Set some cells and check they were set correctly
  game.SetCellState(1, 1, true);
  game.SetCellState(2, 2, true);
  game.SetCellState(3, 3, true);

  EXPECT_TRUE(game.GetCellState(1, 1));
  EXPECT_TRUE(game.GetCellState(2, 2));
  EXPECT_TRUE(game.GetCellState(3, 3));
  EXPECT_FALSE(game.GetCellState(0, 0));
}

// Test the rules of Game of Life
TEST(GameOfLifeTest, Rules) {
  // Test underpopulation (fewer than 2 neighbors)
  {
    GameOfLife game(3, 3);
    game.SetCellState(1, 1, true);
    game.NextGeneration();
    EXPECT_FALSE(game.GetCellState(1, 1));  // Should die from underpopulation
  }

  // Test survival (2 or 3 neighbors)
  {
    GameOfLife game(3, 3);
    game.SetCellState(0, 0, true);
    game.SetCellState(0, 1, true);
    game.SetCellState(1, 1, true);
    game.NextGeneration();
    EXPECT_TRUE(game.GetCellState(1, 1));  // Should survive with 2 neighbors
  }

  // Test overpopulation (more than 3 neighbors)
  {
    GameOfLife game(3, 3);
    game.SetCellState(0, 0, true);
    game.SetCellState(0, 1, true);
    game.SetCellState(0, 2, true);
    game.SetCellState(1, 0, true);
    game.SetCellState(1, 1, true);
    game.NextGeneration();
    EXPECT_FALSE(game.GetCellState(1, 1));  // Should die from overpopulation
  }

  // Test reproduction (exactly 3 neighbors)
  {
    GameOfLife game(3, 3);
    game.SetCellState(0, 0, true);
    game.SetCellState(0, 1, true);
    game.SetCellState(0, 2, true);
    game.NextGeneration();
    EXPECT_TRUE(
        game.GetCellState(1, 1));  // Should become alive with 3 neighbors
  }
}

// Test specific patterns
TEST(GameOfLifeTest, Patterns) {
  // Test still life: Block
  {
    GameOfLife game(4, 4);
    // Create a block pattern
    //  . . . .
    //  . O O .
    //  . O O .
    //  . . . .
    game.SetCellState(1, 1, true);
    game.SetCellState(1, 2, true);
    game.SetCellState(2, 1, true);
    game.SetCellState(2, 2, true);

    game.NextGeneration();

    // Block should remain stable
    EXPECT_TRUE(game.GetCellState(1, 1));
    EXPECT_TRUE(game.GetCellState(1, 2));
    EXPECT_TRUE(game.GetCellState(2, 1));
    EXPECT_TRUE(game.GetCellState(2, 2));
  }

  // Test oscillator: Blinker
  {
    GameOfLife game(5, 5);
    // Create a blinker pattern (vertical)
    //  . . . . .
    //  . . O . .
    //  . . O . .
    //  . . O . .
    //  . . . . .
    game.SetCellState(2, 1, true);
    game.SetCellState(2, 2, true);
    game.SetCellState(2, 3, true);

    game.NextGeneration();

    // Blinker should become horizontal
    //  . . . . .
    //  . . . . .
    //  . O O O .
    //  . . . . .
    //  . . . . .
    EXPECT_FALSE(game.GetCellState(2, 1));
    EXPECT_TRUE(game.GetCellState(1, 2));
    EXPECT_TRUE(game.GetCellState(2, 2));
    EXPECT_TRUE(game.GetCellState(3, 2));
    EXPECT_FALSE(game.GetCellState(2, 3));

    game.NextGeneration();

    // Blinker should return to vertical
    EXPECT_TRUE(game.GetCellState(2, 1));
    EXPECT_TRUE(game.GetCellState(2, 2));
    EXPECT_TRUE(game.GetCellState(2, 3));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}