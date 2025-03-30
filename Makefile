CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O3
LDFLAGS = 

# Google Test flags
GTEST_CFLAGS = -I/usr/src/googletest/googletest/include
GTEST_LIBS = -lgtest -lgtest_main -pthread

SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

# Source files
SRC_FILES = $(SRC_DIR)/game_of_life.cpp
MAIN_SRC = $(SRC_DIR)/main.cpp
TEST_SRC = $(TEST_DIR)/game_of_life_test.cpp

# Object files
MAIN_OBJ = $(BUILD_DIR)/main.o
GAME_OBJ = $(BUILD_DIR)/game_of_life.o
TEST_OBJ = $(BUILD_DIR)/game_of_life_test.o

# Executables
MAIN_TARGET = game_of_life
TEST_TARGET = game_of_life_test

.PHONY: all clean test

all: $(MAIN_TARGET) $(TEST_TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile game of life library
$(GAME_OBJ): $(SRC_DIR)/game_of_life.cpp $(SRC_DIR)/game_of_life.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile main program
$(MAIN_OBJ): $(MAIN_SRC) $(SRC_DIR)/game_of_life.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test program
$(TEST_OBJ): $(TEST_SRC) $(SRC_DIR)/game_of_life.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(GTEST_CFLAGS) -c $< -o $@

# Link main program
$(MAIN_TARGET): $(MAIN_OBJ) $(GAME_OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@

# Link test program
$(TEST_TARGET): $(TEST_OBJ) $(GAME_OBJ)
	$(CXX) $(LDFLAGS) $^ $(GTEST_LIBS) -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Clean up
clean:
	rm -rf $(BUILD_DIR) $(MAIN_TARGET) $(TEST_TARGET) 