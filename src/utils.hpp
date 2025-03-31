#pragma once

#include <cstdlib>

/**
 * Clears the console screen in a cross-platform way
 * @return The return value from the system call
 */
inline int ClearScreen() {
#ifdef _WIN32
  return std::system("cls");
#else
  return std::system("clear");
#endif
}