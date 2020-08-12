#pragma once
#include <stdint.h>

#include <random>
#include <vector>

namespace tools {

// Generates a vector if given size, filled with random int32_t.
std::vector<int32_t> GenerateRandomInt(size_t size);

////////////////////////////////////////////////////////////////////
//
// INLINES:
//
////////////////////////////////////////////////////////////////////

inline std::vector<int32_t> GenerateRandomInt(size_t size) {
  // [NOTE]: I am using inverse transform sampling method here to generate
  // logistic distribution from uniform distribution.
  std::mt19937 mt(1);
  std::uniform_int_distribution<int32_t> dist(-1000, 1000);

  std::vector<int32_t> vec;
  vec.resize(size);

  for (auto &val : vec) {
    val = dist(mt);
  }

  return vec;
}

}  // namespace tools
