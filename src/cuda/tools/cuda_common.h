#pragma once

#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(expression)                                    \
  {                                                                     \
    cudaError_t error = expression;                                     \
    if (error != cudaSuccess) {                                         \
      std::cout << "CUDA error( " << error << " ) on line " << __LINE__ \
                << " in the file: " << __FILE__ << " -> "               \
                << cudaGetErrorName(error) << std::endl;                \
      std::exit(-1);                                                    \
    }                                                                   \
  }
  