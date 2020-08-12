#pragma once

//#define DEBUG_MODE

#ifndef DEBUG_MODE
#define DEBUG_PRINT(...)
#else
#include <stdio.h>
#define DEBUG_PRINT(maxBlockIdx, maxThreadIdx, format, ...)         \
  if (blockIdx.x <= maxBlockIdx && threadIdx.x <= maxThreadIdx) {   \
    printf("[BLK: %i, THRD: %i]: " format, blockIdx.x, threadIdx.x, \
           __VA_ARGS__);                                            \
  }
#endif