#pragma once
#include <stdint.h>

// Performance inplace soring of given data.
// WARNING! Currently dataSize has to be power of 2.
// Returns true if kernel was sucessfully launched, 
// false - when problems has been encountered.
bool BitonicSort(int32_t* in_out_gpuDataPtr, uint32_t dataSize);