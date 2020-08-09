#pragma once

#include "internal/blockPrimitivesInternal.cuh"

// NOTE: all following functions assume that will operate at block and all
// threads will coperatively call given function.

namespace block {

// Sorts random sequence at globalDataPtrForThisBlock of size sharedMemSize.
template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void SortBitionicInSharedMemory(
    int4 *__restrict__ globalDataPtrForThisBlock, int4 *__restrict__ sharedMem,
    const uint32_t sharedMemSize, const bool increasing);

// Performs bitonic stage on shared mem bucket size in shared memory.
template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageInSharedMemory(
    int4 *__restrict__ globalDataPtrForThisBlock, int4 *__restrict__ sharedMem,
    const uint32_t sharedMemSize, const bool increasing);

////////////////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void SortBitionicInSharedMemory(
    int4 *__restrict__ globalDataPtrForThisBlock, int4 *__restrict__ sharedMem,
    const uint32_t sharedMemSize, const bool increasing) {
  block::internal::MemcpyData_sync(globalDataPtrForThisBlock, sharedMem,
                                   sharedMemSize);
  block::internal::CreateBitonicSequence_SharedMemBucketSize_sync<
      COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
      reinterpret_cast<int *>(sharedMem), sharedMemSize * 4, increasing);
  block::internal::MemcpyData_sync(sharedMem, globalDataPtrForThisBlock,
                                   sharedMemSize);
}

////////////////////////////////////////////////////////////////////////////
template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageInSharedMemory(
    int4 *__restrict__ globalDataPtrForThisBlock, int4 *__restrict__ sharedMem,
    const uint32_t sharedMemSize, const bool increasing) {
  block::internal::MemcpyData_sync(globalDataPtrForThisBlock, sharedMem,
                                   sharedMemSize);
  if (increasing)
    block::internal::PerformBitonicStageFixedOrdering_BucketSize_sync<
        true, 8192, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
        reinterpret_cast<int *>(sharedMem), sharedMemSize * 4);
  else
    block::internal::PerformBitonicStageFixedOrdering_BucketSize_sync<
        false, 8192, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
        reinterpret_cast<int *>(sharedMem), sharedMemSize * 4);
  block::internal::MemcpyData_sync(sharedMem, globalDataPtrForThisBlock,
                                   sharedMemSize);
}

}  // namespace block