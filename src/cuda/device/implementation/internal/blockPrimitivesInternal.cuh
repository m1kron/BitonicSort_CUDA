#pragma once

#include "../warpPrimitives.cuh"

// NOTE: all following functions assume that will operate at block and all
// threads will coperatively call given function.

namespace block {
namespace internal {
// Block-wide memcpy for int4* buffers.
__device__ void MemcpyData_sync(const int4* __restrict__ srcMemPtr,
                                int4* __restrict__ dstMemPtr,
                                const uint32_t memSize);

// Creates bitonic sequence of WARP_BUCKET_SIZE out of random data at sharedMem.
template <int WARP_BUCKET_SIZE>
__device__ void CreateBitonicSequence_WarpBucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize);

// Creates bitonic sequence of sharedMemSize out of random data at sharedMem -
// which effectivelly means that all data at shared memory will be sorted.
template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void CreateBitonicSequence_SharedMemBucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize, const bool increasing);

// Performes bitonic's stage for WARP_BUCKET_SIZE bucket size sequence out of
// data at sharedMem.
template <int WARP_BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformBitonicStage_WarpBucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize);

// Performes bitonic's stage for given INITIAL_BUCKET_SIZE. All steps buckets
// will be forced to produce increasing or decresing sepecified by
// PRODUCE_INCREASING flag.
template <bool PRODUCE_INCREASING, int INITIAL_BUCKET_SIZE,
          int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageFixedOrdering_BucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize);

// Performes bitonic's stage for given INITIAL_BUCKET_SIZE. Given step bucket is
// dynamically set depending on bucket index.
template <int INITIAL_BUCKET_SIZE, int COMPARE_SWAP_WARP_BUCKET_SIZE,
          int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageDynamicOrdering_BucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize);

////////////////////////////////////////////////////////////////////////////
//
// INLINES:
//
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void MemcpyData_sync(
    const int4* __restrict__ srcMemPtr, int4* __restrict__ dstMemPtr,
    const uint32_t memSize) {
  for (int allThreadsStride = 0; allThreadsStride < memSize;
       allThreadsStride += blockDim.x)
    dstMemPtr[allThreadsStride + threadIdx.x] =
        srcMemPtr[allThreadsStride + threadIdx.x];

  __syncthreads();
}

////////////////////////////////////////////////////////////////////////////
template <int WARP_BUCKET_SIZE>
__device__ void CreateBitonicSequence_WarpBucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize) {
  assert(__isShared(sharedMem) == 1);
  for (int thisWarpSharedMemOffset = tools::GetWarpIdx() * WARP_BUCKET_SIZE;
       thisWarpSharedMemOffset < sharedMemSize;
       thisWarpSharedMemOffset +=
       tools::GetNumberOfWarpsInBlock() * WARP_BUCKET_SIZE) {
    const int bucketIdx = thisWarpSharedMemOffset / WARP_BUCKET_SIZE;
    const bool increasing = (bucketIdx & 1) == 0;
    // All threads in warp execute the same branch.
    assert((__ballot_sync(tools::ALL_THREADS_MASK, increasing) == 0) ||
           (__ballot_sync(tools::ALL_THREADS_MASK, increasing) ==
            tools::ALL_THREADS_MASK));
    if (increasing)
      warp::PerformBitonicSort_BucketSizeWide_sync<WARP_BUCKET_SIZE, true>(
          sharedMem + thisWarpSharedMemOffset);
    else
      warp::PerformBitonicSort_BucketSizeWide_sync<WARP_BUCKET_SIZE, false>(
          sharedMem + thisWarpSharedMemOffset);
  }
  __syncthreads();
}

////////////////////////////////////////////////////////////////////////////
template <int WARP_BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformBitonicStage_WarpBucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize) {
  assert(__isShared(sharedMem) == 1);
  for (int thisWarpSharedMemOffset = tools::GetWarpIdx() * WARP_BUCKET_SIZE;
       thisWarpSharedMemOffset < sharedMemSize;
       thisWarpSharedMemOffset +=
       tools::GetNumberOfWarpsInBlock() * WARP_BUCKET_SIZE) {
    warp::PerformStage_BucketSizeWide_sync<WARP_BUCKET_SIZE,
                                           PRODUCE_INCREASING>(
        sharedMem + thisWarpSharedMemOffset);
  }
  __syncthreads();
}

////////////////////////////////////////////////////////////////////////////
template <bool PRODUCE_INCREASING, int INITIAL_BUCKET_SIZE,
          int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageFixedOrdering_BucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize) {
#pragma unroll
  for (int currentBucketSize = INITIAL_BUCKET_SIZE;
       currentBucketSize > WARP_BUCKET_SIZE; currentBucketSize /= 2) {
    const int bucketOffset = currentBucketSize / 2;
    const int warpsPerBucket = bucketOffset / COMPARE_SWAP_WARP_BUCKET_SIZE;
    const int thisWarpBucketIdx = tools::GetWarpIdx() / warpsPerBucket;

    int thisWarpSharedMemOffset =
        thisWarpBucketIdx * currentBucketSize +
        (tools::GetWarpIdx() % warpsPerBucket) * COMPARE_SWAP_WARP_BUCKET_SIZE;
    const int allWarpsStride =
        tools::GetNumberOfWarpsInBlock() / warpsPerBucket * currentBucketSize;

#pragma unroll
    for (; thisWarpSharedMemOffset < sharedMemSize;
         thisWarpSharedMemOffset += allWarpsStride)
      warp::CompareSwap_sync<COMPARE_SWAP_WARP_BUCKET_SIZE, PRODUCE_INCREASING>(
          sharedMem + thisWarpSharedMemOffset, bucketOffset);

    __syncthreads();
  }

  block::internal::PerformBitonicStage_WarpBucketSize_sync<WARP_BUCKET_SIZE,
                                                           PRODUCE_INCREASING>(
      sharedMem, sharedMemSize);
}

////////////////////////////////////////////////////////////////////////////
template <int BUCKET_SIZE, int COMPARE_SWAP_WARP_BUCKET_SIZE,
          int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageDynamicOrdering_BucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize) {
  const int initialBucketOffset = BUCKET_SIZE / 2;
  const int initialWarpsPerBucket =
      initialBucketOffset / COMPARE_SWAP_WARP_BUCKET_SIZE;
  const int initialThisWarpBucketIdx =
      tools::GetWarpIdx() / initialWarpsPerBucket;
  const bool increasing = (initialThisWarpBucketIdx & 1) == 0;

  // All threads in warp execute the same branch.
  assert((__ballot_sync(tools::ALL_THREADS_MASK, increasing) == 0) ||
         (__ballot_sync(tools::ALL_THREADS_MASK, increasing) ==
          tools::ALL_THREADS_MASK));

  if (increasing)
    block::internal::PerformBitonicStageFixedOrdering_BucketSize_sync<
        true, BUCKET_SIZE, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
        sharedMem, sharedMemSize);
  else
    block::internal::PerformBitonicStageFixedOrdering_BucketSize_sync<
        false, BUCKET_SIZE, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
        sharedMem, sharedMemSize);
}

////////////////////////////////////////////////////////////////////////////
template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void CreateBitonicSequence_SharedMemBucketSize_sync(
    int* sharedMem, const uint32_t sharedMemSize, const bool increasing) {
  block::internal::CreateBitonicSequence_WarpBucketSize_sync<WARP_BUCKET_SIZE>(
      sharedMem, sharedMemSize);

  block::internal::PerformBitonicStageDynamicOrdering_BucketSize_sync<
      512, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(sharedMem,
                                                            sharedMemSize);
  block::internal::PerformBitonicStageDynamicOrdering_BucketSize_sync<
      1024, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(sharedMem,
                                                             sharedMemSize);
  block::internal::PerformBitonicStageDynamicOrdering_BucketSize_sync<
      2048, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(sharedMem,
                                                             sharedMemSize);
  block::internal::PerformBitonicStageDynamicOrdering_BucketSize_sync<
      4096, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(sharedMem,
                                                             sharedMemSize);

  if (increasing)
    block::internal::PerformBitonicStageFixedOrdering_BucketSize_sync<
        true, 8192, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
        sharedMem, sharedMemSize);
  else
    block::internal::PerformBitonicStageFixedOrdering_BucketSize_sync<
        false, 8192, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
        sharedMem, sharedMemSize);
}

}  // namespace internal
}  // namespace block