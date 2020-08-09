#pragma once

#include <assert.h>
#include "internal/warpPrimitivesInternal.cuh"

// NOTE: all following functions assume that will operate on warp that calling thread belongs to.
// NOTE2: It is assumed that all thread's in given warp will be active when calling any of the
// following functions.

namespace warp 
{
// Performs bitonic sort over sequence of BUCKET_SIZE items that is stored at warpSharedMemPtr.
// Result os sorted incresing/decreasing sequence of BUCKET_SIZE.
// warpSharedMemPtr has to point to shared memory.
template< int BUCKET_SIZE, bool PRODUCE_INCREASING >
__device__ void PerformBitonicSort_BucketSizeWide_sync(int* warpSharedMemPtr);

// Performs bitonic sort's stage over sequence of BUCKET_SIZE items that is stored at warpSharedMemPtr.
// Result is sorted incresing/decreasing sequence of BUCKET_SIZE.
// warpSharedMemPtr has to point to shared memory.
template< int BUCKET_SIZE, bool PRODUCE_INCREASING >
__device__ void PerformStage_BucketSizeWide_sync(int* warpSharedMemPtr);

// Compares and swaps at warp level. WARP_BUCKET_SIZE has to be multiple of 
// tools::WARP_SIZE and specifies the amoout of data data warp will compare.
// WARNING! sharedMem has size to to be >= WARP_BUCKET_SIZE + bucketOffset.
template<int WARP_BUCKET_SIZE,bool PRODUCE_INCREASING>
__device__ void CompareSwap_sync(int* sharedMem, const int bucketOffset);

///////////////////////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
template< int BUCKET_SIZE, bool PRODUCE_INCREASING >
__device__ void PerformBitonicSort_BucketSizeWide_sync(int* warpSharedMemPtr) 
{
  static_assert(BUCKET_SIZE == 256, "Current implementation supports only 256 bucket size!");
  assert(warpSharedMemPtr != nullptr);
  assert(__isShared(warpSharedMemPtr) == 1);

  const int itemsPerThread = BUCKET_SIZE/tools::WARP_SIZE;
  int storage[itemsPerThread];

  // Load data into registers.
  #pragma unroll
  for( int i = 0; i < itemsPerThread; ++i)
    storage[i] = warpSharedMemPtr[tools::GetThreadIdxWithinWarp() + i*tools::WARP_SIZE];

  // Stages 1 - 5.
  #pragma unroll
  for( int i = 0; i < itemsPerThread; i+=2) 
  {
    storage[i] = warp::internal::PerformBitonicSort_32ElementsWide_sync<true>(storage[i]);
    storage[i+1] = warp::internal::PerformBitonicSort_32ElementsWide_sync<false>(storage[i+1]);
  }

  // Stage 6.
  #pragma unroll
  for( int i = 0; i < itemsPerThread; i+=4) 
  {
    warp::internal::PerformStageForGivenBucketSize_sync<64, true >(storage+i);
    warp::internal::PerformStageForGivenBucketSize_sync<64, false >(storage+i+2);
  }

  // Stage 7.
  warp::internal::PerformStageForGivenBucketSize_sync<128, true >(storage);
  warp::internal::PerformStageForGivenBucketSize_sync<128, false >(storage+4);
  
  // Stage 8.
  warp::internal::PerformStageForGivenBucketSize_sync<256, PRODUCE_INCREASING >(storage);
  
  // Store date to sharedMem.
  #pragma unroll
  for( int i = 0; i < itemsPerThread; ++i)
    warpSharedMemPtr[tools::GetThreadIdxWithinWarp() + i*tools::WARP_SIZE] = storage[i];

  __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////////
template< int BUCKET_SIZE, bool PRODUCE_INCREASING >
__device__ void PerformStage_BucketSizeWide_sync(int* warpSharedMemPtr) 
{
  assert(warpSharedMemPtr != nullptr);
  assert(__isShared(warpSharedMemPtr) == 1);
  const int itemsPerThread = BUCKET_SIZE/tools::WARP_SIZE;
  int storage[itemsPerThread];

  // Load data into registers.
  #pragma unroll
  for( int i = 0; i < itemsPerThread; ++i)
    storage[i] = warpSharedMemPtr[tools::GetThreadIdxWithinWarp() + i*tools::WARP_SIZE];

  // Sort in registers.
  warp::internal::PerformStageForGivenBucketSize_sync<BUCKET_SIZE, PRODUCE_INCREASING >(storage);
  
  // Store sorted sequence to sharedMem.
  #pragma unroll
  for( int i = 0; i < itemsPerThread; ++i)
    warpSharedMemPtr[tools::GetThreadIdxWithinWarp() + i*tools::WARP_SIZE] = storage[i];

  __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////////
template<int WARP_BUCKET_SIZE,bool PRODUCE_INCREASING>
__device__ void CompareSwap_sync(int* sharedMem, const int bucketOffset)
{
  const int ITEMS_PER_THREAD = WARP_BUCKET_SIZE/tools::WARP_SIZE;
  #pragma unroll
  for( int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    const int offset1 = i*tools::WARP_SIZE + tools::GetThreadIdxWithinWarp();
    const int offset2 = bucketOffset + i*tools::WARP_SIZE + tools::GetThreadIdxWithinWarp();
    const int val1 = sharedMem[offset1];
    const int val2 = sharedMem[offset2];

    const bool swapNeeded = helpers::NeedsToBeSwapped<int, PRODUCE_INCREASING>(val1,val2);
    if( swapNeeded )
    {
      sharedMem[offset1] = val2;
      sharedMem[offset2] = val1;
    }
  }
}
}