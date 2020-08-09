#pragma once

#include "../helpers.cuh"
#include "../../tools/warpTools.cuh"

// NOTE: all following functions assume that will operate on warp that calling thread belongs to.
// NOTE2: It is assumed that all thread's in given warp will be active when calling any of the
// following functions.

namespace warp
{

namespace internal {
// Performs single bitonic step over val for given offset at warp level.
// Thread mask encodes inforamtion wether given thread should take min or max from values.
template <int THREAD_MASK, int STEP_OFFSET >
__device__ int PerformStep_WarpWide_sync(const int val);

// Performs last five steps of bitonic sort algo over val 
// variable in warp. It is assumed that the function operates on
// 32 4-byte elements bitonic sequence of bucket size 16 represented by val
// in each thread.
// The output is sorted, increasing/decresing, 32 4-byte sequence which is kept
// in val by each thread in warp. 
template < bool SORT_INCREASING >
__device__ int PerformStage_WarpWide_sync(int val);

// Performs bitonic sort over warp's val variable.
// This creates increasing/decreasing sequence over val variable 
// in whole warp. 
// Input is assumed to be unsorted sequence.
// This function limits ilp as it has long dep chain.
template < bool SORT_INCREASING >
__device__ int PerformBitonicSort_32ElementsWide_sync(int val);


// Performs bitonic sort stage for given BUCKET_SIZE.
// It is assumed that threadRegisterBuffer contains every 32's element of given bucket.
template< int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformStageForGivenBucketSize_sync(int* threadRegisterBuffer);

////////////////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
template <int THREAD_MASK, int STEP_OFFSET >
__device__ int PerformStep_32ElementsWide_sync(const int val)
{
  const int otherVal = __shfl_xor_sync(tools::ALL_THREADS_MASK, val, STEP_OFFSET);
  const int thisThreadMaskInWarp = 1 << tools::GetThreadIdxWithinWarp();
  const bool shouldTakeMax = (thisThreadMaskInWarp & THREAD_MASK);
  const int output = helpers::SelectMinMax(val, otherVal, shouldTakeMax);

  return output;
}

////////////////////////////////////////////////////////////////////////////
template <bool SORT_INCREASING>
__device__ int PerformStage_WarpWide_sync(int val)
{
  if ( SORT_INCREASING )
  {
    val = warp::internal::PerformStep_32ElementsWide_sync< 0xFFFF0000, 16 >(val);
    val = warp::internal::PerformStep_32ElementsWide_sync< 0xFF00FF00, 8 >(val);
    val = warp::internal::PerformStep_32ElementsWide_sync< 0xF0F0F0F0, 4 >(val);
    val = warp::internal::PerformStep_32ElementsWide_sync< 0xCCCCCCCC, 2 >(val);
    val = warp::internal::PerformStep_32ElementsWide_sync< 0xAAAAAAAA, 1 >(val);
  }
  else // sort decreasing
  {
    val = warp::internal::PerformStep_32ElementsWide_sync< 0x0000FFFF, 16 >(val);
    val = warp::internal::PerformStep_32ElementsWide_sync< 0x00FF00FF, 8 >(val);
    val = warp::internal::PerformStep_32ElementsWide_sync< 0x0F0F0F0F, 4 >(val);
    val = warp::internal::PerformStep_32ElementsWide_sync< 0x33333333, 2 >(val);
    val = warp::internal::PerformStep_32ElementsWide_sync< 0x55555555, 1 >(val);
  }
  return val;
}

////////////////////////////////////////////////////////////////////////////
template < bool SORT_INCREASING >
__device__ int PerformBitonicSort_32ElementsWide_sync(int val)
{
  // Stage 1
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x66666666, 1 >(val);

  // Stage 2
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x3C3C3C3C, 2 >(val);
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x5A5A5A5A, 1 >(val);

  // Stage 3
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x0FF00FF0, 4 >(val);
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x33CC33CC, 2 >(val);
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x55AA55AA, 1 >(val);

  // Stage 4
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x00FFFF00, 8 >(val);
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x0F0FF0F0, 4 >(val);
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x3333CCCC, 2 >(val);
  val = warp::internal::PerformStep_32ElementsWide_sync< 0x5555AAAA, 1 >(val);

  // Stage 5
  return warp::internal::PerformStage_WarpWide_sync<SORT_INCREASING>(val);
}

////////////////////////////////////////////////////////////////////////////
template< int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformStageForGivenBucketSize_sync(int* threadRegisterBuffer)
{
  const int REGISTER_BUFFER_SIZE = BUCKET_SIZE/tools::WARP_SIZE;
  #pragma unroll
  for( int offset = BUCKET_SIZE/2; offset >= tools::WARP_SIZE; offset /= 2)
  {
    const int storageOffset = offset/tools::WARP_SIZE;

    #pragma unroll
    for( int storageIdx = 0; storageIdx < REGISTER_BUFFER_SIZE; ++storageIdx)
    {
      const int compareToIdx = storageIdx ^ storageOffset;

      // NOTE: do nothing if compareToIdx < storageIdx. This means that the loop does something 
      // only for REGISTER_BUFFER_SIZE/2 items!
      if( compareToIdx < storageIdx ) 
        continue;

      if(helpers::NeedsToBeSwapped<int, PRODUCE_INCREASING>(threadRegisterBuffer[storageIdx],threadRegisterBuffer[compareToIdx]))
        helpers::Swap(threadRegisterBuffer[storageIdx], threadRegisterBuffer[compareToIdx]);
    }
  }

  #pragma unroll
  for( int i = 0; i < REGISTER_BUFFER_SIZE; ++i){
    threadRegisterBuffer[i] = warp::internal::PerformStage_WarpWide_sync<PRODUCE_INCREASING> (threadRegisterBuffer[i]);
  }
}

}
}