#include "bitonicSort.h"

#include "device/debug/debugPrint.cuh"
#include "device/implementation/blockPrimitives.cuh"

const int SHARED_SIZE_INT = 8192;
const int THREADS_PER_BLOCK = 1024;

// Launch params for all kernels.
template <int SHARED_SIZE_INT_PARAM, int THREADS_PER_BLOCK_PARAM>
struct KernelParams {
 public:
  enum {
    SHARED_SIZE_INT = SHARED_SIZE_INT_PARAM,
    THREADS_PER_BLOCK = THREADS_PER_BLOCK_PARAM,
    SHARED_SIZE_INT4 = SHARED_SIZE_INT / 4,
    WARPS_PER_BLOCK = THREADS_PER_BLOCK / tools::WARP_SIZE,
    WARP_BUCKET_SIZE = SHARED_SIZE_INT / WARPS_PER_BLOCK,
    COMPARE_SWAP_WARP_BUCKET_SIZE = WARP_BUCKET_SIZE / 2
  };
};

////////////////////////////////////////////////////////////////////////////////////
template <typename KERNEL_PARAMS>
__global__ void SortBitionicInSharedMemoryKernel(int4* data,
                                                 uint32_t dataSize) {
  __shared__ int4 sharedMem[KERNEL_PARAMS::SHARED_SIZE_INT4];

  for (uint32_t thisBlockOffset = blockIdx.x * KERNEL_PARAMS::SHARED_SIZE_INT4;
       thisBlockOffset < dataSize;
       thisBlockOffset += gridDim.x * KERNEL_PARAMS::SHARED_SIZE_INT4) {
    const int thisBlockBucketIdx =
        thisBlockOffset / KERNEL_PARAMS::SHARED_SIZE_INT4;
    const bool increasing = (thisBlockBucketIdx & 1) == 0;

    block::SortBitionicInSharedMemory<
        KERNEL_PARAMS::COMPARE_SWAP_WARP_BUCKET_SIZE,
        KERNEL_PARAMS::WARP_BUCKET_SIZE>(data + thisBlockOffset, sharedMem,
                                         KERNEL_PARAMS::SHARED_SIZE_INT4,
                                         increasing);
  }
}

////////////////////////////////////////////////////////////////////////////////////
template <typename KERNEL_PARAMS>
__global__ void PerformBitonicLastStagesInSharedMemoryKernel(int4* data,
                                                             uint32_t dataSize,
                                                             int originStage) {
  __shared__ int4 sharedMem[KERNEL_PARAMS::SHARED_SIZE_INT4];

  const int COMPARE_SWAP_BLOCK_BUCKET_SIZE =
      KERNEL_PARAMS::SHARED_SIZE_INT4 / 2;
  const int stageOffset = originStage / 2;
  const int stageBlocksPerBucket = stageOffset / COMPARE_SWAP_BLOCK_BUCKET_SIZE;
  const int stageThisBlockBucketIdx = blockIdx.x / stageBlocksPerBucket;
  const bool increasing = (stageThisBlockBucketIdx & 1) == 0;

  for (uint32_t thisBlockOffset = blockIdx.x * KERNEL_PARAMS::SHARED_SIZE_INT4;
       thisBlockOffset < dataSize;
       thisBlockOffset += gridDim.x * KERNEL_PARAMS::SHARED_SIZE_INT4)
    block::PerformBitonicStageInSharedMemory<
        KERNEL_PARAMS::COMPARE_SWAP_WARP_BUCKET_SIZE,
        KERNEL_PARAMS::WARP_BUCKET_SIZE>(data + thisBlockOffset, sharedMem,
                                         KERNEL_PARAMS::SHARED_SIZE_INT4,
                                         increasing);
}

////////////////////////////////////////////////////////////////////////////////////
template <typename KERNEL_PARAMS>
__global__ void PerformBitonicStageKernel(int4* data, uint32_t dataSize,
                                          int step, int stage) {
  const int COMPARE_SWAP_BLOCK_BUCKET_SIZE =
      KERNEL_PARAMS::SHARED_SIZE_INT4 / 2;

  const int stageOffset = stage / 2;
  const int stageBlocksPerBucket = stageOffset / COMPARE_SWAP_BLOCK_BUCKET_SIZE;
  const int stageThisBlockBucketIdx = blockIdx.x / stageBlocksPerBucket;
  const bool increasing = (stageThisBlockBucketIdx & 1) == 0;

  const int stride = step / 2;
  const int blocksPerBucket = stride / COMPARE_SWAP_BLOCK_BUCKET_SIZE;
  const int thisBlockBucketIdx = blockIdx.x / blocksPerBucket;

  int thisBlockGlobalMemOffset =
      thisBlockBucketIdx * step +
      (blockIdx.x % blocksPerBucket) * COMPARE_SWAP_BLOCK_BUCKET_SIZE;
  const int currentBlocksStride = blockDim.x / blocksPerBucket * step;
  for (; thisBlockGlobalMemOffset < dataSize;
       thisBlockGlobalMemOffset += currentBlocksStride) {
    const int thisThreadOffset = thisBlockGlobalMemOffset + threadIdx.x;
    int4 val1 = data[thisThreadOffset];
    int4 val2 = data[thisThreadOffset + stride];

    bool swapDone = false;

    swapDone |= helpers::PerformSwapIfNeeded(val1.x, val2.x, increasing);
    swapDone |= helpers::PerformSwapIfNeeded(val1.y, val2.y, increasing);
    swapDone |= helpers::PerformSwapIfNeeded(val1.z, val2.z, increasing);
    swapDone |= helpers::PerformSwapIfNeeded(val1.w, val2.w, increasing);

    if (swapDone) {
      data[thisThreadOffset] = val1;
      data[thisThreadOffset + stride] = val2;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
bool BitonicSort(int32_t* in_out_gpuDataPtr, uint32_t dataSize) {
  if ((dataSize & (dataSize - 1)) != 0) return false;  // Not power of two.

  using PARAMS = KernelParams<SHARED_SIZE_INT, THREADS_PER_BLOCK>;
  int4* in_out_gpuDataPtr_int4 = reinterpret_cast<int4*>(in_out_gpuDataPtr);
  const uint32_t dataSize_int4 = dataSize / 4;
  const uint32_t numOfBlocks = dataSize_int4 / PARAMS::THREADS_PER_BLOCK;

  // Sorts chunks of shared mem size.
  SortBitionicInSharedMemoryKernel<PARAMS><<<numOfBlocks, THREADS_PER_BLOCK>>>(
      in_out_gpuDataPtr_int4, dataSize_int4);

  // Performs gloal bitonic steps strating from chunks of shared mem size.
  for (int stage = PARAMS::SHARED_SIZE_INT4 * 2; stage <= dataSize_int4;
       stage *= 2) {
    for (int step = stage; step > PARAMS::SHARED_SIZE_INT4; step /= 2)
      PerformBitonicStageKernel<PARAMS>
          <<<numOfBlocks, PARAMS::THREADS_PER_BLOCK>>>(
              in_out_gpuDataPtr_int4, dataSize_int4, step, stage);
    PerformBitonicLastStagesInSharedMemoryKernel<PARAMS>
        <<<numOfBlocks, PARAMS::THREADS_PER_BLOCK>>>(in_out_gpuDataPtr_int4,
                                                     dataSize_int4, stage);
  }

  return true;
}
