/// Storage for unsed code - which might be helpful in the future.


//////////////////////////////////////////////////////////////////////
__device__  int SortStage_warp_nosync(const int initalVal, const int dataIdx, const int stageOffset, const int bucketSize)
{
  const int otherVal = __shfl_xor_sync(ALL_THREAD_MASK, initalVal, stageOffset);

  // This can be replaced by constant mask in compile time!
  const int bucketIdx = dataIdx / bucketSize;
  const int takeMax = bucketIdx & 1;
  const int secondThread = ((dataIdx - bucketIdx*stageOffset*2)/stageOffset) & 1;
  const bool shouldTakeMax = (takeMax^secondThread) == 1;
  // --
  const int output = SelectMinMax(initalVal, otherVal, shouldTakeMax);

  DEBUG_PRINT(0, 8, "shouldTakeMax: %i, val1 = %i, val2 = %i, output = %i\n", shouldTakeMax, initalVal, otherVal, output );

  return output;
}

//////////////////////////////////////////////////////////////////////
__device__ void Sort_warp_sync(int* data, const uint32_t startOffset, const uint32_t sharedMemSize )
{
  const auto dataIdx = threadIdx.x + startOffset;

  int val = data[dataIdx];
  __syncwarp();

  val = SortStage_warp_nosync(val, dataIdx, 1, 2);

  val = SortStage_warp_nosync(val, dataIdx, 2, 4);
  val = SortStage_warp_nosync(val, dataIdx, 1, 4);

  val = SortStage_warp_nosync(val, dataIdx, 4, 8);
  val = SortStage_warp_nosync(val, dataIdx, 2, 8);
  val = SortStage_warp_nosync(val, dataIdx, 1, 8);

  val = SortStage_warp_nosync(val, dataIdx, 8, 16);
  val = SortStage_warp_nosync(val, dataIdx, 4, 16);
  val = SortStage_warp_nosync(val, dataIdx, 2, 16);
  val = SortStage_warp_nosync(val, dataIdx, 1, 16);

  val = SortStage_warp_nosync(val, dataIdx, 16, 32);
  val = SortStage_warp_nosync(val, dataIdx, 8, 32);
  val = SortStage_warp_nosync(val, dataIdx, 4, 32);
  val = SortStage_warp_nosync(val, dataIdx, 2, 32);
  val = SortStage_warp_nosync(val, dataIdx, 1, 32);

  data[dataIdx] = val;
  __syncwarp();
}


//////////////////////////////////////////////////////////////////////
__device__ void SortSequence64BucketSize_block_sync(int* sharedMem, const uint32_t sharedMemSize, const bool increasing) 
{
  const int bucketSize=64;
  int thisThreadSharedMemOffset = tools::GetWarpIdx()*bucketSize+tools::GetThreadIdxWithinWarp();

  for( ; thisThreadSharedMemOffset < sharedMemSize; thisThreadSharedMemOffset += tools::GetNumberOfWarpsInBlock()*bucketSize )
  {
    const int threadOffset1 = thisThreadSharedMemOffset;
    int val1 = sharedMem[threadOffset1];
    const int threadOffset2 = threadOffset1 + tools::WARP_SIZE;
    int val2 = sharedMem[threadOffset2];

    int max = SelectMinMax<true>(val1, val2);
    int min = SelectMinMax<false>(val1, val2);

    if( increasing ) {
      min = SortInRegister_warp_sync<true>(min);
      max = SortInRegister_warp_sync<true>(max);

      sharedMem[threadOffset1] = min;
      sharedMem[threadOffset2] = max;
    } else {
      min = SortInRegister_warp_sync<false>(min);
      max = SortInRegister_warp_sync<false>(max);

      sharedMem[threadOffset1] = max;
      sharedMem[threadOffset2] = min;
    }
  }

  __syncthreads();
}


//////////////////////////////////////////////////////////////////////
__device__ void SortSequence_block_sync(int* sharedMem, const uint32_t sharedMemSize, const int bucketSize ) 
{
  const int initialBucketOffset = bucketSize/2;
  const int initialWarpsPerBucket = initialBucketOffset/tools::WARP_SIZE;
  const int initialThisWarpBucketIdx = tools::GetWarpIdx()/initialWarpsPerBucket;
  const bool increasing = (initialThisWarpBucketIdx & 1) == 0;

  for( int currentBucketSize = bucketSize; currentBucketSize > 64; currentBucketSize/=2)
  {
    const int bucketOffset = currentBucketSize/2;
    const int warpsPerBucket = bucketOffset/tools::WARP_SIZE;
    const int thisWarpBucketIdx = tools::GetWarpIdx()/warpsPerBucket;
    
    int thisThreadSharedMemOffset = thisWarpBucketIdx*currentBucketSize + (tools::GetWarpIdx()%warpsPerBucket)*tools::WARP_SIZE + tools::GetThreadIdxWithinWarp();
    const int currentThreadOffset = tools::GetNumberOfWarpsInBlock()/warpsPerBucket*currentBucketSize;

    for( ; thisThreadSharedMemOffset < sharedMemSize; thisThreadSharedMemOffset += currentThreadOffset )
    {
      const int threadOffset1 = thisThreadSharedMemOffset;
      int val1 = sharedMem[threadOffset1];
      const int threadOffset2 = threadOffset1 + bucketOffset;
      int val2 = sharedMem[threadOffset2];

      int max = SelectMinMax<true>(val1, val2);
      int min = SelectMinMax<false>(val1, val2);

      if( increasing ) {
        sharedMem[threadOffset1] = min;
        sharedMem[threadOffset2] = max;
      } else {
        sharedMem[threadOffset1] = max;
        sharedMem[threadOffset2] = min;
      }
    }
    __syncthreads();
  }

  SortSequence64BucketSize_block_sync(sharedMem, sharedMemSize, increasing);
}