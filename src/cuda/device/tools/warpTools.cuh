#pragma once

namespace tools {

const uint32_t ALL_THREADS_MASK = 0xFFFFFFFF;

const uint32_t WARP_SIZE = 32;

// Returns index of given warp that calling thread belongs to.
// NOTE: Works only for 1d blocks.
__device__ uint32_t GetWarpIdx();

// Returns number of warps in block.
// NOTE: Works only for 1d blocks.
__device__ uint32_t GetNumberOfWarpsInBlock();

// Returns thread's in warp.
// NOTE: Works only for 1d blocks.
__device__ uint32_t GetThreadIdxWithinWarp();


////////////////////////////////////////////////////////////////
//
// INLINES:
//
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
__device__ __forceinline__ uint32_t GetWarpIdx()
{
    return threadIdx.x / WARP_SIZE;
}

////////////////////////////////////////////////////////////////
__device__ __forceinline__ uint32_t GetNumberOfWarpsInBlock()
{
    return blockDim.x / WARP_SIZE;
}

////////////////////////////////////////////////////////////////
__device__  __forceinline__ uint32_t GetThreadIdxWithinWarp()
{
    return threadIdx.x % WARP_SIZE;
}
}