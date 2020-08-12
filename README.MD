## Bitonic sort implementation for CUDA
The repository contains an implementation of bitonic sort algorithm.

### Overview
NOTE: The code requires at least CUDA 9 and was tested on CUDA 9.1 and CUDA 10.1 on Turing and Kepler architectures. During development I rather thinking about Turing architecture.

Sorting is memory-bound operation so key success is to optimize memory accesses. 
Because of that I tried to do as much as I can in on-chip memory(shared memory plus registers). 
The code is structured around primitives that are responsible for some data chunk.
The first-level primitives are warp primitives. Single warp is responsible for sorting currently a chunk of 256 items(however it depends on launch params). It uses register memory to sort the chunk. There is a trade-off between how many registers a thread 
can have and how many threads per block kernel can have.
The second level primitives is block primitives. Block operates on data stored at shared memory.
Last level is currently kernel call, which operates on gpu memory.
The code assumes that data to sort will fit into GPU memory.

### TODO list
1. Make sure that code will work with block-stride access. 
Currently I launch dataSize/SharedMemSize blocks, however ideally I would like to launch not much more blocks than the number of SMs. With careful implementation it would increase the number of memory operations on-the-fly, as distant memory regions can be sorted independently.

2. Further increase ILP by fine-tuning parameters for block and warp primitives. 
Currently some primitives(especially warp-ones), carries long-dependency chain operations(e.g. PerformStage_WarpWide_sync). Maybe it would be beneficial to make warp responsible for bigger chunk size(at the cost of less threads per block).

3. Check if replacing main for-loop with persistent kernel will be beneficial.
The problem is that I will need a infra-block synchronization. I could use for this cooperative_groups(but that requires rdc) 
or write my own synchronization primitives. In any case, the number of blocks cannot exceed number of SMs.
What is more, I don't think it will give a massive boost, since current main-loop kernels are completely global memory-bound.

4. Add support of other types.