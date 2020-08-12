#Bitonic sort implementation for CUDA
The repository contains an implementation of bitonic sort algorithm.

## Overview
NOTE: The code requires at least CUDA 9 and was tested on CUDA 9.1 and CUDA 10.1 on Turing and Kepler archtectures.
NOTE: During development I cared rather about Turing architecture.

Sorting is memory-bound operation so key success is to optimze memory accesses. 
Becasue of that I tried to do as much as I can in on chip memory(shared memory plus registers). 
The is structured around primitives that can sort some data chunk.
The most first-level primitives is warp primitives. Single warp is responsile for sorting currently a chunk of 256 items(however it depends on launch params). It does it but loading items into registers and perform inregister sort over data. The idea was to maximally use the fastest memory available(registers) and increase ILP per thread. Threre is a tradoff how many registers a thread 
can have and how many thread per block you can have.
The second level primitives is block primitives. Block sorts data stored in shared memory.
Last level is currently kernel call, which operates on gpu memory.
The code assummes that data to sort will fit into GPU memory.

## TODO
1. Make sure that code will work with block-stride access. 
Currently I launch dataSize/SharedMemSize blocks, however ideally I would like to lauch not much more then the number of sms. With carefull implementation, that would increase the number of memory opration is flight, as distant memory regions can be sorted independently.

2. Further increase ILP by fine-tuning parameters for block and warp primitives. Currently some primitives(especially warp-ones), carries long-dependency chain operations(e.g. PerformStage_WarpWide_sync). Mayebe it would be beneficial to even wider warp chunk size(at the cost of less threads per block), since currently block sore is do y different kernel calls.

3. Check if replacing main for-loop with persistent kernel will be beneficial
The problem is that I will need a infra-block synchronization. I could use for this cooperative_groups(but that reguires rdc) 
or write my own synchronization primitvies. In any case, the number of blocks cannot exceed number of SMs.
What is more, I don't think it will give a massive boost, since current main-loop kernels are completetly global memory-bound.

4. Add support of other types.