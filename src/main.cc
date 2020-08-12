#include <assert.h>

#include <iostream>

#include "cuda/bitonicSort.h"
#include "cuda/tools/cudaTimer.h"
#include "tools/randomGenerator.h"
#include "tools/timer.h"

//#define PROFILER_BUILD
const size_t DATA_SIZE0 = 8192 * 4;
const size_t DATA_SIZE1 = 8192 * 8;
const size_t DATA_SIZE2 = 8192 * 16;
const size_t DATA_SIZE3 = 8192 * 128 * 8;

#ifdef PROFILER_BUILD
const size_t WARMUP_RUNS = 0;
const size_t BENCHMARK_RUNS = 1;
#else
const size_t WARMUP_RUNS = 10;
const size_t BENCHMARK_RUNS = 100;
#endif

#define CHECK_ERROR(expression) \
  if (!expression) std::exit(-1)

// Returns true if sequence is bitonic with given bucket size.
bool CheckIfSequenceIsBitonic(const std::vector<int> &data, size_t bucketSize) {
  for (size_t i = 0; i < data.size(); i += 2 * bucketSize) {
    for (size_t j = 1; j < bucketSize; ++j)
      if (!(data[i + j - 1] <= data[i + j])) {
        std::cout << "Failed on checking increasing sequence at " << i + j
                  << " !" << std::endl;
        return false;
      }

    for (size_t j = 1; j < bucketSize; ++j)
      if (!(data[i + bucketSize + j - 1] >= data[i + bucketSize + j])) {
        std::cout << "Failed on checking decreasing sequence at "
                  << i + bucketSize + j << " !" << std::endl;
        return false;
      }
  }

  return true;
}

// Returns true if sequence is sorted.
bool CheckIfSequenceIsSorted(const std::vector<int> &data) {
  for (size_t i = 1; i < data.size(); ++i) {
    if (!(data[i - 1] <= data[i])) {
      std::cout << "Failed on checking sorted sequence at " << i << " !"
                << std::endl;
      return false;
    }
  }

  return true;
}

// Performs test and benchmark over randomly generated data of size dataSize.
void PerformTestAndBenchmark(size_t dataSize) {
  tools::CudaTimer cudaTimer;
  auto vec = tools::GenerateRandomInt(dataSize);
  int32_t *gpuPtr = nullptr;

  CUDA_ERROR_CHECK(cudaMalloc(&gpuPtr, vec.size() * sizeof(int32_t)));

  CUDA_ERROR_CHECK(cudaMemcpy(gpuPtr, vec.data(), vec.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));

  for (size_t i = 0; i < WARMUP_RUNS; ++i)
    CHECK_ERROR(BitonicSort(gpuPtr, vec.size()));

  cudaTimer.Start();
  for (int i = 0; i < BENCHMARK_RUNS; ++i)
    CHECK_ERROR(BitonicSort(gpuPtr, vec.size()));
  cudaTimer.Stop();

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaGetLastError());
  CUDA_ERROR_CHECK(cudaMemcpy(vec.data(), gpuPtr, vec.size() * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));

  std::cout << "Sorting " << vec.size() << " elements took on avg "
            << cudaTimer.GetElapsedTimeInMiliseconds() / BENCHMARK_RUNS
            << " ms gpu time!" << std::endl;

  CUDA_ERROR_CHECK(cudaFree(gpuPtr));

  if (!CheckIfSequenceIsSorted(vec)) {
    std::cout << "Sequence is NOT SORTED!\n";
    std::exit(-1);
  }

  std::cout << "Sequence is sorted!\n";
}

int main() {
  // Init cuda context.
  CUDA_ERROR_CHECK(cudaFree(0));

  PerformTestAndBenchmark(DATA_SIZE0);
  PerformTestAndBenchmark(DATA_SIZE1);
  PerformTestAndBenchmark(DATA_SIZE2);
  PerformTestAndBenchmark(DATA_SIZE3);

  return 0;
}