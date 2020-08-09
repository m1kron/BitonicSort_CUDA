#pragma once
#include <stdint.h>

#include <chrono>

#include "cuda_common.h"

namespace tools {
class CudaTimer {
 public:
  // Ctor.
  CudaTimer();

  // Dtor.
  ~CudaTimer();

  // Starts timer.
  void Start();

  // Stops timer.
  void Stop();

  // Returns elapsed time in miliseconds.
  float GetElapsedTimeInMiliseconds();

 private:
  cudaEvent_t m_start;
  cudaEvent_t m_stop;
};

//////////////////////////////////////////////////////////////////////////
//
// INLINES:
//
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
inline CudaTimer::CudaTimer() {
  CUDA_ERROR_CHECK(cudaEventCreate(&m_start));
  CUDA_ERROR_CHECK(cudaEventCreate(&m_stop));
}

//////////////////////////////////////////////////////////////////////////
inline CudaTimer::~CudaTimer() {
  CUDA_ERROR_CHECK(cudaEventDestroy(m_start));
  CUDA_ERROR_CHECK(cudaEventDestroy(m_stop));
}
//////////////////////////////////////////////////////////////////////////
inline void CudaTimer::Start() { CUDA_ERROR_CHECK(cudaEventRecord(m_start)); }

//////////////////////////////////////////////////////////////////////////
inline void CudaTimer::Stop() { CUDA_ERROR_CHECK(cudaEventRecord(m_stop)); }

//////////////////////////////////////////////////////////////////////////
inline float CudaTimer::GetElapsedTimeInMiliseconds() {
  float elapsedTime;
  CUDA_ERROR_CHECK(cudaEventSynchronize(m_stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, m_start, m_stop));
  return elapsedTime;
}

}  // namespace tools