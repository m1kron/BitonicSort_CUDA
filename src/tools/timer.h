#pragma once
#include <stdint.h>

#include <chrono>

namespace tools {
class Timer {
 public:
  // Starts timer.
  void Start();

  // Stops timer.
  void Stop();

  // Returns elapsed time in miliseconds.
  float GetElapsedTimeInMiliseconds();

  // Returns elapsed time in micorseconds.
  int64_t GetElapsedTimeInMicroseconds();

 private:
  std::chrono::time_point<std::chrono::system_clock> m_start;
  std::chrono::time_point<std::chrono::system_clock> m_stop;
};

//////////////////////////////////////////////////////////////////////////
//
// INLINES:
//
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
inline void Timer::Start() { m_start = std::chrono::system_clock::now(); }

//////////////////////////////////////////////////////////////////////////
inline void Timer::Stop() { m_stop = std::chrono::system_clock::now(); }

//////////////////////////////////////////////////////////////////////////
inline float Timer::GetElapsedTimeInMiliseconds() {
  const float diff_float =
      static_cast<float>(GetElapsedTimeInMicroseconds()) / 1000.0f;

  return diff_float;
}

//////////////////////////////////////////////////////////////////////////
inline int64_t Timer::GetElapsedTimeInMicroseconds() {
  const auto diff =
      std::chrono::duration_cast<std::chrono::microseconds>(m_stop - m_start);
  return static_cast<int64_t>(diff.count());
}

}  // namespace tools