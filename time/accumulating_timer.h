////////////////////////////////////////////////////////////////////////////////
///
/// @file accumulating_timer.h
/// @brief Timer to accumulate time periods and track total elapsed time.
/// @author XeN
/// @author Zenon Parker
/// @date 2014
///
////////////////////////////////////////////////////////////////////////////////

#ifndef XEN_TIME_ACCUMULATING_TIMER
#define XEN_TIME_ACCUMULATING_TIMER

#include <chrono>

namespace xen {
namespace time {

typedef std::chrono::milliseconds MILLI;
typedef std::chrono::microseconds MICRO;
typedef std::chrono::nanoseconds NANO;

template <typename TickSize = NANO>
class AccumulatingTimer {
public:
  typedef std::chrono::high_resolution_clock clock;

  AccumulatingTimer() : m_totalElapsed(0) {}

  /** @brief Start accumulating time.
    */
  void start() {
    if (!m_isRunning) {
      m_start = clock::now();
      m_isRunning = true;
    }
  }

  /** @brief Stop accumulating time.
    */
  void stop() {
    if (m_isRunning) {
      m_totalElapsed += std::chrono::duration_cast<TickSize>(clock::now() - m_start);
      m_isRunning = false;
    }
  }

  /** @brief Manually adds time to the accumulation.
    */
  void accumulate(TickSize accTime) {
    m_totalElapsed += accTime;
  }

  TickSize elapsed() const {
    if (m_isRunning) {
      return m_totalElapsed + std::chrono::duration_cast<TickSize>(clock::now() - m_start);
    } else {
      return m_totalElapsed;
    }
  }

private:

  TickSize m_totalElapsed;
  clock::time_point m_start;
  bool m_isRunning = false;
};

} // end namespace time
} // end namespace xen

#endif // XEN_TIME_ACCUMULATING_TIMER

