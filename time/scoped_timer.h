////////////////////////////////////////////////////////////////////////////////
///
/// @file scoped_timer.h
/// @brief Simple scoped timer class to log elapsed times.
/// @author XeN
/// @author Zenon Parker
/// @date 2014
///
////////////////////////////////////////////////////////////////////////////////

#ifndef XEN_TIME_SCOPED_TIMER
#define XEN_TIME_SCOPED_TIMER

#include <chrono>
#include <xen/time/accumulating_timer.h>

namespace xen {
namespace time {

template <typename TickSize = NANO>
class ScopedTimer {
public:
  typedef std::chrono::high_resolution_clock clock;

  explicit ScopedTimer(const std::string& name) : m_name(name) {
    m_start = clock::now();
  }

  /** @brief Allow the creation of an unnamed scoped timer that
    *        simply adds its duration to an accumulating timer
    *        upon destruction.
    */
  explicit ScopedTimer(AccumulatingTimer<TickSize>& accTimer) 
    : ScopedTimer("") {
    m_hasAccumulater = true;
    m_accTimer = &accTimer;
  }

  ~ScopedTimer() {
    if (m_hasAccumulater && m_accTimer) {
      m_accTimer->accumulate(std::chrono::duration_cast<TickSize>(clock::now() - m_start));
    } else {
      std::cout << "Timer [" << m_name << "] expired after "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(
                     clock::now() - m_start).count()
                << " nanoseconds." << std::endl;
    }
  }

private:
  std::string m_name;
  clock::time_point m_start;
  bool m_hasAccumulater = false;
  AccumulatingTimer<TickSize>* m_accTimer = nullptr;
};

} // end namespace time
} // end namespace xen

#endif // XEN_TIME_SCOPED_TIMER

