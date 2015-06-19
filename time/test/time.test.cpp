// gtest
#include <gtest/gtest.h>
#include <xen/time/scoped_timer.h>
#include <xen/time/accumulating_timer.h>
#include <thread>

using namespace xen::time;
typedef std::chrono::milliseconds milli;

TEST(MathTests, scopedTimer) {

  int temp = 0;
  {
    ScopedTimer<> scoped("TEST");
    for (int i = 0; i < 100000; ++i) temp *= temp;
  }

}

TEST(MathTests, accumulatingTimer) {

  AccumulatingTimer<> acc;

  // Basic accumulate.
  EXPECT_EQ(acc.elapsed(), milli{0});
  acc.accumulate(milli{100});
  EXPECT_NE(acc.elapsed(), milli{0});
  EXPECT_EQ(acc.elapsed(), milli{100});

  std::this_thread::sleep_for(milli{10});

  // Shouldn't do anything.
  acc.stop();
  EXPECT_EQ(acc.elapsed(), milli{100});

  // Timer should be at 100 milliseconds now.
  acc.start();
  std::this_thread::sleep_for(milli{100});

  // Should be ~200ms now.
  EXPECT_GE(acc.elapsed(), milli{195});
  EXPECT_LE(acc.elapsed(), milli{205});

  acc.stop();
  std::this_thread::sleep_for(milli{100});

  // Stopped at around 200ms so should still be there.
  EXPECT_GE(acc.elapsed(), milli{195});
  EXPECT_LE(acc.elapsed(), milli{205});

  acc.start();
  std::this_thread::sleep_for(milli{100});
  acc.stop();
  std::this_thread::sleep_for(milli{100});

  // Should've just added another 100ms.
  EXPECT_GE(acc.elapsed(), milli{295});
  EXPECT_LE(acc.elapsed(), milli{305});

  // Scoped timer with accumulater in constructor should
  // add its duration to the accumulater on destruction.
  {
    ScopedTimer<> scopey(acc);
    std::this_thread::sleep_for(milli{100});
  }

  // Should've just added another 100ms.
  EXPECT_GE(acc.elapsed(), milli{395});
  EXPECT_LE(acc.elapsed(), milli{405});

  std::cout << "Total nanos: " << acc.elapsed().count() << std::endl;

}

