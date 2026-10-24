#include "core/utils.hpp"
#include <gtest/gtest.h>

using namespace hpmc;

class PRNGTest : public ::testing::Test {
protected:
  void SetUp() override {}
};

TEST_F(PRNGTest, DeterministicOutputs) {
  PRNG rng1(42);
  PRNG rng2(42);

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(rng1.random_u64(), rng2.random_u64());
  }
}

TEST_F(PRNGTest, RandomDoubleBounds) {
  PRNG rng(12345);

  for (int i = 0; i < 10'000; ++i) {
    double val = rng.random_double();
    EXPECT_GE(val, 0.0);
    EXPECT_LT(val, 1.0);
  }
}

TEST_F(PRNGTest, RandomIndexBounds) {
  PRNG rng(999);
  size_t N = 10;

  for (int i = 0; i < 10'000; ++i) {
    size_t idx = rng.random_index(N);
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, N);
  }
}

TEST_F(PRNGTest, NegativeAndSpanningRangeBounds) {
  PRNG rng(54321);
  double min = -100.0;
  double max = 50.0;

  for (int i = 0; i < 50'000; ++i) {
    double val = rng.random_double(min, max);
    EXPECT_GE(val, min) << "Value " << val << " fell below negative minimum "
                        << min;
    EXPECT_LT(val, max) << "Value " << val << " exceeded maximum " << max;
  }
}

TEST_F(PRNGTest, ZeroWidthRange) {
  PRNG rng(999);
  double target = 7.77;

  for (int i = 0; i < 100; ++i) {
    double val = rng.random_double(target, target);
    EXPECT_DOUBLE_EQ(val, target);
  }
}
