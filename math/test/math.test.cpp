// gtest
#include <gtest/gtest.h>
#include <xen/math/math.h>

using namespace xen::math;

TEST(MathTests, equalityAndComparison) {

  std::vector<double> v = { 1000, 1000.0000000001, 1000.00001 };

  // Equality
  EXPECT_TRUE(areEqual(v[0], v[1]));
  EXPECT_FALSE(areEqual(v[0], v[2]));

  // Comparison
  EXPECT_EQ(compare(v[1], v[2]), -1);
  EXPECT_EQ(compare(v[2], v[1]), 1);
  EXPECT_EQ(compare(v[0], v[1]), 0);
  EXPECT_EQ(compare(v[0], v[1], 0.00000000001), -1);
                            
}

TEST(MathTests, sigmoid) {

  EXPECT_TRUE(areEqual(sigmoid(-5.0), 0.0066929));
  EXPECT_TRUE(areEqual(sigmoid(-2.0), 0.1192029));
  EXPECT_TRUE(areEqual(sigmoid(-1.0), 0.2689414));
  EXPECT_TRUE(areEqual(sigmoid(-0.5), 0.3775407));
  EXPECT_TRUE(areEqual(sigmoid(0.0), 0.5));
  EXPECT_TRUE(areEqual(sigmoid(0.5), 0.6224593));
  EXPECT_TRUE(areEqual(sigmoid(1.0), 0.7310586));
  EXPECT_TRUE(areEqual(sigmoid(2.0), 0.8807971));
  EXPECT_TRUE(areEqual(sigmoid(5.0), 0.9933071));

}

TEST(MathTests, sigmoidGradient) {

  EXPECT_TRUE(areEqual(sigmoidGradient(-5.0), 0.00664805));
  EXPECT_TRUE(areEqual(sigmoidGradient(-2.0), 0.10499358));
  EXPECT_TRUE(areEqual(sigmoidGradient(-1.0), 0.19661193));
  EXPECT_TRUE(areEqual(sigmoidGradient(-0.5), 0.23500371));
  EXPECT_TRUE(areEqual(sigmoidGradient(0.0), 0.25000000));
  EXPECT_TRUE(areEqual(sigmoidGradient(0.5), 0.23500371));
  EXPECT_TRUE(areEqual(sigmoidGradient(1.0), 0.19661193));
  EXPECT_TRUE(areEqual(sigmoidGradient(2.0), 0.10499358));
  EXPECT_TRUE(areEqual(sigmoidGradient(5.0), 0.00664805));

}

