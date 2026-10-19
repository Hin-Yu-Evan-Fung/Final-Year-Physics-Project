
#include "core/vec3.hpp"
#include <gtest/gtest.h>

using namespace hsmc;

TEST(Vec3Test, DefaultConstructorZeroesCoordinates) {
  // Verify default constructor zeroes out the coordinate states
  Vec3 v;
  EXPECT_DOUBLE_EQ(v.x, 0.0);
  EXPECT_DOUBLE_EQ(v.y, 0.0);
  EXPECT_DOUBLE_EQ(v.z, 0.0);
}

TEST(Vec3Test, ArithmeticOperators) {
  Vec3 v1{1.0, 2.0, 3.0};
  Vec3 v2{4.0, 5.0, 6.0};

  // Vector Addition
  Vec3 v_sum = v1 + v2;
  EXPECT_DOUBLE_EQ(v_sum.x, 5.0);
  EXPECT_DOUBLE_EQ(v_sum.y, 7.0);
  EXPECT_DOUBLE_EQ(v_sum.z, 9.0);

  // Scalar Multiplication
  Vec3 scaled = v1 * 2.5;
  EXPECT_DOUBLE_EQ(scaled.x, 2.5);
  EXPECT_DOUBLE_EQ(scaled.y, 5.0);
  EXPECT_DOUBLE_EQ(scaled.z, 7.5);
}

TEST(Vec3Test, DotProduct) {
  Vec3 x_axis{1.0, 0.0, 0.0};
  Vec3 y_axis{0.0, 1.0, 0.0};
  Vec3 v{2.0, 3.0, 4.0};

  // Verify orthogonal dot products evaluate to zero
  EXPECT_DOUBLE_EQ(x_axis.dot(y_axis), 0.0);

  // Verify directional scalar projection works correctly
  EXPECT_DOUBLE_EQ(v.dot(x_axis), 2.0);
}

TEST(Vec3Test, Norms) {
  Vec3 v{3.0, 4.0, 0.0};
  // Verify norm squared mapping matches 3^2 + 4^2 = 25
  EXPECT_DOUBLE_EQ(v.norm2(), 25.0);
}
