
#include "core/box.hpp"
#include "core/vec3.hpp"
#include <gtest/gtest.h>

using namespace hpmc;

TEST(BoxTest, DimensionsAndOrthorhombicWrapping) {
  Box cubic{10.0};
  Box ortho{5.0, 10.0, 15.0};

  Vec3 out_of_bounds{6.0, 11.0, 16.0};
  Vec3 wrapped = ortho.wrap(out_of_bounds);

  EXPECT_DOUBLE_EQ(wrapped.x, 1.0);
  EXPECT_DOUBLE_EQ(wrapped.y, 1.0);
  EXPECT_DOUBLE_EQ(wrapped.z, 1.0);
}

TEST(BoxTest, CoordinateWrappingBounds) {
  Box box{10.0};

  // Within bounds configuration
  Vec3 p1{2.0, 5.0, 8.0};
  Vec3 w1 = box.wrap(p1);
  EXPECT_DOUBLE_EQ(w1.x, 2.0);
  EXPECT_DOUBLE_EQ(w1.y, 5.0);
  EXPECT_DOUBLE_EQ(w1.z, 8.0);

  // Overflow boundary test (12.0 wraps to 2.0, -3.0 wraps to 7.0, 15.0 wraps
  // to 5.0)
  Vec3 p2{12.0, -3.0, 15.0};
  Vec3 w2 = box.wrap(p2);
  EXPECT_DOUBLE_EQ(w2.x, 2.0);
  EXPECT_DOUBLE_EQ(w2.y, 7.0);
  EXPECT_DOUBLE_EQ(w2.z, 5.0);
}

TEST(BoxTest, MinimumImageConventionDist2) {
  Box box{10.0};

  // Standard distance inside box
  // dx = 3, dy = 4, dz = 0 -> dist2 = 9 + 16 = 25
  Vec3 v1{1.0, 1.0, 1.0};
  Vec3 v2{4.0, 5.0, 1.0};
  EXPECT_DOUBLE_EQ(box.dist2(v1, v2), 25.0);

  // Across periodic boundary: p_a at (1,1,1), p_b at (9,1,1) in a 10.0 box.
  // Raw difference is 8, but periodic minimum image is -2 away. (-2)^2 = 4
  Vec3 p_a{1.0, 1.0, 1.0};
  Vec3 p_b{9.0, 1.0, 1.0};
  EXPECT_DOUBLE_EQ(box.dist2(p_a, p_b), 4.0);
}
