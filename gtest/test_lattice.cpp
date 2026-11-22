#include "core/lattice.hpp"
#include "core/utils.hpp"

#include <gtest/gtest.h>

using namespace hpmc;

namespace {

Lattice random_lat(PRNG &rng) {
  Vec3 a{rng.random_double(2.0, 10.0), 0.0, 0.0};
  Vec3 b{rng.random_double(-2.0, 2.0), rng.random_double(2.0, 10.0), 0.0};
  Vec3 c{rng.random_double(-2.0, 2.0), rng.random_double(-2.0, 2.0),
         rng.random_double(2.0, 10.0)};

  return Lattice::from_upper_triangular(a, b, c);
}

Vec3 random_vec3(PRNG &rng) {
  return {rng.random_double(-50.0, 50.0), rng.random_double(-50.0, 50.0),
          rng.random_double(-50.0, 50.0)};
}

} // namespace

TEST(LatticeTest, RoundTrip) {
  PRNG rng(12345);

  for (unsigned iter = 0; iter < 1000; ++iter) {
    auto lat = random_lat(rng);

    Vec3 r = random_vec3(rng);
    Vec3 r_back = lat.to_cart(lat.to_frac(r));
    EXPECT_NEAR(r_back.x, r.x, 1e-11);
    EXPECT_NEAR(r_back.y, r.y, 1e-11);
    EXPECT_NEAR(r_back.z, r.z, 1e-11);
  }
}

TEST(LatticeTest, Wrap) {
  PRNG rng(12345);

  for (unsigned iter = 0; iter < 1000; ++iter) {
    auto lat = random_lat(rng);

    Vec3 r = random_vec3(rng);

    // Correct wrapping
    Vec3 wrapped = lat.wrap(r);
    Vec3 s_wrapped = lat.to_frac(wrapped);
    EXPECT_GE(s_wrapped.x, 0.0);
    EXPECT_LT(s_wrapped.x, 1.0);
    EXPECT_GE(s_wrapped.y, 0.0);
    EXPECT_LT(s_wrapped.y, 1.0);
    EXPECT_GE(s_wrapped.z, 0.0);
    EXPECT_LT(s_wrapped.z, 1.0);

    int nx = rng.random_index(10);
    int ny = rng.random_index(10);
    int nz = rng.random_index(10);

    // Preserves periodicity
    Vec3 shifted = r + lat.a * nx - lat.b * ny + lat.c * nz;
    Vec3 wrapped_shifted = lat.wrap(shifted);
    EXPECT_NEAR(wrapped_shifted.x, wrapped.x, 1e-11);
    EXPECT_NEAR(wrapped_shifted.y, wrapped.y, 1e-11);
    EXPECT_NEAR(wrapped_shifted.z, wrapped.z, 1e-11);
  }
}

TEST(LatticeTest, MinImg) {
  PRNG rng(12345);

  for (unsigned iter = 0; iter < 1000; ++iter) {
    auto lat = random_lat(rng);

    int nx = rng.random_index(10);
    int ny = rng.random_index(10);
    int nz = rng.random_index(10);

    Vec3 r1 = random_vec3(rng);
    Vec3 r2 = random_vec3(rng);
    Vec3 r1_shifted = r1 + lat.a * nx - lat.b * ny + lat.c * nz;

    Vec3 diff = lat.min_img(r2 - r1);
    Vec3 diff_neg = lat.min_img(r1 - r2);
    Vec3 diff_shifted = lat.min_img(r2 - r1_shifted);

    EXPECT_DOUBLE_EQ(diff.x, -diff_neg.x);
    EXPECT_DOUBLE_EQ(diff.y, -diff_neg.y);
    EXPECT_DOUBLE_EQ(diff.z, -diff_neg.z);

    EXPECT_NEAR(diff.x, diff_shifted.x, 1e-11);
    EXPECT_NEAR(diff.y, diff_shifted.y, 1e-11);
    EXPECT_NEAR(diff.z, diff_shifted.z, 1e-11);
  }
}

TEST(LatticeTest, PerpendicularWidths) {
  PRNG rng(12345);

  for (unsigned iter = 0; iter < 1000; ++iter) {
    auto lat = random_lat(rng);
    Vec3 w = lat.perp_widths();

    // Unit normal to the (b, c) face; projecting 'a' onto it gives w.x
    Vec3 bc_cross = lat.b.cross(lat.c);
    Vec3 nx = bc_cross / bc_cross.norm();
    EXPECT_NEAR(std::abs(lat.a.dot(nx)), w.x, 1e-11);

    // Unit normal to the (c, a) face; projecting 'b' onto it gives w.y
    Vec3 ca_cross = lat.c.cross(lat.a);
    Vec3 ny = ca_cross / ca_cross.norm();
    EXPECT_NEAR(std::abs(lat.b.dot(ny)), w.y, 1e-11);

    // Unit normal to the (a, b) face; projecting 'c' onto it gives w.z
    Vec3 ab_cross = lat.a.cross(lat.b);
    Vec3 nz = ab_cross / ab_cross.norm();
    EXPECT_NEAR(std::abs(lat.c.dot(nz)), w.z, 1e-11);
  }
}
