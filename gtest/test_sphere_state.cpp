#include "core/lattice.hpp"
#include "core/vec3.hpp"
#include "state/shape.hpp"
#include "state/state.hpp"

#include <gtest/gtest.h>

using namespace hpmc;

class StateTest : public ::testing::TestWithParam<Lattice> {
protected:
  SphereConfig config_{1.0};
  Lattice lattice_ = GetParam();
  State state_{config_, lattice_};
};

TEST_P(StateTest, SetCoords) {
  const Vec3 p0 = lattice_.to_cart({0.1, 0.1, 0.1});
  const Vec3 p1 = lattice_.to_cart({0.5, 0.5, 0.5});

  state_.set_coords({{p0.x, p1.x}}, {{p0.y, p1.y}}, {{p0.z, p1.z}});

  ASSERT_EQ(state_.n(), 2);
  EXPECT_NEAR(state_.pos_at(0).x, p0.x, 1e-12);
  EXPECT_NEAR(state_.pos_at(1).x, p1.x, 1e-12);
}

TEST_P(StateTest, TranslationAccept) {
  const Vec3 p0 = lattice_.to_cart({0.1, 0.1, 0.1});
  state_.set_coords({{p0.x}}, {{p0.y}}, {{p0.z}});

  const Vec3 displacement{0.5, -0.2, 0.1};
  const Vec3 expected = lattice_.wrap(p0 + displacement);

  const PosCell proposed = state_.translated(0, displacement);
  const Vec3 new_pos = proposed.pos;
  const int new_cell = proposed.cell;

  EXPECT_FALSE(state_.has_overlaps(new_pos, new_cell, 0));
  state_.set_pos(0, new_pos, new_cell);

  EXPECT_NEAR(state_.pos_at(0).x, expected.x, 1e-12);
  EXPECT_NEAR(state_.pos_at(0).y, expected.y, 1e-12);
  EXPECT_NEAR(state_.pos_at(0).z, expected.z, 1e-12);
  EXPECT_EQ(state_.cell_of(0), proposed.cell);
}

TEST_P(StateTest, TranslationReject) {
  const Vec3 p0 = lattice_.to_cart({0.1, 0.1, 0.1});
  const Vec3 p1 = lattice_.to_cart({0.5, 0.5, 0.5});
  state_.set_coords({{p0.x, p1.x}}, {{p0.y, p1.y}}, {{p0.z, p1.z}});

  // Displace particle 0 right into particle 1
  const Vec3 displacement = p1 - p0;
  const Vec3 new_pos = state_.displaced_pos(0, displacement);
  const int new_cell = state_.cell_of(new_pos);

  EXPECT_TRUE(state_.has_overlaps(new_pos, new_cell, 0));

  // Verify coordinates did not mutate
  EXPECT_NEAR(state_.pos_at(0).x, p0.x, 1e-12);
  EXPECT_NEAR(state_.pos_at(0).y, p0.y, 1e-12);
  EXPECT_NEAR(state_.pos_at(0).z, p0.z, 1e-12);
}

TEST_P(StateTest, TranslatePBC) {
  // Start near the upper fractional boundary (0.95) and push past 1.0 -> wraps
  // to 0.05
  const Vec3 p0 = lattice_.to_cart({0.95, 0.1, 0.1});
  const Vec3 expected = lattice_.to_cart({0.05, 0.1, 0.1});
  state_.set_coords({{p0.x}}, {{p0.y}}, {{p0.z}});

  // Shift by +0.1 along the first lattice vector `a`
  const Vec3 displacement = lattice_.a * 0.1;
  const Vec3 new_pos = state_.displaced_pos(0, displacement);
  const int new_cell = state_.cell_of(new_pos);

  EXPECT_FALSE(state_.has_overlaps(new_pos, new_cell, 0));
  state_.set_pos(0, new_pos, new_cell);

  EXPECT_NEAR(state_.pos_at(0).x, expected.x, 1e-12);
  EXPECT_NEAR(state_.pos_at(0).y, expected.y, 1e-12);
  EXPECT_NEAR(state_.pos_at(0).z, expected.z, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
    Lattices, StateTest,
    ::testing::Values(
        // Standard Cubic (10 x 10 x 10)
        Lattice::from_upper_triangular({10.0, 0.0, 0.0}, {0.0, 10.0, 0.0},
                                       {0.0, 0.0, 10.0}),
        // Orthorhombic
        Lattice::from_upper_triangular({8.0, 0.0, 0.0}, {0.0, 10.5, 0.0},
                                       {0.0, 0.0, 12.0}),
        // Hexagonal / HCP-style (60-degree xy angle)
        Lattice::from_upper_triangular({10.0, 0.0, 0.0}, {5.0, 8.66025, 0.0},
                                       {0.0, 0.0, 11.0}),
        // Strongly Sheared Triclinic
        Lattice::from_upper_triangular({10.0, 0.0, 0.0}, {4.0, 9.0, 0.0},
                                       {-3.0, 3.5, 8.5})));
