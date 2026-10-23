#include "core/vec3.hpp"
#include "particle/sphere.hpp"
#include "state/state.hpp"
#include <gtest/gtest.h>

using namespace hpmc;

class StateTest : public ::testing::Test {
protected:
  double box_length_;
  SphereConfig config_;

  void SetUp() override {
    box_length_ = 10.0;
    config_.radius = 1.0;
  }
};

TEST_F(StateTest, SetCoords) {
  State state(ParticleType::Sphere, box_length_, config_);

  std::vector<double> initial_x = {1.0, 5.0};
  std::vector<double> initial_y = {1.0, 5.0};
  std::vector<double> initial_z = {1.0, 5.0};

  state.set_coords(initial_x, initial_y, initial_z);

  ASSERT_EQ(state.n(), 2);
  EXPECT_DOUBLE_EQ(state.pos_at(0).x, 1.0);
  EXPECT_DOUBLE_EQ(state.pos_at(1).x, 5.0);
}

TEST_F(StateTest, TranslationAccept) {
  State state(ParticleType::Sphere, box_length_, config_);
  std::vector<double> initial_x = {1.0};
  std::vector<double> initial_y = {1.0};
  std::vector<double> initial_z = {1.0};
  state.set_coords(initial_x, initial_y, initial_z);

  Vec3 displacement{0.5, -0.2, 0.1};
  state.try_translate(0, displacement);

  // Verify the state has mutated immediately in memory
  EXPECT_DOUBLE_EQ(state.pos_at(0).x, 1.5);
  EXPECT_DOUBLE_EQ(state.pos_at(0).y, 0.8);
  EXPECT_DOUBLE_EQ(state.pos_at(0).z, 1.1);
}

TEST_F(StateTest, TranslationReject) {
  State state(ParticleType::Sphere, box_length_, config_);
  std::vector<double> initial_x = {1.0, 3.5};
  std::vector<double> initial_y = {1.0, 4.0};
  std::vector<double> initial_z = {1.0, 0.0};
  state.set_coords(initial_x, initial_y, initial_z);

  Vec3 displacement{2.5, 3.0, -1.0};
  state.try_translate(0, displacement);

  EXPECT_DOUBLE_EQ(state.pos_at(0).x, 1.0);
  EXPECT_DOUBLE_EQ(state.pos_at(0).y, 1.0);
  EXPECT_DOUBLE_EQ(state.pos_at(0).z, 1.0);
}

TEST_F(StateTest, TranslatePBC) {
  State state(ParticleType::Sphere, box_length_, config_);
  std::vector<double> initial_x = {9.5};
  std::vector<double> initial_y = {1.0};
  std::vector<double> initial_z = {1.0};

  state.set_coords(initial_x, initial_y, initial_z);

  // Pushing X past the box boundary (9.5 + 1.0 = 10.5 -> wraps to 0.5)
  Vec3 displacement{1.0, 0.0, 0.0};
  state.try_translate(0, displacement);

  EXPECT_DOUBLE_EQ(state.pos_at(0).x, 0.5);
}

TEST_F(StateTest, DetectsOverlaps) {
  State state(ParticleType::Sphere, box_length_, config_);

  std::vector<double> x = {1.0, 2.5};
  std::vector<double> y = {1.0, 1.0};
  std::vector<double> z = {1.0, 1.0};
  state.set_coords(x, y, z);

  EXPECT_TRUE(state.check_overlaps_naive(0, Vec3{1.0, 1.0, 1.0}));
  EXPECT_TRUE(state.check_overlaps_naive(1, Vec3{2.5, 1.0, 1.0}));

  EXPECT_EQ(state.count_overlaps_naive(), 1);
}

TEST_F(StateTest, NoOverlap) {
  State state(ParticleType::Sphere, box_length_, config_);

  std::vector<double> x = {1.0, 4.0}; // Separation along X = 3.0 (Safe)
  std::vector<double> y = {1.0, 1.0};
  std::vector<double> z = {1.0, 1.0};
  state.set_coords(x, y, z);

  EXPECT_FALSE(state.check_overlaps_naive(0, Vec3{1.0, 1.0, 1.0}));
  EXPECT_EQ(state.count_overlaps_naive(), 0);
}

TEST_F(StateTest, OverlapPBC) {
  State state(ParticleType::Sphere, box_length_, config_);

  std::vector<double> x = {0.5, 9.5};
  std::vector<double> y = {1.0, 1.0};
  std::vector<double> z = {1.0, 1.0};
  state.set_coords(x, y, z);

  EXPECT_TRUE(state.check_overlaps_naive(0, Vec3{0.5, 1.0, 1.0}));
  EXPECT_EQ(state.count_overlaps_naive(), 1);
}
