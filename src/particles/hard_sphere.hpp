#pragma once

#include "core/box.hpp"
#include "core/particle.hpp"
#include "core/utils.hpp"
#include "core/vec3.hpp"

namespace hpmc {

struct HardSphere;

// Move definiton
template <> struct Move<HardSphere> {
  static constexpr size_t dof = 3;

  Vec3 vec;

  static Move<HardSphere> generate_move(double max, PRNG &rng) {
    const double dx = (rng.random_double() - 0.5) * 2.0 * max;
    const double dy = (rng.random_double() - 0.5) * 2.0 * max;
    const double dz = (rng.random_double() - 0.5) * 2.0 * max;
    return {{dx, dy, dz}};
  };
};

// Active arguments (parameters that are different per instance)
struct HardSphere {
  static constexpr size_t dof = 3;

  Vec3 pos;

  constexpr HardSphere move(const Move<HardSphere> &m) const {
    return {pos + m.vec};
  }
};

// Passive arguments (parameters that are constant)
struct HardSphereParams {
  double sigma;
};

// Function to check if there is an overlap
constexpr bool has_overlap(const Box &box, const HardSphere &h1,
                           const HardSphere &h2, const HardSphereParams &p) {
  return box.dist2(h1.pos, h2.pos) < 4 * p.sigma * p.sigma;
}

static_assert(sizeof(HardSphere) == HardSphere::dof * sizeof(double),
              "Layout mismatch: HardSphere struct size does not match its DOF "
              "footprint!");

static_assert(std::is_trivially_copyable_v<HardSphere>,
              "Safety violation: HardSphere must be trivially copyable for "
              "zero-copy views!");

static_assert(
    sizeof(Move<HardSphere>) == Move<HardSphere>::dof * sizeof(double),
    "Layout mismatch: Move struct size does not match its DOF footprint!");

static_assert(MoveGeneratable<HardSphere>,
              "Move<Particle>::generate_moves() is not implemented");

} // namespace hpmc
