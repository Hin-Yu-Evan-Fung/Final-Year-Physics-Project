#pragma once

#include "core/box.hpp"
#include "core/particle.hpp"
#include "core/vec3.hpp"

namespace hpmc {

struct HardSphere;

// Move definiton
template <> struct Move<HardSphere> {
  static constexpr size_t dof = 3;

  Vec3 vec;
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

} // namespace hpmc
