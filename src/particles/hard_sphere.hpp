#pragma once

#include "core/box.hpp"
#include "core/vec3.hpp"

namespace hpmc {

struct SphereConfig {
  double radius;

  // Encapsulated overlap rule for spheres
  inline bool overlap_naive(const Box &box, const Vec3 &p1,
                            const Vec3 &p2) const {
    return box.dist2(p1, p2) < (4.0 * radius * radius); // 4 * r^2
  }
};

} // namespace hpmc
