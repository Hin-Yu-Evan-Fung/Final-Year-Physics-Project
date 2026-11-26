#pragma once

#include "core/utils.hpp"
#include "core/vec3.hpp"

namespace hpmc {

inline Vec3 random_disp(PRNG &rng, double max_disp) {
  const double dx = rng.random_double(-max_disp, max_disp);
  const double dy = rng.random_double(-max_disp, max_disp);
  const double dz = rng.random_double(-max_disp, max_disp);
  return Vec3{dx, dy, dz};
}

} // namespace hpmc
