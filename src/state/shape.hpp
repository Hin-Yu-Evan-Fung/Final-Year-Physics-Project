#pragma once

#include "core/lattice.hpp"

#include <stdexcept>

namespace hpmc {

template <typename C>
concept ShapeConfig =
    requires(C cfg, const Lattice &lat, const Vec3 &r1, const Vec3 &r2) {
      // Minimum cell width required for neighbor list correctness
      { cfg.min_cell_width() } -> std::convertible_to<double>;

      // Pairwise overlap check
      { cfg.overlap(lat, r1, r2) } -> std::same_as<bool>;
    };

struct SphereConfig {
  double radius;

  SphereConfig(double radius) : radius(radius) {
    if (radius <= 0.0)
      throw std::invalid_argument("Sphere radius must be strictly positive");
  }

  constexpr double min_cell_width() const { return radius * 2; }

  // Overlap rule for spheres
  constexpr bool overlap(const Lattice &lattice, const Vec3 &r1,
                         const Vec3 &r2) const {
    return lattice.dist2(r1, r2) < (4.0 * radius * radius); // 4 * r^2
  }
};

} // namespace hpmc
