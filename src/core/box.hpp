#pragma once

#include "vec3.hpp"

#include <cmath>

namespace hsmc {

class Box {
  Vec3 dim_;

  static constexpr double wrap_1d(double coord, double dim) {
    return coord - dim * std::floor(coord / dim);
  }
  static constexpr double diff_1d(double coord, double dim) {
    return coord - dim * std::round(coord / dim);
  }

public:
  constexpr Box(double lx, double ly, double lz) : dim_(lx, ly, lz) {}
  constexpr Box(double l) : dim_(l, l, l) {}

  constexpr Vec3 get_dims() const { return dim_; }

  // Wraps arbitrary coordinate back into the fundamental box domain [0, L_i]
  constexpr Vec3 wrap(const Vec3 &v) const {
    return {wrap_1d(v.x, dim_.x), wrap_1d(v.y, dim_.y), wrap_1d(v.z, dim_.z)};
  }
  // Calculate difference between two vectors based on the minimum image
  // convention
  constexpr Vec3 diff(const Vec3 &v1, const Vec3 &v2) const {
    Vec3 d = v2 - v1;
    return {diff_1d(d.x, dim_.x), diff_1d(d.y, dim_.y), diff_1d(d.z, dim_.z)};
    ;
  }
  // Calculate distance squared between two vectors based on the minimum image
  // convention
  constexpr double dist2(const Vec3 &v1, const Vec3 &v2) const {
    return diff(v1, v2).norm2();
  }
};

} // namespace hsmc
