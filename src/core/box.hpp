#pragma once

#include "vec3.hpp"

#include <cmath>

namespace hpmc {

class Box {
  Vec3 dim_;
  Vec3 inv_dim_;

public:
  constexpr Box() : dim_{}, inv_dim_{} {}
  constexpr Box(Vec3 dims)
      : dim_(dims), inv_dim_(1.0 / dims.x, 1.0 / dims.y, 1.0 / dims.z) {}
  constexpr Box(double l)
      : dim_(l, l, l), inv_dim_(1.0 / l, 1.0 / l, 1.0 / l) {}

  constexpr Vec3 get_dims() const { return dim_; }

  // Wraps arbitrary coordinate back into the fundamental box domain [0, L_i]
  constexpr Vec3 wrap(const Vec3 &v) const {
    return {v.x - dim_.x * std::floor(v.x * inv_dim_.x),
            v.y - dim_.y * std::floor(v.y * inv_dim_.y),
            v.z - dim_.z * std::floor(v.z * inv_dim_.z)};
  }

  // Calculate difference between two vectors based on the minimum image
  // convention
  constexpr Vec3 diff(const Vec3 &v1, const Vec3 &v2) const {
    Vec3 d = v2 - v1;
    return {d.x - dim_.x * std::round(d.x * inv_dim_.x),
            d.y - dim_.y * std::round(d.y * inv_dim_.y),
            d.z - dim_.z * std::round(d.z * inv_dim_.z)};
  }

  // Calculate distance squared between two vectors based on the minimum image
  // convention
  constexpr double dist2(const Vec3 &v1, const Vec3 &v2) const {
    return diff(v1, v2).norm2();
  }
};

} // namespace hpmc
