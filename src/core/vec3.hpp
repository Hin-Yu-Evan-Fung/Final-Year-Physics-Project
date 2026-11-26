#pragma once

#include <cmath>
#include <format>
#include <string>

namespace hpmc {

struct Vec3 {
  double x, y, z;

  constexpr Vec3() : x(0.0), y(0.0), z(0.0) {}

  constexpr Vec3(double x_val, double y_val, double z_val)
      : x(x_val), y(y_val), z(z_val) {}

  constexpr Vec3 operator+(const Vec3 &o) const {
    return {x + o.x, y + o.y, z + o.z};
  }
  constexpr Vec3 operator-(const Vec3 &o) const {
    return {x - o.x, y - o.y, z - o.z};
  }
  constexpr Vec3 operator*(double s) const noexcept {
    return {x * s, y * s, z * s};
  }
  constexpr Vec3 operator/(double s) const noexcept {
    return {x / s, y / s, z / s};
  }

  constexpr Vec3 operator+=(const Vec3 &o) noexcept {
    return *this = *this + o;
  }
  constexpr Vec3 operator-=(const Vec3 &o) noexcept {
    return *this = *this - o;
  }
  constexpr Vec3 operator*=(double s) noexcept { return *this = *this * s; }
  constexpr Vec3 operator/=(double s) noexcept { return *this = *this / s; }

  constexpr double dot(const Vec3 &o) const noexcept {
    return x * o.x + y * o.y + z * o.z;
  }

  constexpr Vec3 cross(const Vec3 &o) const noexcept {
    return {
        y * o.z - z * o.y,
        z * o.x - x * o.z,
        x * o.y - y * o.x,
    };
  }

  constexpr double norm2() const { return dot(*this); }
  constexpr double norm() const { return std::sqrt(norm2()); }

  // Truncated due to formatting
  std::string to_str() const {
    return std::format("Vec3[{:.4f},{:.4f},{:.4f}]", x, y, z);
  }
};

inline constexpr Vec3 operator*(double s, const Vec3 &v) { return v * s; }

} // namespace hpmc
