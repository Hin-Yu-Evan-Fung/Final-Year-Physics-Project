#pragma once

#include "core/vec3.hpp"

#include <cmath>

namespace hpmc {

struct Lattice {
  Vec3 a, b, c;
  double inv_ax, inv_by, inv_cz, volume;

  static Lattice from_upper_triangular(const Vec3 &a, const Vec3 &b,
                                       const Vec3 &c);
  static Lattice from_box_lengths(double lx, double ly, double lz);
  static Lattice from_cubic(double l);

  // r = Hs
  constexpr Vec3 to_cart(const Vec3 &s) const {
    double cx = a.x * s.x + b.x * s.y + c.x * s.z;
    double cy = b.y * s.y + c.y * s.z;
    double cz = c.z * s.z;
    return {cx, cy, cz};
  }

  // s = H^-1 r
  constexpr Vec3 to_frac(const Vec3 &r) const {
    double sz = r.z * inv_cz;
    double sy = (r.y - c.y * sz) * inv_by;
    double sx = (r.x - b.x * sy - c.x * sz) * inv_ax;
    return {sx, sy, sz};
  }

  // PBC wrapping
  constexpr Vec3 wrap(const Vec3 &r) const {
    Vec3 s = to_frac(r);
    s.x -= std::floor(s.x);
    s.y -= std::floor(s.y);
    s.z -= std::floor(s.z);
    return to_cart(s);
  }

  // Minimum image convention
  constexpr Vec3 min_img(const Vec3 &dr) const {
    Vec3 ds = to_frac(dr);
    ds.x -= std::round(ds.x);
    ds.y -= std::round(ds.y);
    ds.z -= std::round(ds.z);
    return to_cart(ds);
  }

  constexpr Vec3 perp_widths() const {
    double wx = volume / (b.cross(c).norm());
    double wy = volume / (a.cross(c).norm());
    double wz = volume / (a.cross(b).norm());
    return {wx, wy, wz};
  }

  constexpr double dist2_ortho(const Vec3 &dr) const {
    double sx = dr.x * inv_ax;
    double sy = dr.y * inv_by;
    double sz = dr.z * inv_cz;
    sx -= std::round(sx);
    sy -= std::round(sy);
    sz -= std::round(sz);

    const double dx = a.x * sx;
    const double dy = b.y * sy;
    const double dz = c.z * sz;
    return dx * dx + dy * dy + dz * dz;
  }

  constexpr double dist2(const Vec3 &v1, const Vec3 &v2) const {
    const Vec3 dr = v2 - v1;
    return is_orthorhombic() ? dist2_ortho(dr) : min_img(dr).norm2();
  }

private:
  constexpr bool is_orthorhombic() const {
    return b.x == 0.0 && c.x == 0.0 && c.y == 0.0;
  }

  static void validate_lattice(const Vec3 &a, const Vec3 &b, const Vec3 &c);
};

} // namespace hpmc
