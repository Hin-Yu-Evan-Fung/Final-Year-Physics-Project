
#include "core/lattice.hpp"
#include "core/vec3.hpp"

namespace hpmc {

void Lattice::validate_lattice(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  if (a.y != 0.0 || a.z != 0.0) {
    throw std::invalid_argument(
        std::format("Lattice vector 'a' must lie along the x-axis (a.y == 0, "
                    "a.z == 0), got ({:.6g}, {:.6g}, {:.6g})",
                    a.x, a.y, a.z));
  }
  if (b.z != 0.0) {
    throw std::invalid_argument(
        std::format("Lattice vector 'b' must lie in the xy-plane (b.z == 0), "
                    "got ({:.6g}, {:.6g}, {:.6g})",
                    b.x, b.y, b.z));
  }
  if (a.x <= 0.0 || b.y <= 0.0 || c.z <= 0.0) {
    throw std::invalid_argument(std::format(
        "Lattice diagonal entries (a.x, b.y, c.z) must be strictly positive "
        "and right-handed, got ({:.6g}, {:.6g}, {:.6g})",
        a.x, b.y, c.z));
  }
}

Lattice Lattice::from_upper_triangular(const Vec3 &a, const Vec3 &b,
                                       const Vec3 &c) {
  validate_lattice(a, b, c);
  return Lattice{
      .a = a,
      .b = b,
      .c = c,
      .inv_ax = 1.0 / a.x,
      .inv_by = 1.0 / b.y,
      .inv_cz = 1.0 / c.z,
      .volume = a.x * b.y * c.z,
  };
}

Lattice Lattice::from_box_lengths(double lx, double ly, double lz) {
  return from_upper_triangular({lx, 0.0, 0.0}, {0.0, ly, 0.0}, {0.0, 0.0, lz});
}

Lattice Lattice::from_cubic(double l) { return from_box_lengths(l, l, l); }

} // namespace hpmc
