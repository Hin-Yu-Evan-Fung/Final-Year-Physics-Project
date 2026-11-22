#pragma once

#include "core/vec3.hpp"
#include <nanobind/ndarray.h>
#include <span>
#include <string>

namespace nb = nanobind;

namespace hpmc {

template <typename T> //
auto as_view(std::span<T> data) {
  size_t shape[1] = {data.size()};
  return nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig>(
      data.data(), 1, shape, nb::handle());
}

inline std::array<double, 3> vec3_to_arr(const Vec3 &v) {
  return {v.x, v.y, v.z};
}

inline Vec3 arr_to_vec3(const std::array<double, 3> &a) {
  return {a[0], a[1], a[2]};
}

} // namespace hpmc
