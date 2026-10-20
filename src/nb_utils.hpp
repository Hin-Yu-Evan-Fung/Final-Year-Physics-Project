#pragma once

#include <nanobind/ndarray.h>
#include <span>

namespace nb = nanobind;

namespace hsmc {

template <typename T>
concept HasDof =
    // Condition 1: Must have a static compile-time 'dof' variable
    requires {
      { T::dof } -> std::convertible_to<ssize_t>;
    } &&
    // Condition 2: The dof must be physically valid (greater than 0)
    (T::dof > 0);

template <HasDof T>
using View = nb::ndarray<nb::numpy, double, nb::shape<-1, T::dof>, nb::c_contig,
                         nb::device::cpu>;

using Indices = nb::ndarray<nb::numpy, uint32_t, nb::shape<-1>, nb::c_contig,
                            nb::device::cpu>;

template <HasDof T> constexpr auto view_to_span(View<T> view) {
  auto *raw_ptr = reinterpret_cast<const T *>(view.data());
  return std::span<const T>(raw_ptr, view.shape(0));
}

constexpr auto indices_to_span(Indices indices) {
  return std::span<const uint32_t>(indices.data(), indices.shape(0));
}

} // namespace hsmc
