#pragma once

#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace hsmc {

#ifndef NDEBUG
#define ENGINE_ASSERT(condition, exception_type, message)                      \
  if (!(condition)) {                                                          \
    throw exception_type(message);                                             \
  }
#else
#define ENGINE_ASSERT(condition, exception_type, message)
#endif

template <typename T>
concept HasDof =
    // Condition 1: Must have a static compile-time 'dof' variable
    requires {
      { T::dof } -> std::convertible_to<ssize_t>;
    } &&
    // Condition 2: The dof must be physically valid (greater than 0)
    (T::dof > 0);

template <HasDof T>
using View =
    nb::ndarray<double, nb::shape<-1, T::dof>, nb::c_contig, nb::device::cpu>;

using Indices =
    nb::ndarray<uint32_t, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

} // namespace hsmc
