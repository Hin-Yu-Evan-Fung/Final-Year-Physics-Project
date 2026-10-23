#pragma once

#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace hpmc {

template <typename T>
inline T get_param(const nb::kwargs &kwargs, const std::string &key,
                   bool required) {
  if (!kwargs.contains(key.c_str())) {
    // Throws a clean ValueError back to Python instantly
    throw nb::value_error(
        ("Missing required configuration parameter: '" + key + "'").c_str());
  }

  try {
    return nb::cast<T>(kwargs[key.c_str()]);
  } catch (const nb::cast_error &e) {
    throw nb::type_error(
        ("Parameter '" + key + "' must be a numeric type (float or int).")
            .c_str());
  }
}

} // namespace hpmc
