#pragma once

namespace hpmc {

#ifndef NDEBUG
#define ENGINE_ASSERT(condition, exception_type, message)                      \
  if (!(condition)) {                                                          \
    throw exception_type(message);                                             \
  }
#else
#define ENGINE_ASSERT(condition, exception_type, message)
#endif

} // namespace hpmc
