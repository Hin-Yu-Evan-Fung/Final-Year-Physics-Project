#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

namespace hpmc {

class PRNG {
  uint64_t s[4];

public:
  PRNG() : PRNG(0x6B51FF299F6A3AEEULL) {}
  PRNG(uint64_t seed) {
    if (seed == 0)
      seed = 0x123456789ABCDEFULL;

    auto splitmix64 = [&seed]() -> uint64_t {
      seed += 0x9E3779B97F4A7C15ULL;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0XBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0X94D049BB133111EBULL;
      return z ^ (z >> 31);
    };
    s[0] = splitmix64();
    s[1] = splitmix64();
    s[2] = splitmix64();
    s[3] = splitmix64();
  }

  inline uint64_t random_u64() {
    const uint64_t result = std::rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = std::rotl(s[3], 45);

    return result;
  }

  // Returns random number from [0, N-1]
  inline size_t random_index(size_t N) {
    uint32_t x = (uint32_t)random_u64();
    return ((uint64_t)x * (uint64_t)N) >> 32;
  }

  inline double random_double() {
    constexpr double MULTIPLIER = 1.0 / (1ULL << 53);
    return (random_u64() >> 11) * MULTIPLIER;
  }
};

} // namespace hpmc
