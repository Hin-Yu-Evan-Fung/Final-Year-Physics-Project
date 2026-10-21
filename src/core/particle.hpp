#pragma once

#include "core/utils.hpp"
#include <concepts>

namespace hpmc {

// Move definiton
template <typename Particle> struct Move;

// Particle concept
template <typename T>
concept ParticleLike =
    // Needs move function
    requires(T a, const Move<T> &m) {
      { a.move(m) } -> std::same_as<T>;
    };

template <typename T>
concept MoveGeneratable = requires(double max, PRNG &rng) {
  { Move<T>::generate_move(max, rng) } -> std::same_as<Move<T>>;
};

} // namespace hpmc
