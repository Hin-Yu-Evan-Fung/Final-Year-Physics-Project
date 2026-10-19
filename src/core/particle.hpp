#pragma once

#include <concepts>

namespace hsmc {

// Move definiton
template <typename Particle> struct Move;

// Particle concept
template <typename T>
concept ParticleLike =
    // Needs move function
    requires(T a, const Move<T> &m) {
      { a.move(m) } -> std::same_as<T>;
    };

} // namespace hsmc
