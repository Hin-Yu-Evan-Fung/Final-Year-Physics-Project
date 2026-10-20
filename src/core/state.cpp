#include "state.hpp"
#include "particle.hpp"
#include "particles/hard_sphere.hpp"
#include "utils.hpp"
#include <cstring>

namespace hpmc {

// ====================================================================
//                           Implementation
// ====================================================================

template <ParticleLike Particle, typename Params>
void State<Particle, Params>::set_particles(std::span<const Particle> p) {
  particles_.clear();
  particles_.resize(p.size());
  std::memcpy(particles_.data(), p.data(), p.size_bytes());
}

template <ParticleLike Particle, typename Params>
void State<Particle, Params>::move(unsigned idx, const Move<Particle> &m) {
  ENGINE_ASSERT(idx < particles_.size(), std::out_of_range,
                "Particle index out of bounds");

  particles_[idx] = particles_[idx].move(m);
  particles_[idx].pos = box_.wrap(particles_[idx].pos);
}

template <ParticleLike Particle, typename Params>
bool State<Particle, Params>::try_move(unsigned idx, const Move<Particle> &m) {
  ENGINE_ASSERT(idx < particles_.size(), std::out_of_range,
                "Particle index out of bounds");

  Particle trial = particles_[idx].move(m);
  trial.pos = box_.wrap(trial.pos);
  return !check_overlap(idx, trial) ? (particles_[idx] = trial, true) : false;
}

template <ParticleLike Particle, typename Params>
size_t State<Particle, Params>::row_bytes() {
  constexpr size_t row_bytes = Move<HardSphere>::dof * sizeof(double);

  // Ensure your struct size exactly matches the expected row byte width
  static_assert(
      sizeof(Move<HardSphere>) == row_bytes,
      "Struct size mismatch! Check for compiler padding alignment bugs.");

  return row_bytes;
}

template <ParticleLike Particle, typename Params>
size_t State<Particle, Params>::count_overlaps() const {
  size_t overlaps = 0;
  for (size_t i = 0; i < particles_.size(); ++i)
    for (size_t j = i + 1; j < particles_.size(); ++j)
      overlaps += has_overlap(box_, particles_[i], particles_[j], params_);
  return overlaps;
}

template <ParticleLike Particle, typename Params>
size_t
State<Particle, Params>::try_run_batch(std::span<const uint32_t> indices,
                                       std::span<const Move<Particle>> moves) {
  ENGINE_ASSERT(indices.size() == moves.size(), std::invalid_argument,
                "Batch size mismatch");

  size_t accepted = 0;
  size_t steps = indices.size();

  const uint32_t *idx_ptr = indices.data();

  for (size_t i = 0; i < steps; ++i) {
    if (try_move(idx_ptr[i], moves[i]))
      accepted++;
  }

  return accepted;
}

template <ParticleLike Particle, typename Params>
void State<Particle, Params>::run_batch(std::span<const uint32_t> indices,
                                        std::span<const Move<Particle>> moves) {
  ENGINE_ASSERT(indices.size() == moves.size(), std::invalid_argument,
                "Batch size mismatch");

  size_t steps = indices.size();
  const uint32_t *idx_ptr = indices.data();

  for (size_t i = 0; i < steps; ++i) {
    move(idx_ptr[i], moves[i]);
  }
}

template class State<HardSphere, HardSphereParams>;

}; // namespace hpmc
