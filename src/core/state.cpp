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
  particles_[idx] = particles_[idx].move(m);
  particles_[idx].pos = box_.wrap(particles_[idx].pos);
}

template <ParticleLike Particle, typename Params>
bool State<Particle, Params>::try_move(unsigned idx, const Move<Particle> &m) {
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
size_t State<Particle, Params>::run_batch(size_t sweeps, double max_disp,
                                          uint64_t seed) {
  PRNG rng(seed);
  const size_t N = size();

  size_t accepted = 0;
  for (size_t sweep = 0; sweep < sweeps; ++sweep) {
    for (size_t i = 0; i < N; ++i) {
      size_t target_idx = rng.random_index(N);

      if (try_move(target_idx, Move<Particle>::generate_move(max_disp, rng)))
        accepted++;
    }
  }

  return accepted;
}

template class State<HardSphere, HardSphereParams>;

}; // namespace hpmc
