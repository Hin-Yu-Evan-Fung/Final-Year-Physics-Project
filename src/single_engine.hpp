#pragma once

#include "core/utils.hpp"
#include "state/state.hpp"

namespace hpmc {

class SingleBoxEngine {
  State state_;

  PRNG rng_;

public:
  SingleBoxEngine(ParticleType type, double box_length, ParticleConfig config,
                  uint64_t seed)
      : state_(type, box_length, config), rng_(seed) {}

  State &state() { return state_; }

  Vec3 random_disp(double max_disp);
  size_t run_batch(size_t sweeps, double max_disp);
};

} // namespace hpmc
