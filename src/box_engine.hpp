#pragma once

#include "core/utils.hpp"
#include "state/state.hpp"

#include <memory>

namespace hpmc {

class BoxEngine {
  std::shared_ptr<State> state_;
  PRNG rng_;

public:
  BoxEngine(const std::shared_ptr<State> &state, uint64_t seed)
      : state_(state), rng_(seed) {}

  size_t run_batch(size_t sweeps, double max_disp);
};

} // namespace hpmc
