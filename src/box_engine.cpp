#include "box_engine.hpp"
#include "engine_utils.hpp"

namespace hpmc {

size_t BoxEngine::run_batch(size_t sweeps, double max_disp) {
  if (max_disp < 0.0)
    throw std::invalid_argument("max_disp must be non-negative");

  const size_t N = state_->n();

  size_t accepted = 0;
  for (size_t sweep = 0; sweep < sweeps; ++sweep) {
    for (size_t i = 0; i < N; ++i) {
      size_t p_idx = rng_.random_index(N);

      Vec3 disp = random_disp(rng_, max_disp);

      const PosCell proposed = state_->translated(p_idx, disp);

      // TODO: Add rotate moves and bias function
      if (!state_->has_overlaps(proposed.pos, proposed.cell, p_idx)) {
        state_->set_pos(p_idx, proposed.pos, proposed.cell);
        accepted++;
      }
    }
  }

  return accepted;
}

} // namespace hpmc
