
#include "single_engine.hpp"
#include "state/state.hpp"
namespace hpmc {

Vec3 SingleBoxEngine::random_disp(double max_disp) {
  const double dx = rng_.random_double(-max_disp, max_disp);
  const double dy = rng_.random_double(-max_disp, max_disp);
  const double dz = rng_.random_double(-max_disp, max_disp);
  return Vec3{dx, dy, dz};
}

size_t SingleBoxEngine::run_batch(size_t sweeps, double max_disp) {
  const size_t N = state_->n();

  size_t accepted = 0;
  for (size_t sweep = 0; sweep < sweeps; ++sweep) {
    for (size_t i = 0; i < N; ++i) {
      size_t p_idx = rng_.random_index(N);

      const MoveProposal &prop =
          state_->propse_translation(p_idx, random_disp(max_disp));

      // TODO: Add rotate moves and bias function
      if (!state_->check_overlap(prop)) {
        state_->accept_translation(prop);
        accepted++;
      }
    }
  }

  return accepted;
}

} // namespace hpmc
