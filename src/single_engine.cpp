
#include "single_engine.hpp"
namespace hpmc {

Vec3 SingleBoxEngine::random_disp(double max_disp) {
  const double dx = (rng_.random_double() - 0.5) * 2.0 * max_disp;
  const double dy = (rng_.random_double() - 0.5) * 2.0 * max_disp;
  const double dz = (rng_.random_double() - 0.5) * 2.0 * max_disp;
  return {dx, dy, dz};
}

size_t SingleBoxEngine::run_batch(size_t sweeps, double max_disp) {
  const size_t N = state_.n();

  size_t accepted = 0;
  for (size_t sweep = 0; sweep < sweeps; ++sweep) {
    for (size_t i = 0; i < N; ++i) {
      size_t target_idx = rng_.random_index(N);

      // TODO: Add rotate moves and bias function
      accepted += state_.try_translate(target_idx, random_disp(max_disp));
    }
  }

  return accepted;
}

} // namespace hpmc
