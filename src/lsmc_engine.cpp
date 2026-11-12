#include "lsmc_engine.hpp"
#include "state/state.hpp"

namespace hpmc {

Vec3 LSMCEngine::random_disp(double max_disp) {
  const double dx = rng_.random_double(-max_disp, max_disp);
  const double dy = rng_.random_double(-max_disp, max_disp);
  const double dz = rng_.random_double(-max_disp, max_disp);
  return Vec3{dx, dy, dz};
}

size_t LSMCEngine::run_batch(size_t sweeps, double max_disp) {
  const size_t N = active_->n();
  const size_t total_attempts = sweeps * N;

  size_t accepted_moves = 0;
  for (size_t attempt = 0; attempt < total_attempts; ++attempt) {
    size_t p_idx = rng_.random_index(N);

    Vec3 disp = random_disp(max_disp);

    MoveProposal active_prop = active_->propse_translation(p_idx, disp);
    MoveProposal passive_prop = passive_->propse_translation(p_idx, disp);

    // Accept if not overlapping on active lattice
    if (!active_->check_overlap(active_prop)) {
      size_t overlaps_before = passive_->count_local_overlaps(
          p_idx, passive_prop.old_pos, (size_t)passive_prop.old_cell);
      size_t overlaps_after = passive_->count_local_overlaps(
          p_idx, passive_prop.new_pos, (size_t)passive_prop.new_cell);

      int delta_passive = (int)overlaps_after - (int)overlaps_before;
      int delta_M =
          (active_lattice_ == Lattice::A) ? -delta_passive : delta_passive;
      int M_before = M_;
      int M_after = M_ + delta_M;

      // Record the unbiased transition before applying the sampling bias
      if (in_range(M_after)) {
        transition_matrix_.record(M_before,M_after,1);
      } else {
        // If attempted to go out of the range of M then update diagonal and leave off diagonal the same
        transition_matrix_.record(M_before,M_before,0.0);
      }

      double u_old = lookup_bias(M_before);
      double u_new = lookup_bias(M_after);
      double delta_U = u_new - u_old;

      // Accept move if random double < min(1, exp(-delta_U))
      if (delta_U <= 0.0 || rng_.random_double() < std::exp(-delta_U)) {
        active_->accept_translation(active_prop);
        passive_->accept_translation(passive_prop);
        M_ = M_after;
        accepted_moves++;
      }
    } else {
      // A physical overlap means unbiased acceptance is zero.
      transition_matrix_.record(M_,M_,0.0);
    }

    // Update bias at M
    update_bias(M_);
    update_visit_count(M_);

    // Can switch lattices
    if (M_ == 0 && rng_.random_double() < 0.5) {
      std::swap(active_, passive_);
      active_lattice_ =
          (active_lattice_ == Lattice::A) ? Lattice::B : Lattice::A;
    }
  }

  return accepted_moves;
}

} // namespace hpmc
