#include "lsmc_engine.hpp"
#include "engine_utils.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace hpmc {

LSMCEngine::LSMCEngine(const std::shared_ptr<State> &state_A,
                       const std::shared_ptr<State> &state_B, int min_M,
                       int max_M, uint64_t seed)
    : state_A_(state_A), state_B_(state_B), active_(state_A_.get()),
      passive_(state_B_.get()), rng_(seed), min_M_(min_M), max_M_(max_M) {

  if (!state_A_ || !state_B_)
    throw std::invalid_argument("state_A and state_B must not be null");
  if (state_A_->n() != state_B_->n())
    throw std::invalid_argument(
        "Both states must have the same number of particles");
  if (min_M >= max_M)
    throw std::invalid_argument("min_M must be < max_M");
  if (state_A_->has_any_overlaps())
    throw std::invalid_argument("Initial active state_A must be overlap-free");

  const int initial_overlap_difference =
      static_cast<int>(state_A_->count_all_overlaps()) -
      static_cast<int>(state_B_->count_all_overlaps());
  M_ = initial_overlap_difference;
  if (!in_range(M_))
    throw std::invalid_argument(
        "Initial overlap difference must be within [min_M, max_M]");

  const int range_M = max_M - min_M + 1;
  bias_weights_.assign(static_cast<size_t>(range_M), 0.0);
  visit_counts_.assign(static_cast<size_t>(range_M), 0);
}

void LSMCEngine::update_bias(int M) {
  if (in_range(M))
    bias_weights_[static_cast<size_t>(M - min_M_)] += wl_ln_f_;
}

double LSMCEngine::lookup_bias(int M) const {
  return in_range(M) ? bias_weights_[static_cast<size_t>(M - min_M_)]
                     : std::numeric_limits<double>::max();
}

void LSMCEngine::update_visit_count(int M) {
  if (in_range(M))
    ++visit_counts_[static_cast<size_t>(M - min_M_)];
}

void LSMCEngine::reset_visit_counts() {
  std::fill(visit_counts_.begin(), visit_counts_.end(), 0);
}

void LSMCEngine::set_bias_weights(std::span<const double> new_bias) {
  if (new_bias.size() != bias_weights_.size())
    throw std::invalid_argument(
        "New bias must have the same size as the bias_weights");
  std::copy_n(new_bias.begin(), new_bias.size(), bias_weights_.begin());
}

size_t LSMCEngine::run_batch(size_t sweeps, double max_disp) {
  if (max_disp < 0.0)
    throw std::invalid_argument("max_disp must be non-negative");

  const size_t N = active_->n();
  if (N > 0 && sweeps > std::numeric_limits<size_t>::max() / N)
    throw std::overflow_error("sweeps * particle count exceeds size_t");
  const size_t total_attempts = sweeps * N;

  size_t accepted_moves = 0;
  for (size_t attempt = 0; attempt < total_attempts; ++attempt) {
    size_t p_idx = rng_.random_index(N);

    Vec3 disp = random_disp(rng_, max_disp);

    const PosCell new_a = active_->translated(p_idx, disp);

    // Accept if not overlapping on active lattice
    if (!active_->has_overlaps(new_a.pos, new_a.cell, p_idx)) {
      const PosCell old_p = passive_->pos_cell(p_idx);
      const PosCell new_p = passive_->translated(p_idx, disp);

      size_t overlaps_before =
          passive_->count_overlaps(old_p.pos, old_p.cell, p_idx);
      size_t overlaps_after =
          passive_->count_overlaps(new_p.pos, new_p.cell, p_idx);

      int delta_passive = (int)overlaps_after - (int)overlaps_before;
      int delta_M =
          (active_lattice_ == LatticeType::A) ? -delta_passive : delta_passive;
      int M_before = M_;
      int M_after = M_ + delta_M;

      if (in_range(M_after)) {
        double u_old = lookup_bias(M_before);
        double u_new = lookup_bias(M_after);
        double delta_U = u_new - u_old;

        // Accept move if random double < min(1, exp(-delta_U))
        if (delta_U <= 0.0 || rng_.random_double() < std::exp(-delta_U)) {
          active_->set_pos(p_idx, new_a.pos, new_a.cell);
          passive_->set_pos(p_idx, new_p.pos, new_p.cell);
          M_ = M_after;
          accepted_moves++;
        }
      }
    }

    // Update bias at M
    update_bias(M_);
    update_visit_count(M_);

    // Can switch lattices
    if (M_ == 0 && rng_.random_double() < 0.5) {
      std::swap(active_, passive_);
      active_lattice_ =
          (active_lattice_ == LatticeType::A) ? LatticeType::B : LatticeType::A;
    }
  }

  return accepted_moves;
}

} // namespace hpmc
