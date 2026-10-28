#pragma once

#include "core/utils.hpp"
#include "state/state.hpp"
#include <limits>
#include <memory>
#include <stdexcept>
#include <algorithm>

namespace hpmc {

enum class Lattice { A, B };

// Signature for the bias hook function pointer

class LSMCEngine {
  std::shared_ptr<State> state_A_;
  std::shared_ptr<State> state_B_;

  State *active_ = nullptr;
  State *passive_ = nullptr;

  Lattice active_lattice_ = Lattice::A;

  PRNG rng_;

  double wl_ln_f_ = 1.0;
  std::vector<double> bias_weights_;
  std::vector<int> visit_counts_;

  int min_M_;
  int max_M_;
  int M_ = 0;

public:
  LSMCEngine(const std::shared_ptr<State> &state_A,
             const std::shared_ptr<State> &state_B, int min_M, int max_M,
             uint64_t seed)
      : state_A_(state_A), state_B_(state_B), active_(state_A_.get()),
        passive_(state_B_.get()), rng_(seed), min_M_(min_M), max_M_(max_M) {
    if (state_A_->n() != state_B_->n())
      throw std::runtime_error(
          "Both states should have the same number of particles");
    if (min_M >= max_M)
      throw std::runtime_error("min_M has to be < max_M");

    const int range_M = max_M - min_M + 1;
    bias_weights_.assign(range_M, 0.0);
    visit_counts_.assign(range_M, 0);
  }

  void set_wl_ln_f(double wl_ln_f) { wl_ln_f_ = wl_ln_f; }
  double get_wl_ln_f() const { return wl_ln_f_; }

  Vec3 random_disp(double max_disp);
  size_t run_batch(size_t sweeps, double max_disp);

  bool in_range(int M) { return M >= min_M_ && M <= max_M_; }

  void update_bias(int M) {
    if (in_range(M))
      bias_weights_[M - min_M_] += wl_ln_f_;
  }

  double lookup_bias(int M) {
    return in_range(M) ? bias_weights_[M - min_M_]
                       : std::numeric_limits<double>::max();
  }

  void update_visit_count(int M) {
    if (in_range(M))
      ++visit_counts_[M - min_M_];
  }

  void reset_visit_counts() {
    std::fill(visit_counts_.begin(), visit_counts_.end(), 0);
  }

  void set_bias_weights(std::span<double> new_bias) {
    if (new_bias.size() != bias_weights_.size())
      throw std::runtime_error(
          "New bias must have the same size as the bias_weights");
    std::copy_n(new_bias.begin(), new_bias.size(), bias_weights_.begin());
  }

  int M() const { return M_; }
  std::vector<double> &bias_weights() { return bias_weights_; }
  std::vector<int> &visit_counts() { return visit_counts_; }

  Lattice active_lattice() const { return active_lattice_; }
  const State &active_state() const { return *active_; }
};

} // namespace hpmc
