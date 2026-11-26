#pragma once

#include "core/utils.hpp"
#include "state/state.hpp"

#include <memory>
#include <span>

namespace hpmc {

enum class LatticeType { A, B };

// Signature for the bias hook function pointer

class LSMCEngine {
  std::shared_ptr<State> state_A_;
  std::shared_ptr<State> state_B_;

  State *active_ = nullptr;
  State *passive_ = nullptr;

  LatticeType active_lattice_ = LatticeType::A;

  PRNG rng_;

  double wl_ln_f_ = 1.0;
  std::vector<double> bias_weights_;
  std::vector<uint64_t> visit_counts_;

  int min_M_;
  int max_M_;
  int M_ = 0;

public:
  LSMCEngine(const std::shared_ptr<State> &state_A,
             const std::shared_ptr<State> &state_B, int min_M, int max_M,
             uint64_t seed);

  size_t run_batch(size_t sweeps, double max_disp);

  bool in_range(int M) const { return M >= min_M_ && M <= max_M_; }

  void update_bias(int M);
  double lookup_bias(int M) const;
  void update_visit_count(int M);

  int M() const { return M_; }
  int min_M() const { return min_M_; }
  int max_M() const { return max_M_; }

  void set_wl_ln_f(double wl_ln_f) { wl_ln_f_ = wl_ln_f; }
  double get_wl_ln_f() const { return wl_ln_f_; }

  const std::vector<double> &bias_weights() const { return bias_weights_; }
  void set_bias_weights(std::span<const double> new_bias);

  const std::vector<uint64_t> &visit_counts() const { return visit_counts_; }
  void reset_visit_counts();

  LatticeType active_lattice() const { return active_lattice_; }
  const State &active_state() const { return *active_; }
};

} // namespace hpmc
