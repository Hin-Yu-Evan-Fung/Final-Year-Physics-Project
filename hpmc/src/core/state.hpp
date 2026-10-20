#pragma once

#include "box.hpp"
#include "particle.hpp"

#include <vector>

namespace hpmc {

//==========================
//          State
//==========================

template <ParticleLike Particle, typename ParamType> class State {
  Box box_;
  std::vector<Particle> particles_;
  ParamType params_;

  bool check_overlap(unsigned idx, const Particle &particle) const {
    for (size_t i = 0; i < particles_.size(); ++i)
      if (i != idx && has_overlap(box_, particles_[i], particle, params_))
        return true;
    return false;
  }

  static size_t row_bytes();

public:
  State(double L, ParamType params) : box_(L), params_(params) {}

  constexpr size_t size() const { return particles_.size(); }

  void set_params(ParamType p) { params_ = p; }
  const ParamType &get_params() const { return params_; }

  void move(unsigned idx, const Move<Particle> &m);
  bool try_move(unsigned idx, const Move<Particle> &m);

  // API functions

  const std::vector<Particle> &get_particles() const { return particles_; }
  void set_particles(std::span<const Particle> particles);

  size_t count_overlaps() const;

  size_t try_run_batch(std::span<const uint32_t> indices,
                       std::span<const Move<Particle>> moves);
  void run_batch(std::span<const uint32_t> indices,
                 std::span<const Move<Particle>> moves);
};

} // namespace hpmc
