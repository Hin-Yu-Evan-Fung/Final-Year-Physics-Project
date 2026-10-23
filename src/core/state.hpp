#pragma once

#include "box.hpp"
#include "particle.hpp"
#include "particles/hard_sphere.hpp"

#include <span>
#include <variant>
#include <vector>

namespace hpmc {

using ParticleConfig = std::variant<SphereConfig>;

class State {
  // Config
  ParticleType type_;
  ParticleConfig config_;

  // Internal state
  Box box_;
  std::vector<double> x_{}, y_{}, z_{};
  size_t n_ = 0;

  // Accept reject cache
  int curr_idx_ = -1;
  Vec3 old_pos_ = {};

  template <typename Config>
  bool overlaps_naive(const Config &config, size_t idx, const Vec3 &pos) const {
    for (size_t i = 0; i < n_; ++i)
      if (i != idx && config.overlap_naive(box_, pos, pos_at(i)))
        return true;
    return false;
  }

  template <typename Config>
  size_t count_overlaps_naive(const Config &config) const {
    size_t overlaps = 0;
    for (size_t i = 0; i < n_ - 1; ++i) {
      for (size_t j = i + 1; j < n_; ++j) {
        overlaps += config.overlap_naive(box_, pos_at(i), pos_at(j));
      }
    }
    return overlaps;
  }

public:
  State(ParticleType type, double box_length, ParticleConfig config)
      : type_(type), config_(config), box_(box_length) {}

  constexpr size_t n() const { return n_; }
  const std::vector<double> &x() const { return x_; }
  const std::vector<double> &y() const { return y_; }
  const std::vector<double> &z() const { return z_; }

  constexpr Vec3 pos_at(size_t idx) const {
    return {x_[idx], y_[idx], z_[idx]};
  }

  void set_coords(std::span<const double> x, std::span<const double> y,
                  std::span<const double> z);

  bool check_overlaps_naive(size_t idx) const;
  size_t count_overlaps_naive() const;

  void translate(size_t idx, const Vec3 &displacement);
  void reject_move();
};

} // namespace hpmc
