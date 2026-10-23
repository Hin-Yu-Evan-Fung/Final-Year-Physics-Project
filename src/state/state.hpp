#pragma once

#include "core/box.hpp"
#include "particle/particle.hpp"
#include "particle/sphere.hpp"
#include "state/cell_list.hpp"

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
  CellList cell_list_;
  std::vector<double> x_{}, y_{}, z_{};
  size_t n_ = 0;

  template <typename Config>
  bool overlaps_naive(const Config &config, size_t idx, const Vec3 &pos) const;
  template <typename Config>
  size_t count_overlaps_naive(const Config &config) const;
  template <typename Config>
  bool overlaps(const Config &config, size_t idx, const Vec3 &pos) const;
  template <typename Config> size_t count_overlaps(const Config &config) const;

public:
  State(ParticleType type, double box_length, ParticleConfig config)
      : type_(type), config_(config), box_(box_length), cell_list_{} {}

  constexpr size_t n() const { return n_; }
  const std::vector<double> &x() const { return x_; }
  const std::vector<double> &y() const { return y_; }
  const std::vector<double> &z() const { return z_; }

  constexpr Vec3 pos_at(size_t idx) const {
    return {x_[idx], y_[idx], z_[idx]};
  }

  void set_coords(std::span<const double> x, std::span<const double> y,
                  std::span<const double> z);

  bool check_overlaps_naive(size_t idx, const Vec3 &pos) const;
  size_t count_overlaps_naive() const;

  bool check_overlaps(size_t idx, const Vec3 &pos) const;
  size_t count_overlaps() const;

  // Tries the translation, if overlap reject
  bool try_translate(size_t idx, const Vec3 &displacement);

  // void translate(size_t idx, const Vec3 &displacement);
  // void accept_move();
  // void reject_move();
};

//=========================================
//             Implementation
//=========================================

template <typename Config>
bool State::overlaps_naive(const Config &config, size_t idx,
                           const Vec3 &pos) const {
  for (size_t i = 0; i < n_; ++i)
    if (i != idx && config.overlap(box_, pos, pos_at(i)))
      return true;
  return false;
}

template <typename Config>
size_t State::count_overlaps_naive(const Config &config) const {
  size_t overlaps = 0;
  for (size_t i = 0; i < n_ - 1; ++i) {
    for (size_t j = i + 1; j < n_; ++j) {
      overlaps += config.overlap(box_, pos_at(i), pos_at(j));
    }
  }
  return overlaps;
}

template <typename Config>
bool State::overlaps(const Config &config, size_t idx, const Vec3 &pos) const {
  int cell = cell_list_.get_cell_index(pos);
  const auto &local_cells = cell_list_.get_neighbours(cell);

  for (int cell_idx : local_cells) {
    int neighbour_id = cell_list_.get_head(cell_idx);

    while (neighbour_id != -1) {
      if (neighbour_id != idx &&
          config.overlap(box_, pos, pos_at(neighbour_id)))
        return true;

      neighbour_id = cell_list_.get_next(neighbour_id);
    }
  }

  return false;
}

template <typename Config>
size_t State::count_overlaps(const Config &config) const {
  size_t overlaps = 0;

  for (int cell_idx = 0; cell_idx < cell_list_.total_cells(); ++cell_idx) {
    int p_id = cell_list_.get_head(cell_idx);
    if (p_id == -1)
      continue;

    const auto &local_cells = cell_list_.get_neighbours(cell_idx);

    while (p_id != -1) {
      // Find all neighbouring particles, which is everyone.
      for (int n_cell : local_cells) {
        int neighbour_id = cell_list_.get_head(n_cell);

        while (neighbour_id != -1) {
          // Protects against self checks and double counting
          if (p_id < neighbour_id) {
            overlaps +=
                config.overlap(box_, pos_at(p_id), pos_at(neighbour_id));
          }
          neighbour_id = cell_list_.get_next(neighbour_id);
        }
      }

      p_id = cell_list_.get_next(p_id);
    }
  }
  return overlaps;
}

} // namespace hpmc
