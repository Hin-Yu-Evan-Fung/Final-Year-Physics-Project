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

struct MoveProposal {
  size_t idx;
  Vec3 old_pos, new_pos;
  int old_cell, new_cell;
};

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
  bool check_overlap(const Config &config, const MoveProposal &prop) const;
  template <typename Config>
  size_t count_local_overlap(const Config &config, size_t idx) const;
  template <typename Config>
  size_t count_global_overlaps(const Config &config) const;

public:
  State(ParticleType type, double box_length, ParticleConfig config)
      : type_(type), config_(config), box_(box_length), cell_list_{} {}

  constexpr size_t n() const { return n_; }
  const Box &box() const { return box_; }
  const std::vector<double> &x() const { return x_; }
  const std::vector<double> &y() const { return y_; }
  const std::vector<double> &z() const { return z_; }

  constexpr Vec3 pos_at(size_t idx) const {
    return {x_[idx], y_[idx], z_[idx]};
  }

  void set_coords(std::span<const double> x, std::span<const double> y,
                  std::span<const double> z);

  bool check_overlap(const MoveProposal &prop) const;
  size_t count_local_overlaps(size_t idx) const;
  size_t count_global_overlaps() const;

  // WARNING: Cell list is temporarily invalid before accept/reject
  MoveProposal propse_translation(size_t idx, const Vec3 &displacement);
  void accept_move(const MoveProposal &prop);
  void reject_move(const MoveProposal &prop);
};

//=========================================
//             Implementation
//=========================================

template <typename Config>
bool State::check_overlap(const Config &config,
                          const MoveProposal &prop) const {
  const auto &local_cells = cell_list_.get_neighbours(prop.new_cell);

  for (int cell_idx : local_cells) {
    int neighbour_id = cell_list_.get_head(cell_idx);

    while (neighbour_id != -1) {
      if (neighbour_id != prop.idx &&
          config.overlap(box_, prop.new_pos, pos_at(neighbour_id)))
        return true;

      neighbour_id = cell_list_.get_next(neighbour_id);
    }
  }

  return false;
}

template <typename Config>
size_t State::count_local_overlap(const Config &config, size_t idx) const {
  const int cell = cell_list_.get_particle_cell(idx);
  const Vec3 &pos = pos_at(idx);
  const auto &local_cells = cell_list_.get_neighbours(cell);
  size_t overlaps = 0;

  for (int cell_idx : local_cells) {
    int neighbour_id = cell_list_.get_head(cell_idx);

    while (neighbour_id != -1) {
      overlaps += neighbour_id != idx &&
                  config.overlap(box_, pos, pos_at(neighbour_id));

      neighbour_id = cell_list_.get_next(neighbour_id);
    }
  }

  return overlaps;
}

template <typename Config>
size_t State::count_global_overlaps(const Config &config) const {
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
