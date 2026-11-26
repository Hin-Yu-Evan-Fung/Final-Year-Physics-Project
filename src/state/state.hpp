#pragma once

#include "core/lattice.hpp"
#include "state/cell_list.hpp"
#include "state/shape.hpp"

#include <cmath>
#include <span>
#include <variant>
#include <vector>

namespace hpmc {

using Config = std::variant<SphereConfig>;

struct PosCell {
  Vec3 pos;
  int cell;
};

class State {
  Config config_;

  Lattice lattice_;
  CellList cell_list_;

  std::vector<double> x_{}, y_{}, z_{};
  size_t n_ = 0;

  template <typename Func>
  void for_each_neighbour(size_t cell, Func &&func) const;
  template <bool Break, ShapeConfig Config>
  size_t local_overlaps(const Config &config, const Vec3 &pos, size_t cell,
                        int excl_idx) const;
  template <bool Break, ShapeConfig Config>
  size_t global_overlaps(const Config &config) const;

public:
  State(Config config, Lattice lattice)
      : config_(std::move(config)), lattice_(std::move(lattice)), cell_list_{} {
  }

  void set_coords(std::span<const double> x, std::span<const double> y,
                  std::span<const double> z);

  constexpr size_t n() const { return n_; }
  const Lattice &lattice() const { return lattice_; }
  const std::vector<double> &x() const { return x_; }
  const std::vector<double> &y() const { return y_; }
  const std::vector<double> &z() const { return z_; }

  int cell_of(int p_idx) const { return cell_list_.get_particle_cell(p_idx); }
  int cell_of(const Vec3 &pos) const {
    return cell_list_.get_cell_index(lattice_, pos);
  }

  constexpr Vec3 pos_at(size_t idx) const {
    return {x_[idx], y_[idx], z_[idx]};
  }

  PosCell pos_cell(int idx) const { return {pos_at(idx), cell_of(idx)}; }

  constexpr Vec3 displaced_pos(int idx, const Vec3 &delta) const noexcept {
    return lattice_.wrap(pos_at(idx) + delta);
  }
  PosCell translated(int idx, const Vec3 &delta) const {
    Vec3 s = lattice_.to_frac(pos_at(idx) + delta);
    s.x -= std::floor(s.x);
    s.y -= std::floor(s.y);
    s.z -= std::floor(s.z);

    return {lattice_.to_cart(s), cell_list_.get_cell_index_from_fractional(s)};
  }

  void set_pos(int idx, const Vec3 &new_pos, int new_cell) {
    assert(idx >= 0 && idx < n_);
    x_[idx] = new_pos.x;
    y_[idx] = new_pos.y;
    z_[idx] = new_pos.z;

    const int old_cell = cell_list_.get_particle_cell(idx);
    if (old_cell != new_cell) {
      cell_list_.remove(idx, old_cell);
      cell_list_.insert(idx, new_cell);
    }
  }

  bool has_overlaps(const Vec3 &pos, size_t cell, int excl_idx = -1) const;
  size_t count_overlaps(const Vec3 &pos, size_t cell, int excl_idx = -1) const;
  bool has_any_overlaps() const;
  size_t count_all_overlaps() const;
};

//=========================================
//             Implementation
//=========================================

enum class LoopControl : bool { Continue = false, Break = true };

template <typename Func>
void State::for_each_neighbour(size_t cell, Func &&fn) const {
  for (int cell_idx : cell_list_.get_neighbours(cell)) {
    for (int nid = cell_list_.get_head(cell_idx); nid != -1;
         nid = cell_list_.get_next(nid)) {
      if (fn(nid) == LoopControl::Break)
        return;
    }
  }
}

template <bool Break, ShapeConfig Config>
size_t State::local_overlaps(const Config &config, const Vec3 &pos, size_t cell,
                             int excl_idx) const {
  size_t overlaps = 0;

  for_each_neighbour(cell, [&](int uid) {
    if (uid != excl_idx && config.overlap(lattice_, pos, pos_at(uid))) {
      ++overlaps;
      if constexpr (Break)
        return LoopControl::Break;
    }
    return LoopControl::Continue;
  });

  return overlaps;
}

template <bool Break, ShapeConfig Config>
size_t State::global_overlaps(const Config &config) const {
  size_t overlaps = 0;

  for (int cell_idx = 0; cell_idx < cell_list_.total_cells(); ++cell_idx) {
    for (int p_id = cell_list_.get_head(cell_idx); p_id != -1;
         p_id = cell_list_.get_next(p_id)) {
      const Vec3 p_pos = pos_at(p_id);

      for_each_neighbour(cell_idx, [&](int n_id) {
        if (p_id < n_id && config.overlap(lattice_, p_pos, pos_at(n_id))) {
          ++overlaps;
          if constexpr (Break) {
            return LoopControl::Break;
          }
        }
        return LoopControl::Continue;
      });

      if constexpr (Break)
        if (overlaps > 0)
          return 1;
    }
  }

  return overlaps;
}

} // namespace hpmc
