#pragma once

#include "core/box.hpp"
#include "core/vec3.hpp"

#include <cassert>
#include <vector>

namespace hpmc {

class CellList {
  const Box *box_ = nullptr;
  Vec3 cell_dims_{};
  int x_cells_ = 0, y_cells_ = 0, z_cells_ = 0;
  size_t total_cells_ = 0;

  std::vector<int> head_{};     // Size: total_cells_
  std::vector<int> next_{};     // Size: num_particles
  std::vector<int> cell_map_{}; // Size: num_particles

  std::vector<std::array<int, 27>> neighbours_{};

  // Wraps cell indices based on periodic boundary conditions
  constexpr int grid_pbc_wrap(int d, int n) const {
    return d + ((d < 0) - (d >= n)) * n;
  }

  // Precomputes neighbor indices using the integer wrap logic
  void precompute_neighbours();

public:
  CellList() = default;
  CellList(const Box &box, double min_cell_width, int num_particles)
      : box_(&box) {
    resize(box, min_cell_width, num_particles);
  }

  // Convert to 3d cell coordinates
  constexpr void index_to_3d(int index, int &cx, int &cy, int &cz) const {
    cx = index % x_cells_;
    cy = (index / x_cells_) % y_cells_;
    cz = index / (x_cells_ * y_cells_);
  }

  void resize(const Box &box, double min_cell_width, int num_particles);
  void insert(int particle_idx, int cell_idx);
  void remove(int particle_idx, int cell_idx);

  int get_cell_index(const Vec3 &pos) const;

  constexpr int x_cells() const { return x_cells_; }
  constexpr int y_cells() const { return y_cells_; }
  constexpr int z_cells() const { return z_cells_; }

  int total_cells() const { return total_cells_; }
  int get_head(int cell_idx) const { return head_[cell_idx]; }
  int get_next(int particle_idx) const { return next_[particle_idx]; }
  int get_particle_cell(int particle_idx) const {
    return cell_map_[particle_idx];
  }

  const std::array<int, 27> &get_neighbours(int cell_idx) const {
    return neighbours_[cell_idx];
  }
};

} // namespace hpmc
