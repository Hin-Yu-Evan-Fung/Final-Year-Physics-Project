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
  void precompute_neighbours() {
    neighbours_.resize(total_cells_);
    for (int i = 0; i < total_cells_; ++i) {
      int cx = i % x_cells_;
      int cy = (i / x_cells_) % y_cells_;
      int cz = i / (x_cells_ * y_cells_);

      int n_count = 0;
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            int nx = grid_pbc_wrap(cx + dx, x_cells_);
            int ny = grid_pbc_wrap(cy + dy, y_cells_);
            int nz = grid_pbc_wrap(cz + dz, z_cells_);
            neighbours_[i][n_count++] =
                nx + ny * x_cells_ + nz * x_cells_ * y_cells_;
          }
        }
      }
    }
  }

public:
  CellList() = default;
  CellList(const Box &box, double min_cell_width, int num_particles)
      : box_(&box) {
    resize(box, min_cell_width, num_particles);
  }

  void resize(const Box &box, double min_cell_width, int num_particles) {
    box_ = &box;
    Vec3 box_l = box_->get_dims();

    x_cells_ = std::floor(box_l.x / min_cell_width);
    y_cells_ = std::floor(box_l.y / min_cell_width);
    z_cells_ = std::floor(box_l.z / min_cell_width);

    assert(x_cells_ >= 3);
    assert(y_cells_ >= 3);
    assert(z_cells_ >= 3);

    cell_dims_.x = box_l.x / x_cells_;
    cell_dims_.y = box_l.y / y_cells_;
    cell_dims_.z = box_l.z / z_cells_;

    total_cells_ = x_cells_ * y_cells_ * z_cells_;

    head_.assign(total_cells_, -1);
    next_.assign(num_particles, -1);
    cell_map_.assign(num_particles, -1);

    precompute_neighbours();
  }

  int get_cell_index(const Vec3 &pos) const {
    Vec3 wrapped = box_->wrap(pos);

    int cx = static_cast<int>(wrapped.x / cell_dims_.x);
    int cy = static_cast<int>(wrapped.y / cell_dims_.y);
    int cz = static_cast<int>(wrapped.z / cell_dims_.z);

    cx = std::min(cx, x_cells_ - 1);
    cy = std::min(cy, y_cells_ - 1);
    cz = std::min(cz, z_cells_ - 1);

    return cx + cy * x_cells_ + cz * x_cells_ * y_cells_;
  }

  constexpr void index_to_3d(int index, int &cx, int &cy, int &cz) const {
    cx = index % x_cells_;
    cy = (index / x_cells_) % y_cells_;
    cz = index / (x_cells_ * y_cells_);
  }

  // Insert particle into cell, appending at the start of linked list
  void insert(int particle_idx, int cell_idx) {
    next_[particle_idx] = head_[cell_idx];
    head_[cell_idx] = particle_idx;
    cell_map_[particle_idx] = cell_idx;
  }

  // Remove particle
  void remove(int particle_idx, int cell_idx) {
    int *link = &head_[cell_idx];
    while (*link != particle_idx) {
      link = &next_[*link];
    }
    *link = next_[particle_idx];
    next_[particle_idx] = -1;
    cell_map_[particle_idx] = -1;
  }

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
