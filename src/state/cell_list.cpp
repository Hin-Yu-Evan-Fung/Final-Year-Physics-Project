
#include "state/cell_list.hpp"
#include "core/utils.hpp"

namespace hpmc {

void CellList::precompute_neighbours() {

  // clang-format off
  static constexpr NDArray<int, 27, 3> OFFSETS = {{
    // Center (1)
    { 0,  0,  0},
    // Faces (6)
    { 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0}, { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1},
    // Edges (12)
    { 1,  1,  0}, { 1, -1,  0}, {-1,  1,  0}, {-1, -1,  0},
    { 1,  0,  1}, { 1,  0, -1}, {-1,  0,  1}, {-1,  0, -1},
    { 0,  1,  1}, { 0,  1, -1}, { 0, -1,  1}, { 0, -1, -1},
    // Corners (8)
    { 1,  1,  1}, { 1,  1, -1}, { 1, -1,  1}, { 1, -1, -1},
    {-1,  1,  1}, {-1,  1, -1}, {-1, -1,  1}, {-1, -1, -1}}
  };
  // clang-format on

  neighbours_.resize(total_cells_);
  for (int cell_idx = 0; cell_idx < total_cells_; ++cell_idx) {
    int cx, cy, cz;
    index_to_3d(cell_idx, cx, cy, cz);

    for (int i = 0; i < 27; ++i) {
      int nx = grid_pbc_wrap(cx + OFFSETS[i][0], x_cells_);
      int ny = grid_pbc_wrap(cy + OFFSETS[i][1], y_cells_);
      int nz = grid_pbc_wrap(cz + OFFSETS[i][2], z_cells_);

      neighbours_[cell_idx][i] = nx + ny * x_cells_ + nz * x_cells_ * y_cells_;
    }
  }
}

void CellList::resize(const Lattice &lattice, double min_cell_width,
                      size_t num_particles) {
  Vec3 widths = lattice.perp_widths();

  x_cells_ = (int)std::floor(widths.x / min_cell_width);
  y_cells_ = (int)std::floor(widths.y / min_cell_width);
  z_cells_ = (int)std::floor(widths.z / min_cell_width);

  if (x_cells_ < 3 || y_cells_ < 3 || z_cells_ < 3) {
    throw std::invalid_argument(std::format(
        "CellList requires at least 3x3x3 cells (widths >= 3 * "
        "min_cell_width), "
        "got grid ({}, {}, {}) from perpendicular widths ({:.4g}, {:.4g}, "
        "{:.4g})",
        x_cells_, y_cells_, z_cells_, widths.x, widths.y, widths.z));
  }

  cell_dims_.x = widths.x / x_cells_;
  cell_dims_.y = widths.y / y_cells_;
  cell_dims_.z = widths.z / z_cells_;

  total_cells_ = x_cells_ * y_cells_ * z_cells_;

  head_.assign(total_cells_, EMPTY);
  next_.assign(num_particles, EMPTY);
  cell_map_.assign(num_particles, EMPTY);

  precompute_neighbours();
}

int CellList::get_cell_index(const Lattice &lattice, const Vec3 &pos) const {
  return get_cell_index_from_fractional(lattice.to_frac(pos));
}

int CellList::get_cell_index_from_fractional(const Vec3 &s) const {
  // clang-format off
  assert(s.x >= -1e-12 && s.x <= 1.0 + 1e-12 &&
         s.y >= -1e-12 && s.y <= 1.0 + 1e-12 &&
         s.z >= -1e-12 && s.z <= 1.0 + 1e-12 &&
         "get_cell_index expects a wrapped position!");
  // clang-format on

  int cx = std::clamp(static_cast<int>(s.x * x_cells_), 0, x_cells_ - 1);
  int cy = std::clamp(static_cast<int>(s.y * y_cells_), 0, y_cells_ - 1);
  int cz = std::clamp(static_cast<int>(s.z * z_cells_), 0, z_cells_ - 1);

  return cx + cy * x_cells_ + cz * x_cells_ * y_cells_;
}

// Append to the start of linked list
void CellList::insert(int particle_idx, int cell_idx) {
  next_[particle_idx] = head_[cell_idx];
  head_[cell_idx] = particle_idx;
  cell_map_[particle_idx] = cell_idx;
}

// Remove particle
void CellList::remove(int particle_idx, int cell_idx) {
  int *link = &head_[cell_idx];
  while (*link != particle_idx) {
    link = &next_[*link];
  }
  *link = next_[particle_idx];
  next_[particle_idx] = EMPTY;
  cell_map_[particle_idx] = EMPTY;
}

} // namespace hpmc
