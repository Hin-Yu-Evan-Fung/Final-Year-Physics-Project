#include "cell_list.hpp"

namespace hpmc {

void CellList::precompute_neighbours() {

  // clang-format off
  static const int OFFSETS[27][3] = {
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
    {-1,  1,  1}, {-1,  1, -1}, {-1, -1,  1}, {-1, -1, -1}
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

void CellList::resize(const Box &box, double min_cell_width,
                      int num_particles) {
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

int CellList::get_cell_index(const Vec3 &pos) const {
  Vec3 wrapped = box_->wrap(pos);

  int cx = static_cast<int>(wrapped.x / cell_dims_.x);
  int cy = static_cast<int>(wrapped.y / cell_dims_.y);
  int cz = static_cast<int>(wrapped.z / cell_dims_.z);

  cx = std::min(cx, x_cells_ - 1);
  cy = std::min(cy, y_cells_ - 1);
  cz = std::min(cz, z_cells_ - 1);

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
  next_[particle_idx] = -1;
  cell_map_[particle_idx] = -1;
}

} // namespace hpmc
