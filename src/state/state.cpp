#include "state.hpp"
#include <cassert>
#include <variant>

namespace hpmc {

// ====================================================================
//                           Implementation
// ====================================================================

void State::set_coords(std::span<const double> x, std::span<const double> y,
                       std::span<const double> z) {
  n_ = x.size();

  // Assert that the coordinates have consistent lengths
  if (y.size() != n_ || z.size() != n_)
    throw std::runtime_error("Incoming coordinate span sizes do not match!");

  // Copy the coordinates
  x_.assign(x.begin(), x.end());
  y_.assign(y.begin(), y.end());
  z_.assign(z.begin(), z.end());

  // Resize cell list to include all the new particles
  cell_list_.resize(
      box_,
      std::visit([&](auto &&active_cfg) { return active_cfg.min_cell_width(); },
                 config_),
      n_);

  // Insert particles into the cell list
  for (int i = 0; i < n_; ++i) {
    int cell_idx = cell_list_.get_cell_index(pos_at(i));
    cell_list_.insert(i, cell_idx);
  }
}

// TODO: Expand to work for Dumbbells
bool State::check_overlaps_naive(size_t idx, const Vec3 &pos) const {
  assert(n_ > 0 && idx < n_);

  return std::visit(
      [&](auto &&active_cfg) { return overlaps_naive(active_cfg, idx, pos); },
      config_);
}

// TODO: Expand to work for Dumbbells
size_t State::count_overlaps_naive() const {
  assert(n_ > 0);

  return std::visit(
      [&](auto &&active_cfg) { return count_overlaps_naive(active_cfg); },
      config_);
}

// TODO: Expand to work for Dumbbells
bool State::check_overlaps(size_t idx, const Vec3 &pos) const {
  assert(n_ > 0 && idx < n_);

  return std::visit(
      [&](auto &&active_cfg) { return overlaps(active_cfg, idx, pos); },
      config_);
}

// TODO: Expand to work for Dumbbells
size_t State::count_overlaps() const {
  assert(n_ > 0);

  return std::visit(
      [&](auto &&active_cfg) { return count_overlaps(active_cfg); }, config_);
}

bool State::try_translate(size_t idx, const Vec3 &displacement) {
  assert(n_ > 0 && idx < n_);

  // 1. Snapshot the current position and cell
  Vec3 old_pos = pos_at(idx);
  int old_cell = cell_list_.get_particle_cell(idx);

  // 2. Remove old cell to avoid checking for overlaps with itself
  cell_list_.remove(idx, old_cell);

  // 3. Compute new speculative trial coordinates
  Vec3 new_pos = box_.wrap(old_pos + displacement);

  if (!check_overlaps(idx, new_pos)) {
    x_[idx] = new_pos.x;
    y_[idx] = new_pos.y;
    z_[idx] = new_pos.z;

    int new_cell = cell_list_.get_cell_index(new_pos);
    cell_list_.insert(idx, new_cell);
    return true;
  }

  // Reject move, restore particle to cell
  cell_list_.insert(idx, old_cell);
  return false;
}

}; // namespace hpmc
