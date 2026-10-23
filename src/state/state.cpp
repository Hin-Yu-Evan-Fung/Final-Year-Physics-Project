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
bool State::check_overlaps(const MoveProposal &prop) const {
  assert(n_ > 0 && prop.idx < n_);

  return std::visit(
      [&](auto &&active_cfg) { return overlaps(active_cfg, prop); }, config_);
}

// TODO: Expand to work for Dumbbells
size_t State::count_overlaps() const {
  assert(n_ > 0);

  return std::visit(
      [&](auto &&active_cfg) { return count_overlaps(active_cfg); }, config_);
}

MoveProposal State::propse_translation(size_t idx, const Vec3 &displacement) {
  assert(n_ > 0 && idx < n_);

  MoveProposal prop;
  prop.idx = idx;
  prop.old_pos = pos_at(idx);
  prop.old_cell = cell_list_.get_particle_cell(idx);
  prop.new_pos = box_.wrap(prop.old_pos + displacement);
  prop.new_cell = cell_list_.get_cell_index(prop.new_pos);

  cell_list_.remove(idx, prop.old_cell);

  return prop;
}

void State::accept_move(const MoveProposal &prop) {
  x_[prop.idx] = prop.new_pos.x;
  y_[prop.idx] = prop.new_pos.y;
  z_[prop.idx] = prop.new_pos.z;

  cell_list_.insert(prop.idx, prop.new_cell);
}

void State::reject_move(const MoveProposal &prop) {
  cell_list_.insert(prop.idx, prop.old_cell);
}

}; // namespace hpmc
