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

  if (y.size() != n_ || z.size() != n_)
    throw std::runtime_error("Incoming coordinate span sizes do not match!");

  x_.assign(x.begin(), x.end());
  y_.assign(y.begin(), y.end());
  z_.assign(z.begin(), z.end());
}

// TODO: Expand to work for Dumbbells
bool State::check_overlaps_naive(size_t idx) const {
  assert(n_ > 0 && idx < n_);

  const Vec3 pos = pos_at(idx);

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

void State::translate(size_t idx, const Vec3 &displacement) {
  assert(n_ > 0 && idx < n_);

  curr_idx_ = idx;
  old_pos_ = pos_at(idx);
  Vec3 new_pos = box_.wrap(old_pos_ + displacement);

  x_[curr_idx_] = new_pos.x;
  y_[curr_idx_] = new_pos.y;
  z_[curr_idx_] = new_pos.z;
}

void State::reject_move() {
  assert(curr_idx_ != -1);
  x_[curr_idx_] = old_pos_.x;
  y_[curr_idx_] = old_pos_.y;
  z_[curr_idx_] = old_pos_.z;

  // Reset cache tracker flag
  curr_idx_ = -1;
}

}; // namespace hpmc
