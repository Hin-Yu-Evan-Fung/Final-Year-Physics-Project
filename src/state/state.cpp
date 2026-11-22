#include "state/state.hpp"

namespace hpmc {

void State::set_coords(std::span<const double> x, std::span<const double> y,
                       std::span<const double> z) {
  const size_t new_n = x.size();
  if (y.size() != new_n || z.size() != new_n) {
    throw std::invalid_argument("Incoming coordinate span sizes do not match!");
  }

  n_ = new_n;

  // Copy the coordinates
  x_.assign(x.begin(), x.end());
  y_.assign(y.begin(), y.end());
  z_.assign(z.begin(), z.end());

  double min_cell_width = std::visit(
      [&](const auto &cfg) { return cfg.min_cell_width(); }, config_);

  // Resize cell list to include all the new particles
  cell_list_.resize(lattice_, min_cell_width, n_);

  // Insert particles into the cell list
  for (int i = 0; i < n_; ++i) {
    const Vec3 wrapped = lattice_.wrap(pos_at(i));
    x_[i] = wrapped.x;
    y_[i] = wrapped.y;
    z_[i] = wrapped.z;

    const int cell_idx = cell_list_.get_cell_index(lattice_, wrapped);
    cell_list_.insert(i, cell_idx);
  }
}

bool State::has_overlaps(const Vec3 &pos, size_t cell, int excl_idx) const {
  return std::visit(
      [&](const auto &cfg) {
        return local_overlaps<true>(cfg, pos, cell, excl_idx) > 0;
      },
      config_);
}
size_t State::count_overlaps(const Vec3 &pos, size_t cell, int excl_idx) const {
  return std::visit(
      [&](const auto &cfg) {
        return local_overlaps<false>(cfg, pos, cell, excl_idx);
      },
      config_);
}
bool State::has_any_overlaps() const {
  return std::visit(
      [&](const auto &cfg) { return global_overlaps<true>(cfg) > 0; }, config_);
}
size_t State::count_all_overlaps() const {
  return std::visit(
      [&](const auto &cfg) { return global_overlaps<false>(cfg); }, config_);
}

} // namespace hpmc
