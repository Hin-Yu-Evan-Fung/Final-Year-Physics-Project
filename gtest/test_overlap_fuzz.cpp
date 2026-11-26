#include "core/utils.hpp"
#include "core/vec3.hpp"
#include "state/state.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace hpmc;

namespace {

inline Vec3 random_vec3(PRNG &rng, double max) {
  const double dx = rng.random_double(-max, max);
  const double dy = rng.random_double(-max, max);
  const double dz = rng.random_double(-max, max);
  return {dx, dy, dz};
}

inline void random_grid(PRNG &rng, const Lattice &lattice,
                        std::vector<double> &x, std::vector<double> &y,
                        std::vector<double> &z, size_t N) {
  x.resize(N);
  y.resize(N);
  z.resize(N);

  for (size_t i = 0; i < N; ++i) {
    const Vec3 r = lattice.to_cart({
        rng.random_double(0.0, 1.0),
        rng.random_double(0.0, 1.0),
        rng.random_double(0.0, 1.0),
    });
    x[i] = r.x;
    y[i] = r.y;
    z[i] = r.z;
  }
}

inline bool overlaps_naive(const State &state, const SphereConfig &config,
                           const Vec3 &new_pos, int p_idx) {
  for (size_t i = 0; i < state.n(); ++i)
    if (i != p_idx && config.overlap(state.lattice(), new_pos, state.pos_at(i)))
      return true;
  return false;
}

inline size_t count_overlaps_naive(const State &state,
                                   const SphereConfig &config) {
  size_t n = state.n();
  size_t overlaps = 0;
  for (size_t i = 0; i < n - 1; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      overlaps +=
          config.overlap(state.lattice(), state.pos_at(i), state.pos_at(j));
    }
  }
  return overlaps;
}

} // namespace

class OverlapFuzzTest : public ::testing::TestWithParam<Lattice> {};

TEST_P(OverlapFuzzTest, CellListMatchesNaiveOracle) {
  PRNG rng(1337);

  constexpr size_t N = 64;
  constexpr size_t L = 8.0;

  SphereConfig config{0.5};
  const Lattice lattice = GetParam();
  State state(config, lattice);

  std::vector<double> x, y, z;
  random_grid(rng, lattice, x, y, z, N);
  state.set_coords(x, y, z);

  for (int step = 0; step < 10000; ++step) {
    size_t p_idx = rng.random_index(N);
    Vec3 displacement = random_vec3(rng, 0.3);

    const Vec3 new_pos = state.displaced_pos(p_idx, displacement);
    const int new_cell = state.cell_of(new_pos);

    bool overlap_naive = overlaps_naive(state, config, new_pos, p_idx);
    bool overlap = state.has_overlaps(new_pos, new_cell, p_idx);

    ASSERT_EQ(overlap_naive, overlap) << "Overlap Fuzz failure at iteration "
                                      << step << " for particle " << p_idx;

    // Ensure overlaps do occur
    if (rng.random_double() > 0.1) {
      state.set_pos(p_idx, new_pos, new_cell);
    }

    size_t overlap_count_naive = count_overlaps_naive(state, config);
    size_t overlap_count = state.count_all_overlaps();

    ASSERT_EQ(overlap_count_naive, overlap_count)
        << "Overlap count Fuzz failure at iteration " << step
        << " for particle " << p_idx;
  }
}

TEST_P(OverlapFuzzTest, OverlapTracking) {
  PRNG rng(1337);

  constexpr size_t N = 64;
  constexpr size_t L = 8.0;

  SphereConfig config{0.5};
  const Lattice lattice = GetParam();
  State state(config, lattice);

  std::vector<double> x1, y1, z1;
  random_grid(rng, lattice, x1, y1, z1, N);
  state.set_coords(x1, y1, z1);

  int total_overlaps = state.count_all_overlaps();

  for (int step = 0; step < 10000; ++step) {
    size_t p_idx = rng.random_index(N);
    Vec3 displacement = random_vec3(rng, 0.3);

    const Vec3 old_pos = state.pos_at(p_idx);
    const int old_cell = state.cell_of(old_pos);
    const Vec3 new_pos = state.displaced_pos(p_idx, displacement);
    const int new_cell = state.cell_of(new_pos);

    int old_overlaps = state.count_overlaps(old_pos, old_cell, p_idx);
    int new_overlaps = state.count_overlaps(new_pos, new_cell, p_idx);
    state.set_pos(p_idx, new_pos, new_cell);

    // Use difference in overlaps to update the total overlaps
    total_overlaps += new_overlaps - old_overlaps;

    size_t overlap_count_naive = count_overlaps_naive(state, config);

    ASSERT_EQ(overlap_count_naive, total_overlaps)
        << "Overlap count Fuzz failure at iteration " << step
        << " for particle " << p_idx;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Lattices, OverlapFuzzTest,
    ::testing::Values(
        // Standard Cubic
        Lattice::from_upper_triangular({8.0, 0.0, 0.0}, {0.0, 8.0, 0.0},
                                       {0.0, 0.0, 8.0}),
        // Orthorhombic (non-cubic aspect ratio)
        Lattice::from_upper_triangular({6.0, 0.0, 0.0}, {0.0, 8.5, 0.0},
                                       {0.0, 0.0, 11.0}),
        // Hexagonal / HCP-style (60-degree xy angle)
        Lattice::from_upper_triangular({8.0, 0.0, 0.0}, {4.0, 6.9282, 0.0},
                                       {0.0, 0.0, 9.0}),
        // Strongly Sheared Triclinic
        Lattice::from_upper_triangular({8.0, 0.0, 0.0}, {3.5, 7.5, 0.0},
                                       {-2.5, 3.0, 7.0})));
