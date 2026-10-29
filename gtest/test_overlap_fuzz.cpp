#include "core/utils.hpp"
#include "core/vec3.hpp"
#include "particle/particle.hpp"
#include "particle/sphere.hpp"
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

inline void random_grid(PRNG &rng, std::vector<double> &x,
                        std::vector<double> &y, std::vector<double> &z,
                        size_t n_coords, double box_length) {
  x.resize(n_coords);
  y.resize(n_coords);
  z.resize(n_coords);

  for (size_t i = 0; i < n_coords; ++i) {
    x[i] = rng.random_double(0.0, box_length);
    y[i] = rng.random_double(0.0, box_length);
    z[i] = rng.random_double(0.0, box_length);
  }
}

inline bool overlaps_naive(const State &state, const SphereConfig &config,
                           const MoveProposal &prop) {
  for (size_t i = 0; i < state.n(); ++i)
    if (i != prop.idx &&
        config.overlap(state.box(), prop.new_pos, state.pos_at(i)))
      return true;
  return false;
}

inline size_t count_overlaps_naive(const State &state,
                                   const SphereConfig &config) {
  size_t n = state.n();
  size_t overlaps = 0;
  for (size_t i = 0; i < n - 1; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      overlaps += config.overlap(state.box(), state.pos_at(i), state.pos_at(j));
    }
  }
  return overlaps;
}

} // namespace

TEST(OverlapFuzzTest, CellListMatchesNaiveOracle) {
  PRNG rng(1337);

  constexpr size_t N = 64;
  constexpr size_t L = 8.0;

  SphereConfig config{0.5};
  State state(ParticleType::Sphere, {L, L, L}, config);

  std::vector<double> x, y, z;
  random_grid(rng, x, y, z, N, L);
  state.set_coords(x, y, z);

  for (int step = 0; step < 10000; ++step) {
    size_t p_idx = rng.random_index(N);
    Vec3 displacement = random_vec3(rng, 0.3);

    auto prop = state.propse_translation(p_idx, displacement);

    bool overlap_naive = overlaps_naive(state, config, prop);
    bool overlap = state.check_overlap(prop);

    ASSERT_EQ(overlap_naive, overlap) << "Overlap Fuzz failure at iteration "
                                      << step << " for particle " << p_idx;

    // Ensure overlaps do occur
    if (rng.random_double() > 0.1) {
      state.accept_translation(prop);
    }

    size_t overlap_count_naive = count_overlaps_naive(state, config);
    size_t overlap_count = state.count_global_overlaps();

    ASSERT_EQ(overlap_count_naive, overlap_count)
        << "Overlap count Fuzz failure at iteration " << step
        << " for particle " << p_idx;
  }
}

TEST(OverlapFuzzTest, OverlapTracking) {
  PRNG rng(1337);

  constexpr size_t N = 64;
  constexpr size_t L = 8.0;

  SphereConfig config{0.5};
  State state(ParticleType::Sphere, {L, L, L}, config);

  std::vector<double> x1, y1, z1;
  random_grid(rng, x1, y1, z1, N, L);
  state.set_coords(x1, y1, z1);

  int total_overlaps = state.count_global_overlaps();

  for (int step = 0; step < 10000; ++step) {
    size_t p_idx = rng.random_index(N);
    Vec3 displacement = random_vec3(rng, 0.3);

    auto prop = state.propse_translation(p_idx, displacement);

    int old_overlaps =
        state.count_local_overlaps(p_idx, prop.old_pos, prop.old_cell);
    int new_overlaps =
        state.count_local_overlaps(p_idx, prop.new_pos, prop.new_cell);
    state.accept_translation(prop);

    // Use difference in overlaps to update the total overlaps
    total_overlaps += new_overlaps - old_overlaps;

    size_t overlap_count_naive = count_overlaps_naive(state, config);

    ASSERT_EQ(overlap_count_naive, total_overlaps)
        << "Overlap count Fuzz failure at iteration " << step
        << " for particle " << p_idx;
  }
}
