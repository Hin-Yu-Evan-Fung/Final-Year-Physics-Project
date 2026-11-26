#include "core/utils.hpp"
#include "core/vec3.hpp"
#include "state/cell_list.hpp"

#include <gtest/gtest.h>

using namespace hpmc;

class CellListTest : public ::testing::TestWithParam<Lattice> {
protected:
  CellList grid;
  double min_cell_width = 2.0;
  size_t num_particles = 100000;

  void SetUp() override {
    const Lattice lattice = GetParam();
    grid.resize(lattice, min_cell_width, num_particles);
  }
};

TEST_P(CellListTest, CellIndex) {
  const Lattice lattice = GetParam();

  ASSERT_GT(grid.total_cells(), 0);

  // Fractional (0.01, 0.01, 0.01) must always map to cell 0
  EXPECT_EQ(grid.get_cell_index(lattice, lattice.to_cart({0.01, 0.01, 0.01})),
            0);

  // Fractional (0.999, 0.999, 0.999) must always map to the last cell
  EXPECT_EQ(
      grid.get_cell_index(lattice, lattice.to_cart({0.999, 0.999, 0.999})),
      grid.total_cells() - 1);
}

TEST_P(CellListTest, Neighbours) {
  // Test neighbours are not completely off
  PRNG rng(1);

  for (size_t i = 0; i < 1000; ++i) {
    int test_cell = rng.random_index(grid.total_cells());
    const auto &local_neighbours = grid.get_neighbours(test_cell);

    EXPECT_EQ(local_neighbours.size(), 27);
    bool found_self = false;
    for (int n : local_neighbours) {
      ASSERT_LT(n, grid.total_cells()) << "cell_idx: " << test_cell;
      if (n == test_cell)
        found_self = true;
    }

    EXPECT_TRUE(found_self);
  }
}

TEST_P(CellListTest, CellIndexInsertion) {
  const Lattice lattice = GetParam();
  PRNG rng(42);

  for (size_t i = 0; i < num_particles; ++i) {
    // Sample uniform fractional coords -> works identically for sheared
    // triclinic boxes
    const Vec3 pos = lattice.to_cart({
        rng.random_double(0.0, 1.0),
        rng.random_double(0.0, 1.0),
        rng.random_double(0.0, 1.0),
    });

    int cell_idx = grid.get_cell_index(lattice, pos);

    // Assert that the index is bounded within our valid array memory range
    ASSERT_GE(cell_idx, 0);
    ASSERT_LT(cell_idx, grid.total_cells());

    grid.insert(i, cell_idx);
  }
}

TEST_P(CellListTest, LinkedListFuzz) {
  PRNG rng(1);

  for (int i = 0; i < num_particles; ++i) {
    int target_cell = rng.random_index(grid.total_cells());
    grid.insert(i, target_cell);
  }

  for (int step = 0; step < 5000; ++step) {
    int p_id = rng.random_index(num_particles);
    int current_cell = grid.get_particle_cell(p_id);

    if (current_cell != -1)
      grid.remove(p_id, current_cell);

    if (rng.random_double() > 0.20) {
      int new_cell = rng.random_index(grid.total_cells());
      grid.insert(p_id, new_cell);
    }
  }

  for (size_t cell = 0; cell < grid.total_cells(); ++cell) {
    int current = grid.get_head(cell);
    while (current != -1) {
      ASSERT_EQ(grid.get_particle_cell(current), static_cast<int>(cell))
          << "Data corruption detected! Particle " << current
          << " found in cell " << cell << " but registered elsewhere.";

      current = grid.get_next(current);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Lattices, CellListTest,
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
