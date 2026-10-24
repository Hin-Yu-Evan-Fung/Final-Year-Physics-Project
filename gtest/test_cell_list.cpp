#include "core/utils.hpp"
#include "core/vec3.hpp"
#include "state/cell_list.hpp"
#include <gtest/gtest.h>

using namespace hpmc;

class MassiveCellListTest : public ::testing::Test {
protected:
  // 100.0 x 100.0 x 100.0 simulation box
  Box box{100.0};
  // min_cell_width = 2.0 -> 100 / 2 = 50 cells per axis.
  // 50 * 50 * 50 = 125,000 cells total
  double scale_cell_width = 2.0;
  size_t num_particles = 100000;
};

TEST_F(MassiveCellListTest, CellIndex) {
  CellList grid(box, scale_cell_width, num_particles);

  // Verify cell boundaries map correctly at large scales
  // First cell (cx=0, cy=0, cz=0) -> 0
  EXPECT_EQ(grid.get_cell_index({0.5, 0.5, 0.5}), 0);

  // Middle cell (cx=25, cy=25, cz=25) -> 25 + 25*50 + 25*2500 = 63775
  EXPECT_EQ(grid.get_cell_index({50.5, 50.5, 50.5}), 63775);

  // Far corner cell (cx=49, cy=49, cz=49) -> 49 + 49*50 + 49*2500 = 124999
  EXPECT_EQ(grid.get_cell_index({99.9, 99.9, 99.9}), 124999);
}

TEST_F(MassiveCellListTest, Neighbours) {
  CellList grid(box, scale_cell_width, num_particles);

  // Test neighbours are not completely off
  PRNG rng(1);

  for (size_t i = 0; i < 1000; ++i) {
    int test_cell = rng.random_index(grid.total_cells());
    const auto &local_neighbours = grid.get_neighbours(test_cell);

    EXPECT_EQ(local_neighbours.size(), 27);
    bool found_self = false;
    for (int n : local_neighbours) {
      ASSERT_GE(n, 0) << "cell_idx: " << test_cell;
      ASSERT_LT(n, 125000) << "cell_idx: " << test_cell;
      if (n == test_cell)
        found_self = true;
    }

    EXPECT_TRUE(found_self);
  }
}

TEST_F(MassiveCellListTest, CellIndexInsertion) {
  CellList grid(box, scale_cell_width, num_particles);

  for (size_t i = 0; i < num_particles; ++i) {
    // Distribute coordinates deterministically across the box space
    double x = (i * 0.123);
    double y = (i * 0.456);
    double z = (i * 0.789);

    Vec3 pos{x, y, z};

    int cell_idx = grid.get_cell_index({x, y, z});

    // Assert that the index is bounded within our valid array memory range
    ASSERT_GE(cell_idx, 0);
    ASSERT_LT(cell_idx, 125000);

    grid.insert(i, cell_idx);
  }
}

TEST_F(MassiveCellListTest, LinkedListFuzz) {
  CellList grid(box, scale_cell_width, num_particles);

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
