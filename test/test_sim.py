import numpy as np
import pytest

import hpmc
from hpmc.utils.lattice import generate_cubic_lattice
from hpmc.utils.math import generate_dist_matrix


@pytest.fixture
def sim():
    """Fixture to quickly spawn a fresh sim state for testing."""
    params = hpmc.HardSphereParams(0.5)  # Example radius/interaction parameter
    sim = hpmc.HardSphereSimulation(L=10.0, params=params)

    init_pos = np.array(
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]], dtype=np.float64
    )

    sim.set_particles(init_pos)
    return sim


def test_sim_initialization(sim):
    """Verify size constraints register accurately."""
    assert sim.n_particles() == 3


def test_successful_batch_run(sim):
    """Verify that a valid array of moves executes completely."""
    indices = np.array([0, 1], dtype=np.uint32)

    print(indices.shape)

    moves = np.array([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]], dtype=np.float64)

    accepted = sim.try_run_batch(indices, moves)
    assert isinstance(accepted, int)
    assert accepted <= len(indices)


def test_noconvert_type_checking_fail(sim):
    """Verify nanobind's .noconvert() blocks mismatched dtypes instantly."""
    indices = np.array([0, 1], dtype=np.uint32)

    # Malicious type float32 instead of expected float64
    bad_type_moves = np.array([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]], dtype=np.float32)

    with pytest.raises(RuntimeError):
        sim.try_run_batch(indices, bad_type_moves)


def test_noconvert_shape_checking_fail(sim):
    """Verify nanobind blocks incorrect Degrees of Freedom matrix dimensions."""
    indices = np.array([0, 1], dtype=np.uint32)

    # 4 columns instead of the hardcoded 3 DOFs
    bad_shape_moves = np.array(
        [[0.1, 0.0, 0.0, 9.9], [0.0, -0.1, 0.0, 9.9]], dtype=np.float64
    )

    with pytest.raises(RuntimeError):
        sim.try_run_batch(indices, bad_shape_moves)


def test_non_contiguous_array_rejection(sim):
    """Verify that sliced/non-contiguous matrices are cleanly rejected."""
    indices = np.array([0, 1], dtype=np.uint32)

    large_moves = np.random.rand(10, 3).astype(np.float64)
    sliced_moves = large_moves[::2, :]

    with pytest.raises(RuntimeError):
        sim.try_run_batch(indices, sliced_moves)


def test_hard_core_overlap_matrix_method():
    """
    Validates particle exclusion using the upper-triangular matrix extraction method.
    """
    L = 10.0
    N = 256
    MOVES = 100_000
    SIGMA = 0.5
    DIAM = 2 * SIGMA

    sim = hpmc.HardSphereSimulation(L=L, params=hpmc.HardSphereParams(SIGMA))

    init_pos = generate_cubic_lattice(N, L, SIGMA)
    sim.set_particles(init_pos)

    assert sim.count_overlaps() == 0, "There should be no particle overlaps initially"

    moves = np.random.uniform(-0.1, 0.1, size=(MOVES, 3)).astype(np.float64)
    indices = np.random.randint(0, N, size=MOVES, dtype=np.uint32)

    accepted = sim.try_run_batch(indices=indices, moves=moves)

    new_pos = sim.get_particles()
    dist_matrix = generate_dist_matrix(new_pos, L)

    rows, cols = np.triu_indices(N, k=1)
    unique_distances = dist_matrix[rows, cols]

    min_distance = np.min(unique_distances)

    assert min_distance >= (DIAM - 1e-7), (
        f"Overlap detected! Minimum distance is {min_distance} "
        f"between particle {rows[np.argmin(unique_distances)]} "
        f"and particle {cols[np.argmin(unique_distances)]}."
    )
