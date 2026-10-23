import numpy as np

import hpmc
from hpmc.utils.lattice import generate_cubic_lattice
from hpmc.utils.math import generate_dist_matrix


def test_hard_core_overlap_matrix_method():
    """
    Validates particle exclusion using the upper-triangular matrix extraction method.
    """
    L = 10.0
    N = 256
    SWEEPS = 1_000
    SIGMA = 0.5
    DIAM = 2 * SIGMA
    MAX_DISP = 0.2 * SIGMA

    sim = hpmc.SingleBoxEngine(hpmc.ParticleType.Sphere, L, radius=SIGMA)

    init_coords = generate_cubic_lattice(N, L, SIGMA)
    sim.set_coords(*init_coords)

    assert sim.count_overlaps_naive() == 0, (
        "There should be no particle overlaps initially"
    )

    sim.run_batch(sweeps=SWEEPS, max_disp=MAX_DISP)

    new_pos = sim.get_coords()
    dist_matrix = generate_dist_matrix(*new_pos, L)

    rows, cols = np.triu_indices(N, k=1)
    unique_distances = dist_matrix[rows, cols]

    min_distance = np.min(unique_distances)

    assert min_distance >= (DIAM - 1e-7), (
        f"Overlap detected! Minimum distance is {min_distance} "
        f"between particle {rows[np.argmin(unique_distances)]} "
        f"and particle {cols[np.argmin(unique_distances)]}."
    )
