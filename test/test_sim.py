import numpy as np
import pytest

import hpmc
from hpmc.utils.lattice import generate_cubic_lattice, generate_lattice
from hpmc.utils.math import generate_dist_matrix


# fmt: off
@pytest.mark.parametrize(
    (
        "lattice_generator",
        "lattice_type",
        "packing_fraction",
        "nx",
        "ny",
        "nz",
    ),
    [
        # 1. Standard Baselines
        (generate_cubic_lattice, None, 0.40, 6, 6, 6),
        (generate_lattice, "fcc", 0.40, 4, 4, 18),
        (generate_lattice, "hcp", 0.40, 4, 4, 12),

        # 2. Asymmetric Box Dimensions
        (generate_cubic_lattice, None, 0.35, 4, 6, 8),
        (generate_lattice, "fcc", 0.45, 8, 4, 6),
        (generate_lattice, "hcp", 0.38, 5, 8, 12),

        # 3. Very Low Density 
        (generate_cubic_lattice, None, 0.05, 5, 5, 5),
        (generate_lattice, "fcc", 0.01, 4, 4, 6),

        # 4. Near Maximum Limits 
        (generate_cubic_lattice, None, 0.52, 6, 6, 6),
        (generate_lattice, "fcc", 0.74, 4, 4, 18),
        (generate_lattice, "hcp", 0.74, 4, 4, 12),

        # 5. Small Minimum Systems
        (generate_cubic_lattice, None, 0.30, 3, 3, 3),
        (generate_lattice, "fcc", 0.50, 4, 4, 24),
    ],
)
def test_lattice_overlap(lattice_generator, lattice_type, packing_fraction, nx, ny, nz):
# fmt: on
    """
    Validates particle exclusion using the upper-triangular matrix extraction method.
    """
    SWEEPS = 1_000
    RADIUS = 0.5
    DIAM = RADIUS * 2
    MAX_DISP = 0.2 * RADIUS

    if lattice_type is not None:
        x, y, z, box = lattice_generator(
            packing_fraction=packing_fraction,
            radius=0.5,
            lattice_type=lattice_type,
            nx=nx,
            ny=ny,
            nz=nz,
        )
    else:
        x, y, z, box = lattice_generator(
            packing_fraction=packing_fraction, radius=0.5, nx=nx, ny=ny, nz=nz
        )

    state = hpmc.State(hpmc.ParticleType.Sphere, box, radius=RADIUS)
    state.set_coords(x, y, z)

    assert state.count_overlaps() == 0, "There should be no particle overlaps initially"

    sim = hpmc.SingleBoxEngine(state)
    sim.run_batch(sweeps=SWEEPS, max_disp=MAX_DISP)

    assert state.count_overlaps() == 0, "There should be no particle overlaps after"

    new_pos = state.get_coords()
    dist_matrix = generate_dist_matrix(*new_pos, box)

    rows, cols = np.triu_indices(nx * ny * nz, k=1)
    unique_distances = dist_matrix[rows, cols]
    min_distance = np.min(unique_distances)

    assert min_distance >= (DIAM - 1e-7), (
        f"Overlap detected! Minimum distance is {min_distance} "
        f"between particle {rows[np.argmin(unique_distances)]} "
        f"and particle {cols[np.argmin(unique_distances)]}."
    )
