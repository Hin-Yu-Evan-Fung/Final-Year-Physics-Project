from collections.abc import Sequence

import numpy as np
import pytest

import hpmc
from hpmc.utils.lattice import generate_cubic_lattice, generate_lattice
from hpmc.utils.math import generate_dist_matrix


def calculate_compressibility_factor(
    positions: tuple[np.ndarray, np.ndarray, np.ndarray],
    packing_fraction: float,
    box_dims: Sequence[float],
    radius: float,
):
    """
    Thermodynamic observer: Accurately calculates Z by fitting a polynomial
    to a fine histogram of neighbor distances and extrapolating back to contact.
    """
    N = len(positions[0])

    sigma = radius
    diameter = 2.0 * sigma

    rho = N / np.prod(box_dims)

    dist_matrix = generate_dist_matrix(*positions, box_dims)
    rows, cols = np.triu_indices(N, k=1)
    distances = dist_matrix[rows, cols]

    # Setup a fine-grained histogram shell just outside the hard core
    num_bins = 10
    bin_edges = np.linspace(diameter, diameter + (0.15 * radius), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Count how many distances fall into each thin bin
    counts, _ = np.histogram(distances, bins=bin_edges)

    # Calculate the true g(r) for each bin by normalizing against ideal gas volumes
    g_r = np.zeros(num_bins)
    for i in range(num_bins):
        r_inner = bin_edges[i]
        r_outer = bin_edges[i + 1]

        # Exact geometric volume of this specific thin shell
        shell_volume = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
        expected_ideal_counts = 0.5 * N * rho * shell_volume

        g_r[i] = counts[i] / expected_ideal_counts if expected_ideal_counts > 0 else 0.0

    coefficients = np.polyfit(bin_centers, g_r, deg=1)
    g_contact_extrapolated = np.polyval(coefficients, diameter)

    # If density is extremely low, fallback to 1.0 to prevent fitting noise
    if g_contact_extrapolated < 0.5:
        g_contact_extrapolated = 1.0

    # Calculate compressibility factor Z
    return 1.0 + 4.0 * packing_fraction * g_contact_extrapolated


# fmt: off
@pytest.mark.parametrize(
    "lattice_generator, lattice_type, packing_fraction, nx, ny, nz, eq_sweeps, max_disp, tolerance",
    [
        # 1. Ideal Gas Limit: Fast relaxation, roomy steps
        (generate_cubic_lattice, None,  0.004, 6, 6, 6, 1000,  2.0, 0.03),
        (generate_lattice,       "fcc", 0.004, 6, 6, 6, 1000,  2.0, 0.03),
        
        # 2. Low Density Fluid: Quick thermalization
        (generate_cubic_lattice, None,  0.065, 6, 6, 6, 3000,  1.0, 0.03),
        (generate_lattice,       "hcp", 0.065, 6, 6, 6, 3000,  1.0, 0.06),
        
        # 3. Medium Density Fluid: Moderate cage lifetimes
        (generate_cubic_lattice, None,  0.190, 6, 6, 6, 6000,  0.5, 0.05),
        (generate_lattice,       "fcc", 0.190, 6, 6, 6, 6000,  0.5, 0.10),
        
        # 4. Dense Fluid: Very slow cage relaxation, needs small steps
        (generate_cubic_lattice, None,  0.360, 6, 6, 6, 15000, 0.1, 0.05),
        (generate_lattice,       "hcp", 0.360, 6, 6, 6, 15000, 0.1, 0.10),
    ],
)
def test_thermodynamic_ideal_gas_limit(
    lattice_generator, lattice_type, packing_fraction, nx, ny, nz, eq_sweeps, max_disp, tolerance
):
# fmt: on
    """
    Thermodynamic Test 1: Validates that a highly dilute system
    behaves like an ideal gas (Z safely approaches 1.0).
    """
    RADIUS = 0.5

    # 1. Dynamically build the dictionary arguments for your standard lattice APIs
    gen_kwargs = {
        "packing_fraction": packing_fraction,
        "radius": RADIUS,
        "nx": nx,
        "ny": ny,
        "nz": nz,
    }

    if lattice_type is not None:
        gen_kwargs["lattice_type"] = lattice_type

    # 2. Generate coordinates and the calculated box length matrix dynamically
    x, y, z, box = lattice_generator(**gen_kwargs)

    # 3. Initialize simulation environment
    state = hpmc.State(hpmc.ParticleType.Sphere, box, radius=RADIUS)
    state.set_coords(x, y, z)

    assert state.count_overlaps() == 0, "There should be no particle overlaps initially"

    # Long Equilibrate Phase: Melt the grid completely
    sim = hpmc.SingleBoxEngine(state)
    sim.run_batch(sweeps=eq_sweeps, max_disp=max_disp)

    # Production Phase: Sample the system 5 times to smooth out frame noise
    z_samples = []
    SAMPLE_BLOCKS = 50
    SWEEPS_PER_BLOCK = 100

    for i in range(SAMPLE_BLOCKS):
        sim.run_batch(sweeps=SWEEPS_PER_BLOCK, max_disp=max_disp)

        # Take a snapshot metric at the end of this block
        current_pos = state.get_coords()
        z_samples.append(
            calculate_compressibility_factor(current_pos, packing_fraction, box, RADIUS)
        )

    # Compute the clean ensemble average
    Z_averaged = np.mean(z_samples)

    # Use Carnahan-Starling equation of state
    eta = packing_fraction
    Z_theory = (1 + eta + eta**2 - eta**3) / (1 - eta) ** 3

    assert Z_averaged == pytest.approx(Z_theory, abs=tolerance), (
        f"Thermodynamic Failure: Averaged compressibility factor Z was {Z_averaged:.4f}, "
        f"expected it to converge close to the physical target of {Z_theory}."
    )
