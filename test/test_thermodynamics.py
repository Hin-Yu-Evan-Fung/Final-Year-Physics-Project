import numpy as np
import pytest

import hpmc
from hpmc.utils.lattice import generate_cubic_lattice
from hpmc.utils.math import generate_dist_matrix


def calculate_packing_fraction(
    num_particles: int, box_length: float, radius: float
) -> float:
    # Calculate packing fraction (eta)
    sphere_volume = num_particles * (4.0 / 3.0) * np.pi * (radius**3)
    box_volume = box_length**3
    return sphere_volume / box_volume


def calculate_compressibility_factor(
    positions: tuple[np.ndarray, np.ndarray, np.ndarray],
    box_length: float,
    radius: float,
):
    """
    Thermodynamic observer: Accurately calculates Z by fitting a polynomial
    to a fine histogram of neighbor distances and extrapolating back to contact.
    """
    N = len(positions[0])
    TOLERANCE = 0.1

    sigma = radius
    diameter = 2.0 * sigma

    rho = N / (box_length**3)
    eta = calculate_packing_fraction(N, box_length, sigma)

    dist_matrix = generate_dist_matrix(*positions, box_length=box_length)
    rows, cols = np.triu_indices(N, k=1)
    distances = dist_matrix[rows, cols]

    shell_inner = diameter
    shell_outer = diameter + (TOLERANCE * radius)

    observed_counts = np.sum((distances >= shell_inner) & (distances <= shell_outer))

    exact_shell_volume = (4.0 / 3.0) * np.pi * (shell_outer**3 - shell_inner**3)
    expected_counts = 0.5 * N * rho * exact_shell_volume

    g_contact_estimate = (
        observed_counts / expected_counts if expected_counts > 0 else 1.0
    )

    # Calculate compressibility factor Z
    return 1.0 + 4.0 * eta * g_contact_estimate


def test_thermodynamic_ideal_gas_limit():
    """
    Thermodynamic Test 1: Validates that a highly dilute system
    behaves like an ideal gas (Z safely approaches 1.0).
    """
    L = 50.0
    N = 512
    SIGMA = 1.0
    DISP = 2.0
    sim = hpmc.SingleBoxEngine(hpmc.ParticleType.Sphere, L, 2, radius=SIGMA)
    lattice = generate_cubic_lattice(N, L, SIGMA)
    sim.set_coords(*lattice)

    assert sim.count_overlaps_naive() == 0, (
        "There should be no particle overlaps initially"
    )

    # 1. Long Equilibrate Phase: Melt the grid completely
    sim.run_batch(sweeps=1_000, max_disp=DISP)

    # 2. Production Phase: Sample the system 5 times to smooth out frame noise
    z_samples = []
    SAMPLE_BLOCKS = 5

    for i in range(SAMPLE_BLOCKS):
        sim.run_batch(sweeps=100, max_disp=DISP)

        # Take a snapshot metric at the end of this block
        current_pos = sim.get_coords()
        z_samples.append(calculate_compressibility_factor(current_pos, L, SIGMA))

    # Compute the clean ensemble average
    Z_averaged = np.mean(z_samples)

    # Use Carnahan-Starling equation of state
    eta = calculate_packing_fraction(N, L, SIGMA)
    Z_theory = (1 + eta + eta**2 - eta**3) / (1 - eta) ** 3

    assert Z_averaged == pytest.approx(Z_theory, abs=0.03), (
        f"Thermodynamic Failure: Averaged compressibility factor Z was {Z_averaged:.4f}, "
        f"expected it to converge close to the physical target of {Z_theory}."
    )
