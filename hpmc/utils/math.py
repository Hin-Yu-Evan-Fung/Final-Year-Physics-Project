from collections.abc import Sequence

import numpy as np


def generate_dist_matrix(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, box_dims: Sequence[float]
) -> np.ndarray:
    """
    Computes a full N x N pairwise distance matrix adhering to
    Periodic Boundary Conditions (Minimum Image Convention).

    Args:
        positions: Array of shape (N, 3) containing particle coordinates.
        box_dims: The dimension of the periodic simulation box.

    Returns:
        A symmetric (N, N) matrix where entry [i, j] is the periodic distance
        between particle i and particle j.
    """
    # Force coordinates to numpy arrays and turn them into column vectors (N, 1)
    # The [:, None] syntax expands the shape from (N,) to (N, 1)
    x_col = np.asarray(x, dtype=np.float64)[:, None]
    y_col = np.asarray(y, dtype=np.float64)[:, None]
    z_col = np.asarray(z, dtype=np.float64)[:, None]

    # Broadcasting (N, 1) - (1, N) creates an (N, N) delta matrix for each axis
    dx = x_col - x_col.T
    dy = y_col - y_col.T
    dz = z_col - z_col.T

    # Minimum Image Convention
    dx -= box_dims[0] * np.round(dx / box_dims[0])
    dy -= box_dims[1] * np.round(dy / box_dims[1])
    dz -= box_dims[2] * np.round(dz / box_dims[2])

    # Compute Euclidean norm along the coordinate axis
    return np.sqrt(dx**2 + dy**2 + dz**2)


def log_sum_exp(log_arr: np.ndarray) -> float:
    """Stable log sum exp for free energy difference calculation"""
    x_max = np.max(log_arr)
    return float(x_max + np.log(np.sum(np.exp(log_arr - x_max))))


def calculate_free_energy(
    visit_counts: np.ndarray,
    bias_weights: np.ndarray,
    m_off: int,
    n_particles: int,
) -> float:
    """Calculates free energy difference based on visit counts and bias weights"""

    counts = np.array(visit_counts, dtype=np.float64, copy=True)
    biases = np.array(bias_weights, dtype=np.float64, copy=True)
    ln_g = np.log(counts) + biases

    # Calculate log(Z) directly to avoid underflow/overflow issues
    log_Z_fcc = log_sum_exp(ln_g[:m_off])
    log_Z_hcp = log_sum_exp(ln_g[m_off + 1 :])

    # delta_F = -ln(Z_hcp / Z_fcc)
    return -(log_Z_hcp - log_Z_fcc) / n_particles


def sphere_density_to_pack_frac(reduced_density: float, radius: float) -> float:
    """Calculate packing fraction from reduced density"""
    return reduced_density * np.sqrt(2) * (4 / 3) * np.pi * radius**3
