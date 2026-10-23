import numpy as np


def generate_dist_matrix(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, box_length: float
) -> np.ndarray:
    """
    Computes a full N x N pairwise distance matrix adhering to
    Periodic Boundary Conditions (Minimum Image Convention).

    Args:
        positions: Array of shape (N, 3) containing particle coordinates.
        box_length: The linear size of the periodic simulation box.

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

    # Minimum Image Convention: Wrap differences to [-box_length/2, box_length/2]
    dx -= box_length * np.round(dx / box_length)
    dy -= box_length * np.round(dy / box_length)
    dz -= box_length * np.round(dz / box_length)

    # Compute Euclidean norm along the coordinate axis
    return np.sqrt(dx**2 + dy**2 + dz**2)
