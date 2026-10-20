import numpy as np


def generate_dist_matrix(positions: np.ndarray, box_length: float) -> np.ndarray:
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
    # Broad-casted subtraction calculates raw coordinate deltas (N, N, 3)
    delta = positions[:, None, :] - positions[None, :, :]

    # Minimum Image Convention: Wrap differences to [-box_length/2, box_length/2]
    delta = delta - box_length * np.round(delta / box_length)

    # Compute Euclidean norm along the coordinate axis
    return np.linalg.norm(delta, axis=-1)
