import numpy as np


def generate_cubic_lattice(
    num_particles: int, box_length: float, radius: float
) -> np.ndarray:
    """
    Generates coordinates for a simple cubic lattice
    with the number of particles and box length provided.

    Args:
        num_particles: Total count of spheres to place.
        box_length: The linear side length of the simulation box.
        radius: The radius of a single sphere (Diameter = 2 * radius).

    Returns:
       Coordinates for a simple cubic lattice

    Raises:
        ValueError - Too many particles to generate a valid lattice
    """
    points_per_axis = int(np.ceil(np.pow(num_particles, (1 / 3))))
    diameter = radius * 2

    # Calculate the linear footprint of the required grid
    required_length = points_per_axis * diameter

    if required_length > box_length:
        raise ValueError(
            f"Invalid Lattice Configuration! A {points_per_axis}x{points_per_axis}x{points_per_axis} "
            f"grid requires a linear spacing of at least {required_length:.4f} units, "
            f"but your box_length is only {box_length:.4f}."
        )

    grid_points = np.linspace(radius, box_length - radius, points_per_axis)

    x, y, z = np.meshgrid(grid_points, grid_points, grid_points, indexing="ij")

    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    return positions[:num_particles]
