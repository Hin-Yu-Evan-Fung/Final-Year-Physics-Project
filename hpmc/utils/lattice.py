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

    grid_points = np.linspace(
        radius, box_length - radius, points_per_axis, dtype=np.float64
    )

    x, y, z = np.meshgrid(grid_points, grid_points, grid_points, indexing="ij")

    return (
        np.ravel(x)[:num_particles],
        np.ravel(y)[:num_particles],
        np.ravel(z)[:num_particles],
    )



import numpy as np


def generate_lattice(
    packing_fraction,
    radius,
    lattice_type,
    nx=6,
    ny=6,
    nz=6
):
    """
    Generate FCC or HCP lattice with periodic boundaries.

    Args:
        packing_fraction: Desired packing fraction.
        radius: Radius of the hard spheres.
        lattice_type: "fcc" or "hcp".
        nx: Number of particles per row.
        ny: Number of rows per layer (must be even).
        nz: Number of layers (must be a multiple of 6).

    Returns:
        x, y, z: Separate NumPy coordinate arrays.
        box: Periodic box dimensions [Lx, Ly, Lz].
    """

    max_packing = np.pi / (3 * np.sqrt(2))

    if not 0 < packing_fraction <= max_packing:
        raise ValueError("Invalid packing fraction.")

    if radius <= 0:
        raise ValueError("Radius must be positive.")

    if nx < 2 or ny < 2 or nz < 6:
        raise ValueError("Lattice dimensions are too small.")

    if ny % 2 != 0 or nz % 6 != 0:
        raise ValueError("ny must be even and nz a multiple of 6.")

    # Calculate lattice spacing at the desired packing fraction
    diameter = 2 * radius

    spacing = diameter * (max_packing / packing_fraction)**(1 / 3)

    dx = spacing
    dy = np.sqrt(3) / 2 * spacing
    dz = np.sqrt(2 / 3) * spacing

    # Calculate periodic box dimensions
    Lx = nx * dx
    Ly = ny * dy
    Lz = nz * dz

    box = np.array([Lx, Ly, Lz])

    # Generate triangular-packed A layer
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))

    x_A = (i + 0.5 * (j % 2)) * dx
    y_A = j * dy

    # Define layer shifts
    A = (0, 0)
    B = (dx / 2, dy / 3)
    C = (dx, 2 * dy / 3)

    if lattice_type.lower() == "fcc":
        shifts = [A, B, C]
    elif lattice_type.lower() == "hcp":
        shifts = [A, B]
    else:
        raise ValueError("Lattice type must be 'fcc' or 'hcp'.")

    x, y, z = [], [], []

    # Stack layers
    for k in range(nz):
        shift_x, shift_y = shifts[k % len(shifts)]

        x.append(((x_A + shift_x) % Lx).ravel())
        y.append(((y_A + shift_y) % Ly).ravel())
        z.append(np.full(x_A.size, k * dz))

    return np.concatenate(x), np.concatenate(y), np.concatenate(z), box


