import numpy as np


def generate_cubic_lattice(packing_fraction: float, radius: float, nx=6, ny=6, nz=6):
    """
    Generates coordinates for a simple cubic lattice
    with the radius, number of particles, and packing_fraction defined

    Args:
        packing_fraction: Desired packing fraction.
        radius: Radius of the hard spheres.
        nx: Number of particles per row.
        ny: Number of rows per layer (must be even).
        nz: Number of layers (must be a multiple of 6).

    Returns:
       Coordinates for a simple cubic lattice

    Raises:
        ValueError - Too many particles to generate a valid lattice
    """

    max_packing = np.pi / 6

    if not 0 < packing_fraction <= max_packing:
        raise ValueError("Invalid packing fraction.")

    if radius <= 0:
        raise ValueError("Radius must be positive.")

    # C++ Engine cell list requires nx, ny, nz >= 3
    if nx < 3 or ny < 3 or nz < 3:
        raise ValueError("Lattice dimensions are too small.")

    # Calculate lattice spacing at the desired packing fraction
    diameter = 2 * radius

    spacing = diameter * (max_packing / packing_fraction) ** (1 / 3)

    Lx = nx * spacing
    Ly = ny * spacing
    Lz = nz * spacing
    box = np.array([Lx, Ly, Lz])

    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))

    x = i * spacing
    y = j * spacing
    z = k * spacing

    return x.flatten(), y.flatten(), z.flatten(), box


def generate_mixed_lattice(packing_fraction, radius, seq, nx, ny, nz):
    """
    Generate mixed stacking lattices with periodic boundaries.

    Args:
        packing_fraction: Desired packing fraction.
        radius: Radius of the hard spheres.
        seq: Stacking sequence.
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

    # C++ Engine cell list requires nx, ny, nz >= 3
    if nx < 3 or ny < 3 or nz < 6:
        raise ValueError("Lattice dimensions are too small.")

    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError("nx, ny must be even.")

    if nz % len(seq) != 0:
        raise ValueError(f"nz must be a multiple of {len(seq)}.")

    if any(seq[i] == seq[(i + 1) % len(seq)] for i in range(len(seq))):
        raise ValueError(f"Adjacent or periodic-boundary layers match in {seq}")

    # Calculate lattice spacing at the desired packing fraction
    diameter = 2 * radius

    spacing = diameter * (max_packing / packing_fraction) ** (1 / 3)

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

    def layer_shift(t: str) -> tuple[float, float]:
        # Define layer shifts
        if t == "A":
            return (0, 0)
        if t == "B":
            return (dx / 2, dy / 3)
        if t == "C":
            return (dx, 2 * dy / 3)

        raise ValueError(f"Unrecognised layer type {t}")

    shifts = [layer_shift(t) for t in seq]

    x, y, z = [], [], []

    # Stack layers
    for k in range(nz):
        shift_x, shift_y = shifts[k % len(shifts)]

        x.append(((x_A + shift_x) % Lx).ravel())
        y.append(((y_A + shift_y) % Ly).ravel())
        z.append(np.full(x_A.size, k * dz))

    return np.concatenate(x), np.concatenate(y), np.concatenate(z), box


def generate_lattice(packing_fraction, radius, lattice_type, nx=6, ny=6, nz=6):
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

    if lattice_type.lower() == "fcc":
        return generate_mixed_lattice(packing_fraction, radius, "ABC", nx, ny, nz)
    elif lattice_type.lower() == "hcp":
        return generate_mixed_lattice(packing_fraction, radius, "AB", nx, ny, nz)
    raise ValueError("Lattice type must be 'fcc' or 'hcp'.")
