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


def generate_lattice(radius = 1, box_length = 10, structure = "fcc"):

    diameter = 2 * radius

    a1 = np.array([diameter,0])
    a2 = np.array([diameter/2,diameter*(np.sqrt(3)/2)])

    layer_spacing = np.sqrt(2/3) * diameter

    # The coordinate shift from the layer below:

    shifts = [np.zeros(3),  # A
              (a1 + a2)/3,  # B
              2*(a1 + a2)/3 # C
              ]
    
    if structure == "fcc":
        period = 3
    else:
        period = 2

    positions = []
    k = 0 # layer number in z direction

    while radius + k * layer_spacing < box_length - radius:
        sx,sy = shifts[k%period]
        z = radius + k * layer_spacing

        j = 0 # layer number but in y direction
        while radius + j * a2[1] + sy < box_length - radius:
            y = radius + j*a2[1] + sy

            x_start = radius + (j%2) * radius + sx

            for x in np.arange(x_start,box_length - radius - 1e-10,diameter):
                positions.append((x,y,z))

            j+=1

        k+=1

    x,y,z = np.array(positions)

    return x,y,z


import numpy as np


def generate_lattice2(box_length, radius, lattice_type):
    """
    Generate FCC or HCP lattice coordinates inside a cubic box.

    Args:
        box_length: Side length of the cubic box.
        radius: Radius of each hard sphere.
        lattice_type: "fcc" or "hcp".

    Returns:
        x, y, z: Separate NumPy arrays of particle coordinates.
    """
    if box_length <= 0 or radius <= 0:
        raise ValueError("Box length and radius must be positive.")

    positions = []
    diameter = 2 * radius

    if lattice_type.lower() == "fcc":
        spacing = np.sqrt(2) * diameter

        basis = [
            (0, 0, 0),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ]

        n = int(np.ceil(box_length / spacing)) + 1

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for dx, dy, dz in basis:
                        x = (i + dx) * spacing
                        y = (j + dy) * spacing
                        z = (k + dz) * spacing

                        if x < box_length and y < box_length and z < box_length:
                            positions.append((x, y, z))

    elif lattice_type.lower() == "hcp":
        spacing_x = diameter
        spacing_y = np.sqrt(3) / 2 * diameter
        spacing_z = np.sqrt(2 / 3) * diameter

        nx = int(np.ceil(box_length / spacing_x)) + 1
        ny = int(np.ceil(box_length / spacing_y)) + 1
        nz = int(np.ceil(box_length / spacing_z)) + 1

        for k in range(nz):
            for j in range(ny):
                for i in range(-1, nx):
                    x = (i + 0.5 * (j % 2)) * spacing_x
                    y = j * spacing_y
                    z = k * spacing_z

                    if k % 2 == 1:
                        x += spacing_x / 2
                        y += spacing_y / 3

                    if 0 <= x < box_length and 0 <= y < box_length and z < box_length:
                        positions.append((x, y, z))

    else:
        raise ValueError("Lattice type must be 'fcc' or 'hcp'.")

    positions = np.array(positions).reshape(-1, 3)

    return positions[:, 0], positions[:, 1], positions[:, 2]