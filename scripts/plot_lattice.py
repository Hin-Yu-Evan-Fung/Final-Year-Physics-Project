import numpy as np
import matplotlib.pyplot as plt

from hpmc.utils.lattice import generate_lattice2


def plot_lattice(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    radius: float,
    box_length: float,
) -> None:
    """Plot hard spheres inside a cubic simulation box."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 12)
    v = np.linspace(0, np.pi, 10)

    for xi, yi, zi in zip(x, y, z):
        xs = xi + radius * np.outer(np.cos(u), np.sin(v))
        ys = yi + radius * np.outer(np.sin(u), np.sin(v))
        zs = zi + radius * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(xs, ys, zs, color="steelblue")

    ax.set_xlim(0, box_length)
    ax.set_ylim(0, box_length)
    ax.set_zlim(0, box_length)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

radius = 1
box_length = 10
x,y,z = generate_lattice2(box_length, radius, "hcp")

plot_lattice(x,y,z,radius,box_length)

