import numpy as np
from hpmc.utils.lattice import generate_lattice3
from ase import Atoms
from ase.visualize import view


def plot_lattice_ase(x, y, z, box, lattice_type):
    """
    Visualise an FCC or HCP lattice using ASE.
    """

    positions = np.column_stack((x, y, z))

    atoms = Atoms(
        symbols=["Ar"] * len(x),
        positions=positions,
        cell=box,
        pbc=True
    )

    # Identify the stacking layer of each particle
    _, layer_indices = np.unique(
        np.round(z, 10),
        return_inverse=True
    )

    if lattice_type.lower() == "fcc":
        tags = layer_indices % 3
    elif lattice_type.lower() == "hcp":
        tags = layer_indices % 2
    else:
        raise ValueError("Invalid lattice type")

    atoms.set_tags(tags)

    view(atoms)

    return atoms

x, y, z, box = generate_lattice3(10, 0.5, "fcc")

atoms = plot_lattice_ase(x, y, z, box, "fcc")