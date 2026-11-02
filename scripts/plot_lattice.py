import numpy as np
from hpmc.utils.lattice import generate_lattice
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

    view(atoms, viewer="ase", block=True)

    return atoms

packing_fraction = 0.60
radius = 0.5

x_fcc, y_fcc, z_fcc, box_fcc = generate_lattice(
    packing_fraction, radius, "fcc"
)

x_hcp, y_hcp, z_hcp, box_hcp = generate_lattice(
    packing_fraction, radius, "hcp"
)

fcc_atoms = plot_lattice_ase(
    x_fcc, y_fcc, z_fcc, box_fcc, "fcc"
)

hcp_atoms = plot_lattice_ase(
    x_hcp, y_hcp, z_hcp, box_hcp, "hcp"
)

# Open both structures in the interactive ASE viewer
view([fcc_atoms, hcp_atoms], viewer="ase", block=True)
