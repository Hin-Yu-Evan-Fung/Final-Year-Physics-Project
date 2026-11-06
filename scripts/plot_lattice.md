---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: .venv
    language: python
    name: .venv
---

```python
import numpy as np

from hpmc.utils.lattice import generate_lattice
from ase import Atoms
from ase.visualize import view
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors
from IPython.display import display
```

```python
def plot_lattice_ase(x, y, z, box, radius, lattice_type):
    """
    Display FCC or HCP lattice interactively using ASE.

    Spheres are drawn with the specified hard-sphere radius.
    A, B and C stacking layers have different colours.
    """

    positions = np.column_stack((x, y, z))

    # Create the actual lattice
    atoms = Atoms(
        symbols=["Ar"] * len(x),
        positions=positions,
        cell=box,
        pbc=True
    )

    # Determine the stacking sequence
    lattice_type = lattice_type.lower()

    if lattice_type == "fcc":
        n_layers = 3
    elif lattice_type == "hcp":
        n_layers = 2
    else:
        raise ValueError("Lattice type must be 'fcc' or 'hcp'.")

    # Identify each particle's layer
    _, layer_indices = np.unique(
        np.round(z, 10),
        return_inverse=True
    )

    tags = layer_indices % n_layers
    atoms.set_tags(tags)

    # Use different elements as visual placeholders for A, B, C
    symbols = ["H", "He", "Li"][:n_layers]
    colors = [
        (0.2, 0.4, 0.9),   # A: Blue
        (1.0, 0.5, 0.1),   # B: Orange
        (0.2, 0.8, 0.3)    # C: Green
    ]

    display_atoms = atoms.copy()
    display_atoms.set_chemical_symbols(
        [symbols[tag] for tag in tags]
    )

    # Set the correct display radii and colours
    numbers = [atomic_numbers[s] for s in symbols]

    original_radii = covalent_radii[numbers].copy()
    original_colors = jmol_colors[numbers].copy()

    try:
        covalent_radii[numbers] = radius
        jmol_colors[numbers] = colors[:n_layers]

        display(view(display_atoms, viewer="x3d"))

    finally:
        # Restore ASE's original element properties
        covalent_radii[numbers] = original_radii
        jmol_colors[numbers] = original_colors

    return atoms
```

```python
# Simulation parameters
packing_fraction = 0.6
radius = 0.5

# Generate FCC
x_fcc, y_fcc, z_fcc, box_fcc = generate_lattice(
    packing_fraction, radius, "fcc"
)

# Generate HCP
x_hcp, y_hcp, z_hcp, box_hcp = generate_lattice(
    packing_fraction, radius, "hcp"
)
```

```python
# Visualise FCC
print("FCC Lattice")
fcc_atoms = plot_lattice_ase(
    x_fcc, y_fcc, z_fcc, box_fcc, radius, "fcc"
)


# Visualise HCP
print("\nHCP Lattice")
hcp_atoms = plot_lattice_ase(
    x_hcp, y_hcp, z_hcp, box_hcp, radius, "hcp"
)

```

```python

from ase.io import write
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors


def export_lattice_html(atoms, radius, filename, lattice_type):
    """Export an interactive 3D lattice as an HTML file."""

    lattice_type = lattice_type.lower()

    if lattice_type == "fcc":
        symbols = ["H", "He", "Li"]
    elif lattice_type == "hcp":
        symbols = ["H", "He"]
    else:
        raise ValueError("Lattice type must be 'fcc' or 'hcp'.")

    # Use layer tags to assign display colours
    tags = atoms.get_tags()

    display_atoms = atoms.copy()
    display_atoms.set_chemical_symbols(
        [symbols[tag] for tag in tags]
    )

    colors = np.array([
        [0.2, 0.4, 0.9],
        [1.0, 0.5, 0.1],
        [0.2, 0.8, 0.3]
    ])

    numbers = [atomic_numbers[s] for s in symbols]

    original_radii = covalent_radii[numbers].copy()
    original_colors = jmol_colors[numbers].copy()

    try:
        covalent_radii[numbers] = radius
        jmol_colors[numbers] = colors[:len(symbols)]

        write(filename, display_atoms, format="html")

    finally:
        covalent_radii[numbers] = original_radii
        jmol_colors[numbers] = original_colors

    print(f"Exported: {filename}")


# Export your existing lattices
export_lattice_html(fcc_atoms, radius, "fcc.html", "fcc")
export_lattice_html(hcp_atoms, radius, "hcp.html", "hcp")

```

```python

```
