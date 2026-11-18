import matplotlib.pyplot as plt
import numpy as np

import hpmc
from hpmc.utils.lattice import generate_lattice

""" 1. Define simulation settings, use same as WL or paper for standard result
    2. Generate FCC and HCP states
    3. Create the LSMC engine starting with 0 bias. The engine will collect transition statistics from 
    the hard sphere moves
    4. Collect each monte carlo move in the collection matrix over a batch run plotting heatmap and row totals
    5. Apply tests created on the collection matrix (makes sure the total number of moves matches the entries in the collection matrix)
    6. Check whether a bias function can be approximated from the matrix, otherwise return error and keep exploring.
    7. If possible convert the collection matrix into the transition matrix. Check all the rows sum to 1.
    8. Calculate the stationary probability distribution by calculating the right eigenvector of the transpose of the transition matrix. (need to normalise aswell)
    9. Calculate and plot the bias using b(M) = lnP(M) + c, plot the bias against M
    10. Apply the bias and collect another batch, i.e repeat from step 4
    11. Assess stability of the bias function then freeze the bias once it has converged within a margin of error.
"""

# %% 1. Simulation settings
radius = 0.5
packing_fraction = 0.5

nx, ny, nz = 6, 6, 6

min_M, max_M = -150, 150
seed = 7

sweeps_per_batch = 100
max_disp = 0.1


# %% 2. Generate FCC and HCP states
def make_state(lattice_type: str) -> hpmc.State:
    x, y, z, box = generate_lattice(
        packing_fraction,
        radius,
        lattice_type,
        nx,
        ny,
        nz,
    )

    state = hpmc.State(
        hpmc.ParticleType.Sphere,
        box,
        radius=radius,
    )
    state.set_coords(x, y, z)

    assert state.count_overlaps() == 0, (
        f"Initial {lattice_type.upper()} configuration contains overlaps"
    )

    return state


fcc = make_state("fcc")
hcp = make_state("hcp")

n_particles = fcc.n_particles()

assert n_particles == hcp.n_particles()
assert n_particles == nx * ny * nz


# %% 3. Create the LSMC engine
engine = hpmc.LSMCEngine(
    fcc,
    hcp,
    min_M,
    max_M,
    seed=seed,
)

engine.wl_ln_f = 0.0

n_bins = max_M - min_M + 1
engine.set_bias_weights(np.zeros(n_bins, dtype=np.float64))

print(f"Particles per configuration: {n_particles}")
print(f"Macrostate range: {min_M} to {max_M} ({n_bins} bins)")
print(f"Initial macrostate: {engine.current_M}")
print(f"Wang–Landau increment: {engine.wl_ln_f}")