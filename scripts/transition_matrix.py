import matplotlib.pyplot as plt
import numpy as np
import random

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

""" Need to have small range of M, start with -2 to 2. Calculate the bias function within this range then grow range
    Loop:
        1. same as before run batches, until bias function can be calculated
        2. Calculate bias function, run again with new bias function
        3. Loop until you can calculate bias function. 
"""

# %% 1. Simulation settings (Standard result at density 0.7778)
radius = 0.5
packing_fraction = 0.5

nx, ny, nz = 6, 6, 6

min_M, max_M = -100, 100


# randomise, seed needs to be positive (unsigned)
#seed = random.randint(0,1000000)
seed = 67

sweeps_per_batch = 10000
max_disp = 0.1


# %% 2. Generates FCC and HCP states
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

engine.wl_ln_f = 0.0 # Set to non-zero to enable W-L method

macrostates = np.arange(min_M,max_M+1)

n_bins = len(macrostates)

# Tests the number of attempted moves matches sum of the values in the collection matrix, check values are valid, shape is correct

def test_collection_matrix(collection,expected_attempts):
    assert collection.shape == (n_bins, n_bins)
    assert np.all(np.isfinite(collection))
    assert np.all(collection >= 0)
    assert collection.sum() == expected_attempts
    return 0

# %% Plot collection counts and attempts per macrostate
def plot_collection(collection):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    extent = [
        min_M - 0.5, max_M + 0.5,
        min_M - 0.5, max_M + 0.5,
    ]

    image = axes[0].imshow(
        np.log1p(collection),
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )

    axes[0].set_xlabel("Destination macrostate M")
    axes[0].set_ylabel("Starting macrostate M")
    axes[0].set_title("Collection matrix")
    fig.colorbar(image, ax=axes[0], label="ln(1 + count)")

    axes[1].plot(macrostates, row_totals)
    axes[1].set_xlabel("Starting macrostate M")
    axes[1].set_ylabel("Recorded attempts")
    axes[1].set_title("Attempts per macrostate")

    fig.tight_layout()
    plt.show()

    return 0

# %% Plots the collection matrix as a heatmap
def plot_collection_heatmap(collection: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        collection,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Column index j")
    ax.set_ylabel("Row index i")
    ax.set_title("Collection matrix C[i, j]")

    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    plt.show()

# %% Check whether the bias function can be approximated from collection matrix, otherwise run another batch

def run_collection_batch() -> np.ndarray:
    engine.run_batch(
    sweeps=sweeps_per_batch,
    max_disp=max_disp,
    )

    macrostates = np.arange(min_M, max_M + 1)

    return np.array([
        [engine.collection_count(i, j) for j in macrostates]
        for i in macrostates
    ], dtype=np.float64)

# Calculates the transition matrix from collection, keeps unsampled rows as 0

def calculate_transition_matrix(collection: np.ndarray) -> np.ndarray:
    counts = np.asarray(collection, dtype=np.float64)

    if (
        counts.ndim != 2
        or counts.shape[0] != counts.shape[1]
        or counts.shape[0] == 0
    ):
        raise ValueError("Collection matrix must be nonempty and square")

    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Collection counts must be finite and nonnegative")

    row_totals = counts.sum(axis=1)

    if not np.all(np.isfinite(row_totals)):
        raise ValueError("Collection row totals are not finite")

    T = np.zeros_like(counts)

    np.divide(
        counts,
        row_totals[:, None],
        out=T,
        where=row_totals[:, None] > 0,
    )

    return T

# need to change this so that instead you can just make the transition matrix regardless of if the rows in the collection matrix is 0
def calculate_bias(collection) -> np.ndarray:
    counts = np.asarray(collection,dtype = np.float64)

    if (counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.shape[0] == 0):
        raise ValueError("Collection matrix must be nonempty and square")

    # Convert collection to transition matrix
    T = calculate_transition_matrix(collection)

    # Calculate the eigenvalues and eigenvectors of Transition matrix
    eigenvalues,eigenvectors = np.linalg.eig(T.T)

    stationary_count = np.count_nonzero(np.abs(eigenvalues - 1.0) < 1e-10)

    if stationary_count != 1:
        raise RuntimeError(
            f"Expected one stationary eigenvalue; found {stationary_count}"
        )

    # Checks eigenvalues which are very close 1. Need to further test this when increase bounds when multiple eigenvalues are close to 1
    index = np.argmin(np.abs(eigenvalues - 1)) 

    if abs(eigenvalues[index] - 1) > 1e-10:
        raise RuntimeError("Could not find stationary eigenvalue")

    # Choose probability eigenvector and normalise
    vector = np.real_if_close(eigenvectors[:,index])
    total = vector.sum()

    if not np.isfinite(total) or total == 0:
        raise RuntimeError("Cannot normalise stationary eigenvector")
    
    probabilities = vector / total

    if not np.isfinite(total) or total == 0:
        raise RuntimeError("Cannot normalise stationary eigenvector")

    residual = np.linalg.norm(probabilities @ T - probabilities, ord = 1)

    if residual > 1e-10:
        raise RuntimeError("Stationary probabilities failed validation")

    if np.iscomplexobj(probabilities):
        raise RuntimeError("Stationary probabilities must be real")

    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0)):
        raise RuntimeError("Stationary probabilities must be finite and positive")
    
    # Match the engine's exp(-(bias_new - bias_old)) acceptance rule
    bias = np.log(probabilities)

    return bias - np.max(bias)
    
max_batches = 1000
bias_updates = 0
bias_history = []
rows_sampled = False
M_increment = 2
maximum_M = 100
# rows = len(collection.sum(axis = 1))
converged = False


# Main loop
# T = calculate_transition_matrix(collection)

# sampled = collection.sum(axis=1) > 0

# assert np.allclose(T[sampled].sum(axis=1), 1.0)
# assert np.all(T[~sampled] == 0.0)

# print(f"Sampled rows: {np.count_nonzero(sampled)}/{len(T)}")

    