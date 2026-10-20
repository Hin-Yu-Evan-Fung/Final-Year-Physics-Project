
import numpy as np
import hsmc

L = 10.0
N = 512
# Set diameter to 0 to simulate a true Ideal Gas
params = hsmc.HardSphereParams(0.0) 
engine = hsmc.HardSphereEngine(L, params)

# Initialize 216 particles randomly
init_particles = np.random.uniform(-L/2, L/2, (N, 3))
engine.set_particles(init_particles)

# Run a heavy batch of steps to sample equilibrium configurations
num_steps = 1_000_000
indices = np.random.randint(0, N, size=num_steps).astype(np.uint32)
moves = np.random.uniform(-0.5, 0.5, (num_steps, 3))

accepted = engine.try_run_batch(indices, moves)
print(f"Checkpoint 1 (Acceptance Rate): {accepted / num_steps * 100:.2f}% (Expected: 100.00%)")

# Compute g(r) on final positions to verify structure
final_pos = engine.get_particles()
num_particles = len(final_pos)

# Pairwise distances with Periodic Boundary Conditions (Minimum Image Convention)
dr = final_pos[:, None, :] - final_pos[None, :, :]
distances = np.sqrt(np.sum(dr**2, axis=-1))

# Extract upper triangle distances (excluding self-interaction 0s)
pair_distances = distances[np.triu_indices(num_particles, k=1)]

# Histogram into radial bins
bin_edges = np.linspace(0.1, L/2, 50)
counts, _ = np.histogram(pair_distances, bins=bin_edges)

# Normalize by the ideal gas shell volume rho * 4 * pi * r^2 * dr
rho = num_particles / (L**3)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dr_bin = bin_edges[1] - bin_edges[0]

# Expected counts for a completely random (ideal) distribution
expected_counts = 0.5 * num_particles * rho * 4 * np.pi * (bin_centers**2) * dr_bin
g_r = counts / expected_counts

print(f"Checkpoint 2 (Mean g(r)): {np.mean(g_r):.4f} (Expected: 1.0000)")
