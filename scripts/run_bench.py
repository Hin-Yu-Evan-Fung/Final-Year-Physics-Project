import argparse
import csv
import hpmc
from hpmc.utils.lattice import generate_cubic_lattice
import numpy as np
import time
import os


def main():
    parser = argparse.ArgumentParser(description="hpmc Performance Benchmark")
    parser.add_argument("-n", "--particles", type=int, default=256, help="Number of particles")
    parser.add_argument("-m", "--moves", type=int, default=2_000_000, help="Number of MC moves")
    parser.add_argument("-l", "--box-length", type=float, default=20.0, help="Box side length (L)")
    parser.add_argument("-r", "--radius", type=float, default=0.5, help="Hard sphere radius (sigma)")
    parser.add_argument("--save-csv", type=str, default="", help="Path to save results for plotting")

    args = parser.parse_args()
    
    params = hpmc.HardSphereParams(args.radius)
    sim = hpmc.HardSphereSimulation(L=args.box_length, params=params)

    print(f" -> Generating {args.particles:,} random particle coordinates...")

    # Evenly space them or scatter them randomly across the box volume
    init_pos = generate_cubic_lattice(args.particles, args.box_length, args.radius)
    sim.set_particles(init_pos)

    assert sim.count_overlaps() == 0, "There should be no particle overlaps initially"
    assert sim.n_particles() == args.particles

    print(f" -> Generating {args.moves:,} random trial steps...")

    moves = np.random.uniform(-0.1, 0.1, size=(args.moves, 3)).astype(np.float64)
    indices = np.random.randint(0, args.particles, size=args.moves, dtype=np.uint32)

    print(" -> Executing C++ batch update sweep...")

    start_time = time.perf_counter()
    accepted = sim.try_run_batch(indices=indices, moves=moves)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    throughput = args.moves / elapsed
    density = args.particles / (args.box_length ** 3)

    print(f" - Elapsed Time: {elapsed:.4f} seconds")
    print(f" - Sweep Velocity: {throughput:,.0f} moves/sec")
    print(f" - Accepted Moves: {accepted:,} / {args.moves:,}")

    # Verify sanity structural bounds
    assert isinstance(accepted, int)
    assert 0 <= accepted <= args.moves

    if args.save_csv:
        file_exists = os.path.isfile(args.save_csv)
        with open(args.save_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["particles", "box_length", "density", "moves", "elapsed", "throughput"])
            writer.writerow([args.particles, args.box_length, density, args.moves, elapsed, throughput])

if __name__ == "__main__":
    main()
