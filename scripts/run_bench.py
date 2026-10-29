import argparse
import csv
import os
import time

import hpmc
from hpmc.utils.lattice import generate_cubic_lattice


def main():
    parser = argparse.ArgumentParser(description="hpmc Performance Benchmark")
    parser.add_argument(
        "-n", "--particles", type=int, default=256, help="Number of particles"
    )
    parser.add_argument(
        "-s", "--sweeps", type=int, default=10_000, help="Number of MC sweeps"
    )
    parser.add_argument(
        "-l", "--box-length", type=float, default=20.0, help="Box side length (L)"
    )
    parser.add_argument(
        "-r", "--radius", type=float, default=0.5, help="Hard sphere radius (sigma)"
    )
    parser.add_argument(
        "--save-csv", type=str, default="", help="Path to save results for plotting"
    )

    args = parser.parse_args()

    box_dims = [args.box_length] * 3
    state = hpmc.State(hpmc.ParticleType.Sphere, box_dims, radius=args.radius)

    print(f" -> Generating {args.particles:,} random particle coordinates...")

    # Evenly space them or scatter them randomly across the box volume
    init_coords = generate_cubic_lattice(args.particles, args.box_length, args.radius)
    state.set_coords(*init_coords)

    assert state.count_overlaps() == 0, "There should be no particle overlaps initially"
    assert state.n_particles() == args.particles

    print(" -> Executing C++ batch update sweep...")

    sim = hpmc.SingleBoxEngine(state)

    start_time = time.perf_counter()
    accepted = sim.run_batch(sweeps=args.sweeps, max_disp=0.1)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    moves = args.sweeps * args.particles
    throughput = moves / elapsed
    density = args.particles / (args.box_length**3)

    print(f" - Elapsed Time: {elapsed:.4f} seconds")
    print(f" - Sweep Velocity: {throughput:,.0f} moves/sec")
    print(f" - Accepted Moves: {accepted:,} / {moves:,}")

    # Verify sanity structural bounds
    assert isinstance(accepted, int)
    assert 0 <= accepted <= args.sweeps * args.particles

    if args.save_csv:
        file_exists = os.path.isfile(args.save_csv)
        with open(args.save_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "particles",
                        "box_length",
                        "density",
                        "sweeps",
                        "elapsed",
                        "throughput",
                    ]
                )
            writer.writerow(
                [
                    args.particles,
                    args.box_length,
                    density,
                    args.sweeps,
                    elapsed,
                    throughput,
                ]
            )


if __name__ == "__main__":
    main()
