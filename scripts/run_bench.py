import argparse
import csv
import os
import time

import hpmc
from hpmc.utils.lattice import generate_cubic_lattice, generate_lattice

RADIUS = 0.5


def main():
    parser = argparse.ArgumentParser(description="hpmc Performance Benchmark")

    # fmt: off

    # Core architectural scaling parameters
    parser.add_argument(
        "-p", "--phi", type=float, default=0.20, help="Target packing fraction (density)"
    )
    parser.add_argument(
        "-t", "--lattice-type", type=str, default="fcc", choices=["sc", "fcc", "hcp"],
        help="Initial crystal lattice structure (sc, fcc, hcp)"
    )
    parser.add_argument(
        "--nx", type=int, default=6, help="Number of unit cells along X axis"
    )
    parser.add_argument(
        "--ny", type=int, default=6, help="Number of unit cells along Y axis"
    )
    parser.add_argument(
        "--nz", type=int, default=6, help="Number of unit cells along Z axis"
    )

    # Simulation engine constants
    parser.add_argument(
        "-s", "--sweeps", type=int, default=10_000, help="Number of MC sweeps"
    )
    parser.add_argument(
        "--save-csv", type=str, default="", help="Path to save results for plotting"
    )

    # fmt: on

    args = parser.parse_args()

    print(f" -> Mapping lattice structure: type={args.lattice_type}, phi={args.phi}...")

    # Assemble dynamic lattice configurations matching your test suite API
    gen_kwargs = {
        "packing_fraction": args.phi,
        "radius": RADIUS,
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
    }

    if args.lattice_type == "sc":
        x, y, z, box = generate_cubic_lattice(**gen_kwargs)
    else:
        x, y, z, box = generate_lattice(lattice_type=args.lattice_type, **gen_kwargs)

    num_particles = len(x)
    print(f" -> Initialised {num_particles:,} particles in box dimensions: {box}")

    state = hpmc.State(hpmc.ParticleType.Sphere, list(box), radius=RADIUS)
    state.set_coords(x, y, z)

    assert state.count_overlaps() == 0, "There should be no particle overlaps initially"
    assert state.n_particles() == num_particles

    print(" -> Executing C++ batch update sweep...")

    sim = hpmc.SingleBoxEngine(state)

    start_time = time.perf_counter()
    accepted = sim.run_batch(sweeps=args.sweeps, max_disp=0.1)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    moves = args.sweeps * num_particles
    throughput = moves / elapsed

    print(f" - Elapsed Time: {elapsed:.4f} seconds")
    print(f" - Sweep Velocity: {throughput:,.0f} moves/sec")
    print(f" - Accepted Moves: {accepted:,} / {moves:,}")

    # Verify sanity structural bounds
    assert isinstance(accepted, int)
    assert 0 <= accepted <= args.sweeps * num_particles

    if args.save_csv:
        file_exists = os.path.isfile(args.save_csv)
        with open(args.save_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "lattice_type",
                        "particles",
                        "packing_fraction",
                        "nx",
                        "ny",
                        "nz",
                        "sweeps",
                        "elapsed",
                        "throughput",
                    ]
                )
            writer.writerow(
                [
                    args.lattice_type,
                    num_particles,
                    args.phi,
                    args.nx,
                    args.ny,
                    args.nz,
                    args.sweeps,
                    elapsed,
                    throughput,
                ]
            )


if __name__ == "__main__":
    main()
