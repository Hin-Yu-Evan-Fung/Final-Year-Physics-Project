import argparse
import os

import numpy as np

import hpmc
from hpmc.sim import Sim, SimConfig
from hpmc.utils.lattice import generate_lattice

RADIUS = 0.5
NX, NY, NZ = 6, 6, 6
N = NX * NY * NZ

RHO = 0.7778
PACK_FRAC = RHO * np.sqrt(2) * (4 / 3) * np.pi * RADIUS**3

MIN_M = -130
M_OFF = -MIN_M
MAX_M = 130
MAX_DISP = 0.1

BATCHES = 50_000
SWEEPS_PER_BATCH = 1_000

PRINT_INT = 10
SAMPLE_INT = 500
PLOT_FROM = 2000
E_SCALE = 1e5
MIN_E = -500
MAX_E = 500


def build_sphere_state(lattice_type: str) -> hpmc.State:
    lat = generate_lattice(PACK_FRAC, RADIUS, lattice_type, NX, NY, NZ)
    state = hpmc.State(hpmc.ParticleType.Sphere, list(lat[3]), radius=RADIUS)
    state.set_coords(*lat[:3])
    return state


def run_sim(
    seed: int,
    base_dir: str,
) -> None:
    fcc = build_sphere_state("fcc")
    hcp = build_sphere_state("hcp")
    assert fcc.n_particles() == hcp.n_particles()

    cfg = SimConfig(SWEEPS_PER_BATCH, BATCHES)
    engine = hpmc.LSMCEngine(fcc, hcp, MIN_M, MAX_M, seed=seed)
    engine.wl_ln_f = 1.0

    outdir = os.path.join(base_dir, f"seed_{seed}")

    with Sim(engine, cfg, outdir=outdir) as sim:
        for batch in range(1, cfg.batches + 1):
            engine.run_batch(sweeps=cfg.sweeps_per_batch, max_disp=MAX_DISP)
            sim.normalize_bias_weights()
            sim.record_batch(M_OFF, N)

            if sim.is_visit_counts_flat():
                sim.advance_wl_stage(batch)
                sim.console.log(
                    "[bold green][CONVERGED][/] ln(f) dropped to [bold white]"
                    f"{sim.lsmc.wl_ln_f:.10f}[/] at {batch * cfg.sweeps_per_batch} sweeps."
                )

            if batch > PLOT_FROM and batch % SAMPLE_INT == 0:
                filename = sim.save_sample_plot(
                    plot_from_batch=PLOT_FROM,
                    batch=batch,
                    e_scale=E_SCALE,
                    min_e=MIN_E,
                    max_e=MAX_E,
                    hline=(133, "Literature Benchmark (133)"),
                )
                sim.console.log(
                    f"[bold red][SAVED][/] Saved energy estimate graph to {filename}"
                )

            if batch % PRINT_INT == 0:
                sim.update_log(batch)

        sim.console.print("\n[bold green]All batches completed successfully![/]")
        sim.save_summary_plots()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", "-s", type=int, default=1, help="RNG seed for this replica"
    )
    parser.add_argument(
        "--outdir",
        "-d",
        type=str,
        required=True,
        help="Output directory (outputs in outdir/seed_[seed]/)",
    )
    args = parser.parse_args()
    dirname = f"{args.outdir}/"
    os.makedirs(dirname, exist_ok=True)
    run_sim(args.seed, dirname)
