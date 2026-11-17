import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import hpmc
from hpmc.sim import Sim, SimConfig
from hpmc.utils.lattice import generate_mixed_lattice

# Used Google Gemini for data class generation


@dataclass
class SystemConfig:
    radius: float = 0.5
    rho: float = 0.7778
    nx: int = 6
    ny: int = 6
    nz: int = 6

    @property
    def pack_frac(self) -> float:
        return self.rho * np.sqrt(2) * (4 / 3) * np.pi * self.radius**3

    @property
    def n_particles(self) -> int:
        return self.nx * self.ny * self.nz


@dataclass
class LatticeConfig:
    sequence: str = "ABC"


@dataclass
class LSMCConfig:
    min_m: int = -130
    max_m: int = 130
    max_disp: float = 0.1
    initial_ln_f: float = 1.0
    batches: int = 50_000
    sweeps_per_batch: int = 1_000

    @property
    def m_off(self) -> int:
        return -self.min_m


@dataclass
class OutputConfig:
    print_int: int = 10
    sample_int: int = 500
    plot_from: int = 2000
    e_scale: float = 1e5
    min_e: float = -500.0
    max_e: float = 500.0
    benchmark_e: float | None = 133.0


@dataclass
class Config:
    system: SystemConfig
    lattice_1: LatticeConfig
    lattice_2: LatticeConfig
    lsmc: LSMCConfig
    output: OutputConfig

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        cfg = cls(
            system=SystemConfig(**raw["system"]),
            lattice_1=LatticeConfig(**raw["lattice_1"]),
            lattice_2=LatticeConfig(**raw["lattice_2"]),
            lsmc=LSMCConfig(**raw["lsmc"]),
            output=OutputConfig(**raw["output"]),
        )
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def build_sphere_state(s: SystemConfig, l: LatticeConfig) -> hpmc.State:
    lat = generate_mixed_lattice(s.pack_frac, s.radius, l.sequence, s.nx, s.ny, s.nz)
    state = hpmc.State(hpmc.ParticleType.Sphere, list(lat[3]), radius=s.radius)
    state.set_coords(*lat[:3])
    return state


def run_sim(
    seed: int,
    base_dir: str,
    config: Config,
) -> None:
    fcc = build_sphere_state(config.system, config.lattice_1)
    hcp = build_sphere_state(config.system, config.lattice_2)
    assert fcc.n_particles() == hcp.n_particles()

    cfg = SimConfig(config.lsmc.sweeps_per_batch, config.lsmc.batches)
    engine = hpmc.LSMCEngine(fcc, hcp, config.lsmc.min_m, config.lsmc.max_m, seed=seed)
    engine.wl_ln_f = 1.0

    outdir = os.path.join(base_dir, f"seed_{seed}")

    with Sim(engine, cfg, outdir=outdir) as sim:
        for batch in range(1, cfg.batches + 1):
            engine.run_batch(sweeps=cfg.sweeps_per_batch, max_disp=config.lsmc.max_disp)
            sim.normalize_bias_weights()
            sim.record_batch(config.lsmc.m_off, config.system.n_particles)

            if sim.is_visit_counts_flat():
                sim.advance_wl_stage(batch)
                sim.console.log(
                    "[bold green][CONVERGED][/] ln(f) dropped to [bold white]"
                    f"{sim.lsmc.wl_ln_f:.10f}[/] at {batch * cfg.sweeps_per_batch} sweeps."
                )

            if (
                batch > config.output.plot_from
                and batch % config.output.sample_int == 0
            ):
                bench = config.output.benchmark_e
                hline = (
                    (
                        bench,
                        f"Literature Benchmark ({bench})",
                    )
                    if bench is not None
                    else None
                )

                filename = sim.save_sample_plot(
                    plot_from_batch=config.output.plot_from,
                    batch=batch,
                    e_scale=config.output.e_scale,
                    min_e=config.output.min_e,
                    max_e=config.output.max_e,
                    hline=hline,
                )
                sim.console.log(
                    f"[bold red][SAVED][/] Saved energy estimate graph to {filename}"
                )

            if batch % config.output.print_int == 0:
                sim.update_log(batch)

        sim.console.print("\n[bold green]All batches completed successfully![/]")


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
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Config filename"
    )

    args = parser.parse_args()

    # Make output directory
    dirname = f"{args.outdir}/"
    os.makedirs(dirname, exist_ok=True)

    # Parse config from json
    config = Config.from_json(args.config)

    run_sim(args.seed, dirname, config)
