import os
import time

import matplotlib

# Force Matplotlib to run in the background (no window popups/event loops)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

import hpmc
from hpmc.utils.lattice import generate_lattice

console = Console()

# Used Google Search / Gemini to generate some of the progress bar related code below


class Simulation:
    def __init__(self) -> None:
        self.radius = 0.5
        self.nx, self.ny, self.nz = 6, 6, 6
        self.n = self.nx * self.ny * self.nz

        self.reduced_density = 0.7778
        self.packing_fraction = (
            self.reduced_density * np.sqrt(2) * (4 / 3) * np.pi * self.radius**3
        )

        # LSMC engine config
        self.min_m = -130
        self.max_m = 130
        self.m_off = -self.min_m
        self.sweeps = 1_000  # sweeps per batch
        self.batches = 50_000
        self.seed = 1

        # Log config
        self.print_interval = 10

        # Free energy estimate plot config
        self.sample_interval = 500  # in batches
        self.plot_from_batch = 1400
        self.e_scale = 1e5
        self.max_e = 500  # in 10^5 k_BT
        self.min_e = -500  # in 10^5 k_BT

        # Checkpoint
        current_time_str = time.strftime("%Y%m%d-%H%M%S")
        self.outdir = f"./run_{current_time_str}"
        os.makedirs(self.outdir, exist_ok=True)

        fcc, hcp = self.generate_lattices()
        self.lsmc = hpmc.LSMCEngine(fcc, hcp, self.min_m, self.max_m, seed=self.seed)
        self.lsmc.wl_ln_f = 1.0
        self.delta_F_hist = []
        self.total_visit_counts = np.zeros_like(self.lsmc.visit_counts, dtype=np.uint64)
        self.delta_F = self.delta_F_hist[-1] if self.delta_F_hist else 0

        self.start_time = time.perf_counter()
        self.batches_executed = 0

    def save_results(self):
        np.savez_compressed(
            f"{self.outdir}/results.npz",
            wl_ln_f=self.lsmc.wl_ln_f,
            delta_F=self.delta_F,
            delta_F_hist=np.array(self.delta_F_hist),
            total_visit_counts=np.array(self.total_visit_counts),
            batches_executed=self.batches_executed,
        )

    def calculate_free_energy(self, visit_counts: np.ndarray, bias_weights: np.ndarray):
        counts = np.array(visit_counts, dtype=np.float64, copy=True)
        biases = np.array(bias_weights, dtype=np.float64, copy=True)

        ln_g = np.log(counts) + biases

        def log_sum_exp(log_arr):
            x_max = np.max(log_arr)
            return x_max + np.log(np.sum(np.exp(log_arr - x_max)))

        # Calculate log(Z) directly to avoid underflow/overflow issues
        log_Z_fcc = log_sum_exp(ln_g[: self.m_off])
        log_Z_hcp = log_sum_exp(ln_g[self.m_off + 1 :])

        # delta_F = -ln(Z_hcp / Z_fcc)
        delta_F = -(log_Z_hcp - log_Z_fcc)

        return delta_F / self.n

    def generate_lattices(self) -> tuple[hpmc.State, hpmc.State]:
        """
        Generates FCC and HCP lattices with a specific packing fraction, nx, ny, and nz
        """
        fcc_lattice = generate_lattice(
            self.packing_fraction, self.radius, "fcc", self.nx, self.ny, self.nz
        )
        hcp_lattice = generate_lattice(
            self.packing_fraction, self.radius, "hcp", self.nx, self.ny, self.nz
        )

        fcc = hpmc.State(
            hpmc.ParticleType.Sphere, list(fcc_lattice[3]), radius=self.radius
        )
        hcp = hpmc.State(
            hpmc.ParticleType.Sphere, list(hcp_lattice[3]), radius=self.radius
        )

        fcc.set_coords(*fcc_lattice[:3])
        hcp.set_coords(*hcp_lattice[:3])

        assert fcc.n_particles() == hcp.n_particles()

        return fcc, hcp

    def save_sample(
        self,
        progress: Progress,
        batch: int,
    ):
        if batch > self.plot_from_batch and batch % self.sample_interval == 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(
                self.plot_from_batch * self.sweeps, batch * self.sweeps, self.sweeps
            )
            y = np.array(self.delta_F_hist)[self.plot_from_batch :] * self.e_scale
            ax.plot(x, y, color="blue", label="Current Estimate")
            ax.axhline(
                y=133,
                color="red",
                linestyle="--",
                label="Literature Benchmark (133)",
            )
            ax.set_ylim(self.min_e, self.max_e)
            ax.set_title(f"Free Energy Convergence (Batch: {batch})")
            ax.set_xlabel("No. of Monte Carlo Sweeps")
            ax.set_ylabel("Free energy estimate ($10^5k_BT$)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

            filename = f"{self.outdir}/convergence_batch_{batch}.png"
            fig.savefig(filename, dpi=150)
            plt.close(fig)

            progress.console.log(
                f"[bold red][SAVED][/] Saved energy estimate graph to {filename}"
            )

    def update_log(
        self, progress: Progress, batch_task: TaskID, batch: int, start_time: float
    ):
        if batch % self.print_interval == 0:
            current_elapsed = time.perf_counter() - start_time
            batches_per_sec = batch / current_elapsed

            progress.update(
                batch_task,
                description=(
                    f"[bold blue]Simulating[/] "
                    f"[yellow](ln(f)={self.lsmc.wl_ln_f:.10f}, "
                    f"dF={self.delta_F:.6f}, "
                    f"[bold cyan]{batches_per_sec:.1f} b/s[/])"
                ),
            )

    def print_summary(self):
        total_elapsed_time = time.perf_counter() - self.start_time
        total_sweeps_executed = self.batches_executed * self.sweeps
        sweeps_per_second = total_sweeps_executed / max(total_elapsed_time, 1e-6)
        final_batches_per_second = self.batches_executed / max(total_elapsed_time, 1e-6)

        summary_table = Table(
            title="\n📊 Lattice Switching Simulation Summary",
            title_style="bold magenta",
            show_header=True,
            header_style="bold cyan",
        )
        summary_table.add_column("Parameter / Metric", style="bold white", width=35)
        summary_table.add_column("Value", justify="right", style="green")
        summary_table.add_section()
        summary_table.add_row("Total Batches Run", f"{self.batches_executed:,}")
        summary_table.add_row("Sweeps Per Batch", f"{self.sweeps:,}")
        summary_table.add_row("Total Engine Sweeps", f"{total_sweeps_executed:,}")
        summary_table.add_row("Packing Fraction (η)", f"{self.packing_fraction:.5f}")
        summary_table.add_section()
        summary_table.add_row("Final Convergence ln(f)", f"{self.lsmc.wl_ln_f:.10f}")
        summary_table.add_row("Final Delta F Estimate", f"{self.delta_F:.5f} Kb*T")
        summary_table.add_section()
        summary_table.add_row(
            "Total Execution Time", f"{total_elapsed_time:.2f} seconds"
        )
        summary_table.add_row(
            "Engine Batches Velocity",
            f"[bold cyan]{final_batches_per_second:.2f} b/s[/]",
        )
        summary_table.add_row(
            "Engine Sweeps Velocity",
            f"[bold yellow]{sweeps_per_second:,.0f} sweeps/sec[/]",
        )
        summary_table.add_row("Output Run Directory", f"{self.outdir}")
        console.print(summary_table)

    def save_summary_plots(self):
        fig, ax = plt.subplots(figsize=(7, 4))
        m_axis = np.arange(self.min_m, self.max_m + 1, dtype=np.int32)
        plt.xlim(self.min_m, self.max_m)

        ax.plot(m_axis, self.lsmc.bias_weights)
        ax.set_title("Wang Landau Bias (HCP vs FCC)")
        ax.set_xlabel("Overlap parameter M")
        ax.set_ylabel("Bias")
        fig.savefig(f"{self.outdir}/final_bias.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(m_axis, np.exp(self.lsmc.bias_weights))
        ax.set_title("Wang Landau Bias (HCP vs FCC)")
        ax.set_xlabel("Overlap parameter M")
        ax.set_ylabel("Exp(Bias)")
        fig.savefig(f"{self.outdir}/final_bias_exp.png", dpi=150)
        plt.close(fig)

        delta_F = self.calculate_free_energy(
            self.total_visit_counts, self.lsmc.bias_weights
        )
        print(f"The free energy difference (ΔF_HCP-FCC) is: {delta_F:.10f} k_B*T")

    def run(self):
        try:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,  # Links logging text to print cleanly above the bar
            ) as progress:
                batch_task = progress.add_task(
                    "Simulating Lattice...", total=self.batches
                )

                for batch in range(1, self.batches + 1):
                    self.lsmc.run_batch(sweeps=self.sweeps, max_disp=0.1)
                    self.batches_executed = batch

                    # We only care about the bias difference so we can subtract it by the min
                    self.lsmc.set_bias_weights(
                        self.lsmc.bias_weights - np.max(self.lsmc.bias_weights)
                    )

                    counts = self.lsmc.visit_counts

                    if np.min(self.total_visit_counts + counts) > 0:
                        self.delta_F = self.calculate_free_energy(
                            self.total_visit_counts + counts, self.lsmc.bias_weights
                        )

                    self.delta_F_hist.append(self.delta_F)

                    self.save_sample(progress, batch)
                    self.update_log(progress, batch_task, batch, self.start_time)

                    min_val = np.min(counts)
                    avg_val = np.mean(counts)
                    max_val = np.max(counts)

                    # 1. Ensure all accessible bins have been touched at least once
                    # 2. Check for the overall flatness
                    if (
                        min_val > 0
                        and (min_val > 0.8 * avg_val)
                        and (max_val < 1.2 * avg_val)
                    ):
                        self.lsmc.wl_ln_f /= 2
                        self.total_visit_counts += counts
                        self.lsmc.reset_visit_counts()

                        progress.console.log(
                            f"[bold green][CONVERGED][/] ln(f) dropped to [bold white]{self.lsmc.wl_ln_f:.10f}[/] at {batch * self.sweeps} sweeps."
                        )

                    # Tick the progress tracker forward
                    progress.advance(batch_task)

                console.print("\n[bold green]✔ All batches completed successfully![/]")
        except KeyboardInterrupt:
            console.print(
                "\n[bold orange3][INTERRUPT][/] Simulation paused by user (Ctrl+C)."
            )

        self.save_results()
        self.print_summary()
        self.save_summary_plots()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
