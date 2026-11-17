import os
import time
from dataclasses import dataclass
from typing import Self

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from hpmc import LSMCEngine
from hpmc.utils.math import calculate_free_energy
from hpmc.utils.plot import save_1d_plot


@dataclass
class SimConfig:
    # Batch config
    sweeps_per_batch: int = 1_000
    batches: int = 50_000


class Sim:
    def __init__(self, engine: LSMCEngine, cfg: SimConfig, outdir: str):
        self.lsmc: LSMCEngine = engine
        self.cfg = cfg
        self.outdir = outdir

        self.start_time = 0.0
        self.elapsed_time = 0.0

        self.delta_F: float = 0.0
        self.delta_F_hist: list[float] = []
        self.total_visit_counts: np.ndarray = np.zeros_like(
            self.lsmc.visit_counts, dtype=np.uint64
        )
        self.batches_executed = 0

        self.console: Console = Console()
        self._progress: Progress | None = None
        self._task_id = None

    def __enter__(self) -> Self:
        os.makedirs(self.outdir, exist_ok=True)
        self.start_time = time.perf_counter()

        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._progress.start()
        self._task_id = self._progress.add_task("Simulating", total=self.cfg.batches)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed_time = time.perf_counter() - self.start_time
        if self._progress is not None:
            self._progress.stop()

        self.save_result()
        self.save_summary_plots()
        self.print_summary()

        if exc_type is KeyboardInterrupt:
            print(f"\n[Interrupted] Saved checkpoint at batch {self.batches_executed}.")
            return True

        return False

    def is_visit_counts_flat(self, tolerance: float = 0.2) -> bool:
        """Check if all bins are visited and lie within (1 ± tolerance) of the mean."""
        counts = self.lsmc.visit_counts
        min_val = np.min(counts)
        if min_val <= 0:
            return False
        avg_val = np.mean(counts)
        return bool(
            (min_val > (1.0 - tolerance) * avg_val)
            and (np.max(counts) < (1.0 + tolerance) * avg_val)
        )

    def normalize_bias_weights(self) -> None:
        """Shift bias weights by their maximum to prevent numerical overflow."""
        weights = self.lsmc.bias_weights
        self.lsmc.set_bias_weights(weights - np.max(weights))

    def record_batch(self, m_off: int, n_particles: int) -> None:
        """Record a batch and calculate the free energy difference estimate"""

        counts = self.total_visit_counts + self.lsmc.visit_counts
        if np.min(counts) > 0:
            self.delta_F = calculate_free_energy(
                counts, self.lsmc.bias_weights, m_off, n_particles
            )
        self.delta_F_hist.append(self.delta_F)

        self.batches_executed += 1

        if self._progress is not None:
            assert self._task_id is not None, "TaskID has to be defined"
            self._progress.advance(self._task_id)

    def advance_wl_stage(self, factor: float = 0.5) -> None:
        """Multiples current wl_ln_f by factor and resets visit counts"""
        self.lsmc.wl_ln_f /= 2
        self.total_visit_counts += self.lsmc.visit_counts
        self.lsmc.reset_visit_counts()

    def save_result(self):
        np.savez_compressed(
            f"{self.outdir}/results.npz",
            wl_ln_f=self.lsmc.wl_ln_f,
            delta_F=self.delta_F,
            delta_F_hist=np.array(self.delta_F_hist),
            bias_weights=np.array(self.lsmc.bias_weights),
            total_visit_counts=np.array(self.total_visit_counts),
            batches_executed=self.batches_executed,
        )

    def save_summary_plots(self):
        min_m = self.lsmc.min_M
        max_m = self.lsmc.max_M
        m_axis = np.arange(min_m, max_m + 1, dtype=np.int32)

        save_1d_plot(
            m_axis,
            self.lsmc.bias_weights,
            f"{self.outdir}/final_bias.png",
            "Wang Landau Bias",
            "Overlap parameter M",
            "Bias",
            xlim=(min_m, max_m),
        )

        save_1d_plot(
            m_axis,
            np.exp(self.lsmc.bias_weights),
            f"{self.outdir}/final_bias_exp.png",
            "Wang Landau Bias",
            "Overlap parameter M",
            "Exp(Bias)",
            xlim=(min_m, max_m),
        )

    def save_sample_plot(
        self,
        plot_from_batch: int,
        batch: int,
        e_scale: float,
        min_e: float,
        max_e: float,
        hline: tuple[float, str] | None = None,
    ) -> str:

        sweeps = self.cfg.sweeps_per_batch
        filename = f"{self.outdir}/convergence_batch_{batch}.png"

        save_1d_plot(
            np.arange(plot_from_batch * sweeps, batch * sweeps, sweeps),
            np.array(self.delta_F_hist)[plot_from_batch:] * e_scale,
            filename,
            f"Free Energy Convergence (Batch: {batch})",
            "No. of Monte Carlo Sweeps",
            "Free energy estimate ($10^5k_BT$)",
            ylim=(min_e, max_e),
            hline=hline,
        )

        return filename

    def update_log(self, batch: int) -> None:
        current_elapsed = time.perf_counter() - self.start_time
        batches_per_sec = batch / current_elapsed

        assert self._task_id is not None, "Task has to be defined"
        assert self._progress is not None, "Progress has to be defined"

        self._progress.update(
            self._task_id,
            description=(
                f"[bold blue]Simulating[/] "
                f"[yellow](ln(f)={self.lsmc.wl_ln_f:.10f}, "
                f"dF={self.delta_F:.6f}, "
                f"[bold cyan]{batches_per_sec:.1f} b/s[/])"
            ),
        )

    def print_summary(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        total_sweeps_executed = self.batches_executed * self.cfg.sweeps_per_batch
        sweeps_per_second = total_sweeps_executed / max(elapsed, 1e-6)
        batches_per_second = self.batches_executed / max(elapsed, 1e-6)

        table = Table(
            title="\nSummary",
            title_style="bold magenta",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Parameter / Metric", style="bold white", width=35)
        table.add_column("Value", justify="right", style="green")

        sections = [
            [
                ("Total Batches Run", f"{self.batches_executed:,}"),
                ("Sweeps Per Batch", f"{self.cfg.sweeps_per_batch:,}"),
                ("Total Engine Sweeps", f"{total_sweeps_executed:,}"),
            ],
            [
                ("Final Convergence ln(f)", f"{self.lsmc.wl_ln_f:.10f}"),
                ("Final Delta F Estimate", f"{self.delta_F:.5f} Kb*T"),
            ],
            [
                ("Total Execution Time", f"{elapsed:.2f} seconds"),
                (
                    "Engine Batches Velocity",
                    f"[bold cyan]{batches_per_second:.2f} b/s[/]",
                ),
                (
                    "Engine Sweeps Velocity",
                    f"[bold yellow]{sweeps_per_second:,.0f} sweeps/sec[/]",
                ),
                ("Output Run Directory", f"{self.outdir}"),
            ],
        ]

        for section in sections:
            table.add_section()
            for label, value in section:
                table.add_row(label, value)
        self.console.print(table)
