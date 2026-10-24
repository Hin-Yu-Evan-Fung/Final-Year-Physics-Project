from typing import Any

import numpy as np

from .hpmc_ext import ParticleType
from .hpmc_ext import SingleBoxEngine as _SingleBoxEngine


class SingleBoxEngine:
    """
    A high-level Python wrapper for the C++ HPMC Engine.
    """

    def __init__(
        self,
        particle_type: ParticleType,
        box_length: float,
        seed: int = 1,
        **kwargs: Any,
    ):
        """
        Single Box Monte Carlo simulation for any supported rigid molecule type.
        """
        self._engine = _SingleBoxEngine(particle_type, box_length, seed, **kwargs)

    def n_particles(self) -> int:
        """Returns the total number of particles currently in the system."""
        return self._engine.n()

    def set_coords(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Sets the particles via a np.ndarray, [particle, degree of freedom]"""
        self._engine.set_coords(x, y, z)

    def get_coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns particles as a np.ndarray, indexed by [particle, degree of freedom]"""
        return self._engine.get_coords()

    def count_overlaps(self) -> int:
        """Returns number of overlaps in the system"""
        return self._engine.count_overlaps()

    def __validate_moves(self) -> None:
        if self.n_particles() == 0:
            raise RuntimeError(
                "Cannot execute simulation steps with 0 particles initialized."
            )

    def run_batch(self, sweeps: int, max_disp: float) -> int:
        """Perform a batch of monte carlo sweeps at a certain max displacement"""
        self.__validate_moves()
        return self._engine.run_batch(sweeps, max_disp)
