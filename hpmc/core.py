from typing import Any

import numpy as np

from .hpmc_ext import ParticleType
from .hpmc_ext import SingleBoxEngine as _SingleBoxEngine
from .hpmc_ext import State as _State


class State:
    def __init__(
        self, particle_type: ParticleType, box_length: float, **kwargs: Any
    ) -> None:
        """
        A state containing a periodic bounding box and particles of the same type and config
        """
        self.state = _State(particle_type, box_length, **kwargs)

    def n_particles(self) -> int:
        """Returns the total number of particles currently in the system."""
        return self.state.n()

    def set_coords(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Sets the particles via a np.ndarray, [particle, degree of freedom]"""
        self.state.set_coords(x, y, z)

    def get_coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns particles as a np.ndarray, indexed by [particle, degree of freedom]"""
        return self.state.get_coords()

    def count_overlaps(self) -> int:
        """Returns number of overlaps in the system"""
        return self.state.count_overlaps()


class SingleBoxEngine:
    """
    A high-level Python wrapper for the C++ HPMC Engine.
    """

    def __init__(self, state: State, seed: int = 0):
        """
        Single Box Monte Carlo simulation for any supported rigid molecule type.
        """
        self._engine = _SingleBoxEngine(state.state, seed)

    def run_batch(self, sweeps: int, max_disp: float) -> int:
        """Perform a batch of monte carlo sweeps at a certain max displacement"""
        return self._engine.run_batch(sweeps, max_disp)
