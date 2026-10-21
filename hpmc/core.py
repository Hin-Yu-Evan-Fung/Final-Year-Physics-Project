import numpy as np

from .hpmc_ext import HardSphereEngine, HardSphereParams


class HardSphereSimulation:
    """
    A high-level Python wrapper for the C++ HPMC Engine.
    """

    def __init__(self, L: float, params: HardSphereParams):
        self.L = float(L)
        self._dof = 3  # Degrees of freedom
        self._params = params
        self._engine = HardSphereEngine(self.L, self._params)

    def n_particles(self) -> int:
        """Returns the total number of particles currently in the system."""
        return self._engine.size()

    def set_particles(self, positions: np.ndarray):
        """Sets the particles via a np.ndarray, [particle, degree of freedom]"""
        coords = np.asarray(positions, dtype=np.float64, order="C")
        self._engine.set_particles(coords)

    def get_particles(self) -> np.ndarray:
        """Returns particles as a np.ndarray, indexed by [particle, degree of freedom]"""
        return self._engine.get_particles()

    def count_overlaps(self) -> int:
        """Returns number of overlaps in the system"""
        return self._engine.count_overlaps()

    def __validate_moves(self) -> None:
        if self.n_particles() == 0:
            raise RuntimeError(
                "Cannot execute simulation steps with 0 particles initialized."
            )

    def run_batch(self, sweeps: int, max_disp: float, seed: int = 1) -> int:
        """Perform a batch of monte carlo sweeps at a certain max displacement"""
        self.__validate_moves()
        return self._engine.run_batch(sweeps, max_disp, seed)
