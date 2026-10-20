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

    def __validate_moves(self, indices: np.ndarray, moves: np.ndarray) -> None:
        if self.n_particles() == 0:
            raise RuntimeError(
                "Cannot execute simulation steps with 0 particles initialized."
            )
        n_particles = indices.shape[0]
        if indices.ndim != 1 or indices.dtype != np.uint32:
            raise RuntimeError("Wrong indices format, should be a 1d float64 np.array")
        if moves.shape != (n_particles, self._dof) or moves.dtype != np.float64:
            raise RuntimeError(
                f"Wrong moves format, should be a 2d float64 np.array with shape (self.size, {self._dof})"
            )

    def try_run_batch(self, indices: np.ndarray, moves: np.ndarray) -> int:
        """Tries to displace batches of particles at different indicies by the provided move, reject if collision occurs"""
        self.__validate_moves(indices, moves)
        return self._engine.try_run_batch(indices, moves)

    def run_batch(self, indices: np.ndarray, moves: np.ndarray) -> None:
        """Displaces batches of particles at different indicies by the provided move"""
        self.__validate_moves(indices, moves)
        self._engine.run_batch(indices, moves)
