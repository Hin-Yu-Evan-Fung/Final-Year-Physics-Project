from typing import Any

import numpy as np

from .hpmc_ext import ParticleType
from .hpmc_ext import LSMCEngine as _LSMCEngine
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


class LSMCEngine:
    """A high-level Python wrapper for the C++ Lattice Switching Monte Carlo Engine."""

    def __init__(
        self, state_a: State, state_b: State, min_M: int, max_M: int, seed: int = 0
    ):
        # Anchor the Python state wrappers to prevent garbage collection hollowing
        self._state_a_ref = state_a
        self._state_b_ref = state_b

        # Instantiate the raw nanobind C++ engine
        self._engine = _LSMCEngine(state_a.state, state_b.state, min_M, max_M, seed)

    def run_batch(self, sweeps: int, max_disp: float) -> int:
        """Performs particle translation sweeps, records bias weights, and updates the macrostate.

        Returns:
            The number of successfully accepted moves.
        """
        return self._engine.run_batch(sweeps, max_disp)

    @property
    def wl_ln_f(self) -> float:
        """Return the Wang Landau bias update parameter wl_ln_f"""
        return self._engine.wl_ln_f

    @wl_ln_f.setter
    def wl_ln_f(self, val: float) -> None:
        self._engine.wl_ln_f = val

    @property
    def current_M(self) -> int:
        """Returns the current signed macrostate value M (Overlaps A - Overlaps B)."""
        return self._engine.M

    @property
    def active_lattice(self) -> str:
        """Returns whether the active lattice is currently 'A' or 'B'."""
        # Maps the C++ enum state back into a clean readable string
        return "A" if self._engine.active_lattice == Lattice.A else "B"

    @property
    def bias_weights(self) -> np.ndarray:
        """Returns bias weights"""
        return self._engine.bias_weights

    def set_bias_weights(self, new_bias: np.ndarray) -> None:
        """Resets visit counts"""
        return self._engine.set_bias_weights(new_bias)

    @property
    def visit_counts(self) -> np.ndarray:
        """Returns visit counts"""
        return self._engine.visit_counts

    def reset_visit_counts(self) -> None:
        """Resets visit counts"""
        return self._engine.reset_visit_counts()
