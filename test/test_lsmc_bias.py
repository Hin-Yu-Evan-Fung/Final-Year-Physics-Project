import numpy as np
import pytest

import hpmc

L, N, SIGMA = 8.0, 216, 0.5
MIN_M, MAX_M = -50, 50
OFFSET = -MIN_M
SIZE = MAX_M - MIN_M + 1


def make_state(seed):
    rng = np.random.default_rng(seed)

    points_per_axis = int(np.ceil(np.pow(N, (1 / 3))))

    # Calculate the linear footprint of the required grid
    required_length = points_per_axis * SIGMA * 2

    if required_length > L:
        raise ValueError(
            f"Invalid Lattice Configuration! A {points_per_axis}x{points_per_axis}x{points_per_axis} "
            f"grid requires a linear spacing of at least {required_length:.4f} units, "
            f"but your box_length is only {L:.4f}."
        )

    grid_points = np.linspace(
        SIGMA * 2, L - SIGMA * 2, points_per_axis, dtype=np.float64
    )

    x, y, z = np.meshgrid(grid_points, grid_points, grid_points, indexing="ij")

    x = np.ravel(x)[:N]
    y = np.ravel(y)[:N]
    z = np.ravel(z)[:N]

    s = hpmc.State(hpmc.ParticleType.Sphere, L, radius=SIGMA)
    s.set_coords(*[(c + rng.normal(0, 0.05 * SIGMA, N)) % L for c in (x, y, z)])
    assert s.count_overlaps() == 0
    return s


@pytest.fixture
def make_sim():
    def _make(swap=False, seed=1):
        a, b = make_state(10), make_state(20)
        if swap:
            a, b = b, a
        sim = hpmc.LSMCEngine(a, b, MIN_M, MAX_M, seed=seed)
        sim.wl_ln_f = 1.0
        return sim

    return _make


def test_starts_at_zero(make_sim):
    assert make_sim().current_M == 0


def test_negative_and_positive_M_both_visited(make_sim):
    sim = make_sim()
    sim.set_bias_weights(np.zeros_like(sim.bias_weights))
    for _ in range(50):
        sim.run_batch(sweeps=50, max_disp=0.15)
    v = sim.visit_counts
    assert v[:OFFSET].sum() > 0, "never visited M < 0"
    assert v[OFFSET + 1 :].sum() > 0, "never visited M > 0"


@pytest.mark.parametrize("swap", [False, True])
def test_visits_roughly_symmetric(make_sim, swap):
    sim = make_sim(swap=swap)
    for _ in range(200):
        sim.run_batch(sweeps=50, max_disp=0.15)
    v = sim.visit_counts
    neg, pos = v[:OFFSET].sum(), v[OFFSET + 1 :].sum()
    assert 0.7 < neg / pos < 1.3  # loose: identical lattices, so near 1


def test_M_stays_in_range(make_sim):
    sim = make_sim()
    for _ in range(30):
        sim.run_batch(sweeps=20, max_disp=0.15)
        sim.reset_visit_counts()
        assert MIN_M <= sim.current_M <= MAX_M


def test_bias_equals_visits_for_constant_f(make_sim):
    sim = make_sim()
    sim.run_batch(sweeps=100, max_disp=0.15)
    np.testing.assert_allclose(sim.bias_weights, sim.visit_counts)
