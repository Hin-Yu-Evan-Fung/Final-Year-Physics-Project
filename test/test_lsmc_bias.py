import numpy as np
import pytest

import hpmc
from hpmc.utils.lattice import generate_lattice

L, N, SIGMA = 8.0, 216, 0.5
MIN_M, MAX_M = -50, 50
OFFSET = -MIN_M
SIZE = MAX_M - MIN_M + 1


def make_state(seed, lattice_type):
    RADIUS = 0.5
    ETA = 0.5

    assert lattice_type == "fcc" or lattice_type == "hcp", (
        "lattice types have to be valid"
    )

    x, y, z, box = generate_lattice(ETA, RADIUS, lattice_type, 6, 6, 6)

    s = hpmc.State(hpmc.ParticleType.Sphere, list(box), radius=RADIUS)
    s.set_coords(x, y, z)
    assert s.count_overlaps() == 0
    return s


@pytest.fixture
def make_sim():
    def _make(swap=False, seed=1):
        a, b = make_state(10, "fcc"), make_state(20, "hcp")
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
    assert 0.5 < neg / pos < 1.5  # loose: identical lattices, so near 1


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
