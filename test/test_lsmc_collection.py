import hpmc
from hpmc.utils.lattice import generate_lattice


def test_collection_counts_from_python():
    radius = 0.5
    packing_fraction = 0.5
    min_M, max_M = -50, 50
    sweeps = 10

    def make_state(lattice_type):
        x, y, z, box = generate_lattice(
            packing_fraction, radius, lattice_type, 6, 6, 6
        )

        state = hpmc.State(
            hpmc.ParticleType.Sphere,
            box,
            radius=radius,
        )
        state.set_coords(x, y, z)
        return state

    fcc = make_state("fcc")
    hcp = make_state("hcp")

    engine = hpmc.LSMCEngine(
        fcc, hcp, min_M, max_M, seed=7
    )

    engine.wl_ln_f = 0.0

    engine.run_batch(sweeps=sweeps, max_disp=0.1)

    total = sum(
        engine.collection_count(from_M, to_M)
        for from_M in range(min_M, max_M + 1)
        for to_M in range(min_M, max_M + 1)
    )

    assert total == sweeps * fcc.n_particles()

    engine.reset_transition_matrix()

    assert all(
        engine.collection_count(from_M, to_M) == 0.0
        for from_M in range(min_M, max_M + 1)
        for to_M in range(min_M, max_M + 1)
    )