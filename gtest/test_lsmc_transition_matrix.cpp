#include <gtest/gtest.h>

#include "lsmc_engine.hpp"

#include <memory>
#include <vector>

TEST(LSMCTransitionMatrixTest, RecordsEveryTranslationAttempt) {
    using namespace hpmc;

    // Two initially overlap-free configurations, with two spheres each.
    auto state_a = std::make_shared<State>(
        ParticleType::Sphere,
        Vec3{10.0, 10.0, 10.0},
        SphereConfig{0.5}
    );

    auto state_b = std::make_shared<State>(
        ParticleType::Sphere,
        Vec3{10.0, 10.0, 10.0},
        SphereConfig{0.5}
    );

    const std::vector<double> x_a{2.0, 6.0};
    const std::vector<double> x_b{2.0, 3.1};
    const std::vector<double> y{2.0, 2.0};
    const std::vector<double> z{2.0, 2.0};

    state_a->set_coords(x_a, y, z);
    state_b->set_coords(x_b, y, z);

    // With two spheres, each configuration has at most one overlap.
    LSMCEngine engine(state_a, state_b, -1, 1, 7);

    constexpr std::size_t sweeps = 100;
    engine.run_batch(sweeps, 0.4);

    // Sum every entry in the collection matrix.
    double total = 0.0;

    for (int from_M = -1; from_M <= 1; ++from_M) {
        for (int to_M = -1; to_M <= 1; ++to_M) {
            total += engine.transition_matrix().count(from_M, to_M);
        }
    }

    const double expected = static_cast<double>(
        sweeps * state_a->n()
    );

    EXPECT_DOUBLE_EQ(total, expected);

    // Run another batch without clearing the collection matrix.
    engine.run_batch(sweeps, 0.4);

    double accumulated_total = 0.0;

    for (int from_M = -1; from_M <= 1; ++from_M) {
        for (int to_M = -1; to_M <= 1; ++to_M) {
        accumulated_total += engine.transition_matrix().count(from_M, to_M);
        }
    }

    EXPECT_DOUBLE_EQ(accumulated_total, 2.0 * expected);

    // Check that explicitly resetting clears every entry.
    engine.reset_transition_matrix();

    for (int from_M = -1; from_M <= 1; ++from_M) {
        for (int to_M = -1; to_M <= 1; ++to_M) {
            EXPECT_DOUBLE_EQ(
                engine.transition_matrix().count(from_M, to_M),
                0.0
            );
        }
    }
    }
