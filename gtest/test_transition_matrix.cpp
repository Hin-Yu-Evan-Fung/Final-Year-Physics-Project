#include <gtest/gtest.h>
#include <stdexcept>

#include "transition_matrix.hpp"

TEST(TransitionMatrixTest, InitiallyZero) {
    hpmc::TransitionMatrix matrix(-1, 1);

    for (int from = -1; from <= 1; ++from) {
        for (int to = -1; to <= 1; ++to) {
            EXPECT_DOUBLE_EQ(matrix.count(from, to), 0.0);
        }
    }
}

TEST(TransitionMatrixTest, AllowedProposalRecordedEvenIfBiasRejects) {
    hpmc::TransitionMatrix matrix(-1, 1);

    // The proposal is physically allowed. Its later biased acceptance
    // decision does not affect what we record here.
    matrix.record(-1, 0, 1.0);

    EXPECT_DOUBLE_EQ(matrix.count(-1, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.count(-1, -1), 0.0);
}

TEST(TransitionMatrixTest, ForbiddenProposalCountsAsStay) {
    hpmc::TransitionMatrix matrix(-1, 1);

    matrix.record(-1, -1, 0.0);

    EXPECT_DOUBLE_EQ(matrix.count(-1, -1), 1.0);
}

TEST(TransitionMatrixTest, UnchangedMacrostateCountsOnce) {
    hpmc::TransitionMatrix matrix(-1, 1);

    matrix.record(0, 0, 1.0);

    EXPECT_DOUBLE_EQ(matrix.count(0, 0), 1.0);
}

TEST(TransitionMatrixTest, AccumulatesAndResets) {
    hpmc::TransitionMatrix matrix(-1, 1);

    matrix.record(-1, 0, 1.0);
    matrix.record(-1, 0, 1.0);
    matrix.record(-1, -1, 0.0);

    EXPECT_DOUBLE_EQ(matrix.count(-1, 0), 2.0);
    EXPECT_DOUBLE_EQ(matrix.count(-1, -1), 1.0);

    matrix.reset();

    EXPECT_DOUBLE_EQ(matrix.count(-1, 0), 0.0);
    EXPECT_DOUBLE_EQ(matrix.count(-1, -1), 0.0);
}

TEST(TransitionMatrixTest, NormalisesRow) {
    hpmc::TransitionMatrix matrix(-1, 1);

    matrix.record(0, -1, 1.0);
    matrix.record(0, 0, 1.0);
    matrix.record(0, 0, 1.0);
    matrix.record(0, 1, 1.0);

    EXPECT_DOUBLE_EQ(matrix.transition_probability(0, -1), 0.25);
    EXPECT_DOUBLE_EQ(matrix.transition_probability(0, 0), 0.50);
    EXPECT_DOUBLE_EQ(matrix.transition_probability(0, 1), 0.25);

    // Normalisation must not modify the stored counts.
    EXPECT_DOUBLE_EQ(matrix.count(0, 0), 2.0);
}

// Checks division by 0 when trying to convert collection matrix to transition matrix
TEST(TransitionMatrixTest, UnsampledRowThrows) {
    hpmc::TransitionMatrix matrix(-1, 1);

    EXPECT_THROW(
        matrix.transition_probability(0, 1),
        std::runtime_error
    );
}
