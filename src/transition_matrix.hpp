#pragma once

#include <cstddef>
#include <vector>

namespace hpmc {

class TransitionMatrix {
public:
    TransitionMatrix(int min_M, int max_M);

    // Record one attempted move using its acceptance probability
    // with the sampling bias removed.
    void record(int from_M, int to_M, double unbiased_acceptance);

    // Read a collection-matrix entry using macrostate values.
    double count(int from_M, int to_M) const;

    // Return the estimated unbiased transition probability.
    // Throws if no attempts have been recorded from from_M.
    double transition_probability(int from_M, int to_M) const;

    // Clear all accumulated statistics.
    void reset();

private:
    int min_M_;
    int max_M_;
    std::size_t num_bins_;

    // Dense matrix stored as a single row-major array.
    std::vector<double> collection_;

    // Convert a macrostate value into a bin index.
    std::size_t bin_index(int M) const;

    // Convert a pair of macrostate values into an array index.
    std::size_t matrix_index(int from_M, int to_M) const;
};

} // namespace hpmc
