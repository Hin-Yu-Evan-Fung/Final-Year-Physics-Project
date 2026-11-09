#include "transition_matrix.hpp"

#include <cstdint>
#include <stdexcept>

#include <algorithm>
#include <cmath>

namespace hpmc {

TransitionMatrix::TransitionMatrix(int min_M, int max_M)
    : min_M_(min_M), max_M_(max_M), num_bins_(0) {
    if (min_M > max_M) {
        throw std::invalid_argument("min_M must be <= max_M");
    }

    // Use a wider integer type before subtracting to avoid int overflow.
    num_bins_ = static_cast<std::size_t>(
        static_cast<std::int64_t>(max_M_) -
        static_cast<std::int64_t>(min_M_) + 1
    );

    // Check before multiplying the dimensions.
    if (num_bins_ > collection_.max_size() / num_bins_) {
        throw std::length_error("Transition matrix is too large");
    }

    collection_.assign(num_bins_ * num_bins_, 0.0);
}

std::size_t TransitionMatrix::bin_index(int M) const {
    if (M < min_M_ || M > max_M_) {
        throw std::out_of_range("Macrostate is outside the matrix range");
    }

    return static_cast<std::size_t>(
        static_cast<std::int64_t>(M) -
        static_cast<std::int64_t>(min_M_)
    );
}

std::size_t TransitionMatrix::matrix_index(int from_M, int to_M) const {
    const std::size_t row = bin_index(from_M);
    const std::size_t column = bin_index(to_M);

    return row * num_bins_ + column;
}


void TransitionMatrix::record(
    int from_M, int to_M, double unbiased_acceptance
) {
    if (!std::isfinite(unbiased_acceptance) ||
        unbiased_acceptance < 0.0 ||
        unbiased_acceptance > 1.0) {
        throw std::invalid_argument(
            "Acceptance probability must be finite and between 0 and 1"
        );
    }

    const std::size_t transition = matrix_index(from_M, to_M);
    const std::size_t diagonal = matrix_index(from_M, from_M);

    collection_[transition] += unbiased_acceptance;
    collection_[diagonal] += 1.0 - unbiased_acceptance;
}

double TransitionMatrix::count(int from_M, int to_M) const {
    return collection_[matrix_index(from_M, to_M)];
}

void TransitionMatrix::reset() {
    std::fill(collection_.begin(), collection_.end(), 0.0);
}

} // namespace hpmc