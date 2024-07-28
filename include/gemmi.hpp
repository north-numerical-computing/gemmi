#include <bit>
#include <cassert>
#include <vector>

/* Everything is defined to use column-major. */
enum class matrix_position {
    on_the_left,
    on_the_right
};

template <typename splitint_t, typename fp_type>
struct MatrixSplit {
    size_t m;
    size_t n;
    size_t num_splits;
    std::vector<splitint_t> memory;
    std::vector<fp_type> powers_vector;

    MatrixSplit(const size_t m, const size_t n, const size_t num_splits,
                std::vector<splitint_t>& memory,
                std::vector<fp_type>& powers_vector) :
                m(m), n(n), num_splits(num_splits),
                memory(memory),
                powers_vector(powers_vector) {}
};

template <typename splitint_t, typename accumulator_t,
          typename fp_t, typename uint_t,
          size_t n_exp_bits, size_t n_frac_bits>
MatrixSplit<splitint_t, fp_t> splitFloatToInt(const std::vector<fp_t> A,
                                              const size_t m, const size_t n,
                                              matrix_position position,
                                              const size_t num_splits,
                                              const size_t bits_per_slice) {

    // Allocate additional memory for splits.
    std::vector <splitint_t> memory (A.size() * num_splits);

    // Compute maximum number of bits per integer. k is the dimension alng which
    // the inner product is computed. This will be the number of columns for the
    // matrix on the left of the product, which is normalised by rows, and the
    // number of rows for the matrix on the right of the product, which is
    // normalised by columns.
    const size_t k = (position == matrix_position::on_the_left) ? n : m;
    const size_t other_dim = k == n ? m : n;

    // Compute normalisation vector.
    std::vector<fp_t> powers_vector (other_dim);
    std::vector<int> scaling_exponents (other_dim);
    const size_t i_stride = (position == matrix_position::on_the_left) ? 1 : m;
    const size_t j_stride = (position == matrix_position::on_the_left) ? m : 1;
    for (size_t i = 0; i < other_dim; i++) {
        powers_vector[i] = 0.0;
        for (size_t j = 0; j < k; j++) {
            const size_t index = i * i_stride + j * j_stride;
            powers_vector[i] = std::max(powers_vector[i], std::abs(A[index]));
        }
        // Compute the smallest power of 2 that is strictly greater than the
        // maximum value in the row/column.
        // NOTE 1: This is a fix on the paper.
        // NOTE 2: I use exponents instead of powers of 2, as I need the former
        //         to shift correctly. This is also a fix on the paper.
        frexp(powers_vector[i], scaling_exponents.data() + i);
        const auto largest_log = log2(powers_vector[i]);
        auto temp = ceil(largest_log) +
                    (ceil(largest_log) == floor(largest_log) ? 1 : 0);
        powers_vector[i] = std::ldexp(1.0, temp);
    }

    // Compute splits one row/column at a time.
    std::vector<uint_t> tmp (k);
    std::vector<bool> sign (k);
    for (size_t i = 0; i < other_dim; i++) {
        // Get binary representation of normalised row/column.
        for (size_t j = 0; j < k; j++) {
            const size_t index = i * i_stride + j * j_stride;
            fp_t value = A[index]; // / powers_vector[i];
            tmp[j] = std::bit_cast<uint_t>(value);        // To bitstring.
            sign[j] = std::signbit(value);                // Extract sign.
            tmp[j] &= (~(uint_t)(0)) >> (n_exp_bits + 1); // Remove exponent.
            // Restore implicit bit for normal numbers.
            // TODO: NaNs and infs are currently not supported..
            if (std::fpclassify(value) == FP_NORMAL)
                tmp[j] |= ((uint_t)1 << (n_frac_bits - 1));
        }

        // Create bitmask.
        const uint_t small_bitmask = (1 << bits_per_slice) - 1;
        // Perform the split.
        for (size_t j = 0; j < k; j++) {
            int8_t shift_counter = n_frac_bits - bits_per_slice;
            int current_exponent;
            frexp(A[i * i_stride + j * j_stride], &current_exponent);
            int8_t exponent_difference = scaling_exponents[i] - current_exponent;
            for (size_t ell = 0; ell < num_splits; ell++) {
                if (exponent_difference > (signed)bits_per_slice) {
                    exponent_difference -= bits_per_slice;
                } else {
                    shift_counter += exponent_difference;
                    exponent_difference = 0;
                    uint_t bitmask = shift_counter > 0 ?
                        small_bitmask << shift_counter :
                        small_bitmask >> -shift_counter;
                    uint_t current_slice = tmp[j] & bitmask;
                    uint_t current_split = shift_counter > 0 ?
                        current_slice >> shift_counter :
                        current_slice << -shift_counter;
                    splitint_t value = (splitint_t)(current_split) * (sign[j] ? -1 : 1);
                    memory[i * i_stride + j * j_stride + ell * A.size()] = value;
                    shift_counter -= bits_per_slice;
                }
            }
        }
    }
    return MatrixSplit<splitint_t, fp_t>(m, n, num_splits, memory, powers_vector);
}

template <typename splitint_t, typename accumulator_t,
          typename fp_t, typename uint_t,
          size_t n_exp_bits, size_t n_frac_bits>
std::vector<fp_t> mergeFloatfromInt(const MatrixSplit<splitint_t, fp_t> &A,
                                    const size_t bits_per_slice) {
    std::vector<fp_t> C (A.m * A.n, 0.0);

    for (size_t i = 0; i < A.m; i++) {
        uint_t tmp = 0;
        for (size_t j = 0; j < A.n; j++) {
            int8_t shift_value = n_frac_bits - bits_per_slice;
            for (size_t i_block = 0; i_block < A.num_splits; i_block++) {
                uint_t slice = A.memory[i + j * A.m + i_block * A.m * A.n];
                uint_t new_slice = shift_value > 0 ?
                    slice << shift_value :
                    slice >> -shift_value;
                tmp |= new_slice;
                shift_value -= bits_per_slice;
            }
            C[i + j * A.m] = std::ldexp(tmp, -(int)n_frac_bits) *
                             A.powers_vector[i];
        }
    }

    return C;
}

template <typename splitint_t, typename accumulator_t,
          typename fp_t, typename uint_t,
          size_t n_exp_bits, size_t n_frac_bits>
std::vector<fp_t> compute_products(const MatrixSplit<splitint_t, fp_t> &A,
                                   const MatrixSplit<splitint_t, fp_t> &B,
                                   const size_t bits_per_slice) {

    std::vector<fp_t > C (A.m * B.n);

    for (size_t i_block = 0; i_block < A.num_splits; i_block++) {
        for (size_t j_block = 0; j_block < B.num_splits - i_block; j_block++) {
            for (size_t i = 0; i < A.m; i++) {
                for (size_t j = 0; j < B.n; j++) {
                    accumulator_t sum = 0;
                    for (size_t k = 0; k < A.n; k++) {
                        sum += A.memory[i + k * A.m + i_block * A.m * A.n] *
                               B.memory[k + j * B.m + j_block * B.m * B.n];
                    }
                    fp_t scaled_sum = std::ldexp(sum, -(i_block+1 + j_block+1) * bits_per_slice);
                    fp_t scaling_factor = A.powers_vector[i] * B.powers_vector[j];
                    C[i + j * A.m] += scaled_sum * scaling_factor;
                }
            }
        }
    }

    return C;
}

template <typename splitint_t, typename accumulator_t, typename fp_t, typename uint_t,
          size_t n_exp_bits, size_t n_frac_bits>
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const std::vector<fp_t> &B,
                         const size_t m, const size_t p, const size_t n, const size_t num_splits) {

    const size_t bits_in_accumulator = std::numeric_limits<accumulator_t>::digits;
    const size_t bits_in_integer = std::numeric_limits<splitint_t>::digits;
    assert(bits_in_integer <= bits_in_accumulator / 2);
    const size_t alpha = std::floor((bits_in_accumulator - log2(n)) / 2);
    const size_t bits_per_slice = std::min(bits_in_integer, static_cast<size_t>(alpha));

    auto splitA = splitFloatToInt<splitint_t, accumulator_t, fp_t, uint_t, n_exp_bits, n_frac_bits>
        (A, m, p, matrix_position::on_the_left, num_splits, bits_per_slice);

    auto splitB = splitFloatToInt<splitint_t, accumulator_t, fp_t, uint_t, n_exp_bits, n_frac_bits>
        (B, p, n, matrix_position::on_the_right, num_splits, bits_per_slice);

    return compute_products<splitint_t, accumulator_t, fp_t, uint_t, n_exp_bits, n_frac_bits>(splitA, splitB, bits_per_slice);
}

template
std::vector<float> gemmi<int8_t, int32_t, float, uint32_t, 8, 24> (const std::vector<float> &A, const std::vector<float> &B,
                         const size_t m, const size_t p, const size_t n, const size_t num_splits);
template
std::vector<double> gemmi<int8_t, int32_t, double, uint64_t, 11, 53> (const std::vector<double> &A, const std::vector<double> &B,
                         const size_t m, const size_t p, const size_t n, const size_t num_splits);
