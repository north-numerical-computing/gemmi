#include <bit>
#include <cassert>
#include <vector>

template <typename fp_t> size_t numExpBits() {return 0;}
template <> size_t numExpBits<float>() {return 8;}
template <> size_t numExpBits<double>() {return 11;}

template <typename fp_t> size_t numFracBits() {return 0;}
template <> size_t numFracBits<float>() {return 24;}
template <> size_t numFracBits<double>() {return 53;}

/* Everything is defined to use column-major. */
enum class normalisationDimension {
    byRows,
    byCols
};

template <typename splitint_t, typename fp_type, typename uint_t>
struct MatrixSplit {
    size_t m;
    size_t n;
    size_t num_splits;
    normalisationDimension dimension;

    std::vector<fp_type> matrix;
    std::vector<splitint_t> memory;
    std::vector<fp_type> powers_vector;
    std::vector<int> scaling_exponents;

    MatrixSplit(const size_t m, const size_t n, const size_t num_splits,
                const normalisationDimension dimension,
                const std::vector<fp_type>& matrix,
                std::vector<splitint_t>& memory,
                std::vector<fp_type>& powers_vector,
                std::vector<int>& scaling_exponents) :
                m(m), n(n), num_splits(num_splits), dimension(dimension),
                matrix(matrix), memory(memory),
                powers_vector(powers_vector),
                scaling_exponents(scaling_exponents) {}

    MatrixSplit(const size_t m, const size_t n, const size_t num_splits,
                const normalisationDimension dimension, const std::vector<fp_type>& matrix) :
                m(m), n(n), num_splits(num_splits), dimension(dimension),
                matrix(matrix) {
                    this->memory.resize(m * n * num_splits);
                    this->powers_vector.resize(this->otherDimension());
                    this->scaling_exponents.resize(this->otherDimension());
                }

    // This is the dimension alng which the inner product is computed.
    // This will be the number of columns for the matrix on the left of the
    // product, which is normalised by rows, and the number of rows for the
    // matrix on the right, which is normalised by columns.
    size_t innerProductDimension() {
        return (dimension == normalisationDimension::byRows) ? n : m;
    }

    size_t otherDimension() {
        return (dimension == normalisationDimension::byRows) ? m : n;
    }

    size_t iStride() {
        return (dimension == normalisationDimension::byRows) ? 1 : m;
    }

    size_t jStride() {
        return (dimension == normalisationDimension::byRows) ? m : 1;
    }

    void computeNormalisationVectors() {
        // Compute normalisation vector.
        const size_t i_stride = this->iStride();
        const size_t j_stride = this->jStride();
        for (size_t i = 0; i < this->otherDimension(); i++) {
            this->powers_vector[i] = 0.0;
            for (size_t j = 0; j < this->innerProductDimension(); j++) {
                const size_t index = i * i_stride + j * j_stride;
                this->powers_vector[i] = std::max(this->powers_vector[i],
                                                  std::abs(this->matrix[index]));
            }
            // Compute the smallest power of 2 that is strictly greater than the
            // maximum value in the row/column.
            // NOTE 1: This is not the same technique used in uoi24.
            // NOTE 2: I use exponents instead of powers of 2, as I need the former
            //         to shift correctly.
            frexp(this->powers_vector[i], this->scaling_exponents.data() + i);
            const auto largest_log = log2(this->powers_vector[i]);
            auto temp = ceil(largest_log) +
                        (ceil(largest_log) == floor(largest_log) ? 1 : 0);
            this->powers_vector[i] = std::ldexp(1.0, temp);
        }
    }

    void computeSplits(const size_t bits_per_slice) {
        // Compute splits one row/column at a time.
        auto n_exp_bits = numExpBits<fp_type>();
        auto n_frac_bits = numFracBits<fp_type>();
        auto k = this->innerProductDimension();
        auto i_stride = this->iStride();
        auto j_stride = this->jStride();
        std::vector<uint_t> tmp (k);
        std::vector<bool> sign (k);
        for (size_t i = 0; i < this->otherDimension(); i++) {
            // Get binary representation of normalised row/column.
            for (size_t j = 0; j < this->innerProductDimension(); j++) {
                const size_t index = i * i_stride + j * j_stride;
                fp_type value = this->matrix[index]; // / powers_vector[i];
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
                int16_t shift_counter = n_frac_bits - bits_per_slice;
                int current_exponent;
                frexp(this->matrix[i * i_stride + j * j_stride], &current_exponent);
                int16_t exponent_difference = scaling_exponents[i] - current_exponent;
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
                        this->memory[i * i_stride + j * j_stride + ell * this->matrix.size()] = value;
                        shift_counter -= bits_per_slice;
                    }
                }
            }
        }
    }
};

template <typename splitint_t, typename fp_t, typename uint_t>
MatrixSplit<splitint_t, fp_t, uint_t> splitFloatToInt(const std::vector<fp_t> A,
                                              const size_t m, const size_t n,
                                              normalisationDimension dimension,
                                              const size_t num_splits,
                                              const size_t bits_per_slice) {
    auto splits = MatrixSplit<splitint_t, fp_t, uint_t>(m, n, num_splits, dimension, A);
    splits.computeNormalisationVectors();
    splits.computeSplits(bits_per_slice);

    return splits;
}

template <typename splitint_t, typename accumulator_t,
          typename fp_t, typename uint_t>
std::vector<fp_t> mergeFloatfromInt(const MatrixSplit<splitint_t, fp_t, uint_t> &A,
                                    const size_t bits_per_slice) {
    std::vector<fp_t> C (A.m * A.n, 0.0);

    for (size_t i = 0; i < A.m; i++) {
        uint_t tmp = 0;
        for (size_t j = 0; j < A.n; j++) {
            int8_t shift_value = numFracBits<fp_t>() - bits_per_slice;
            for (size_t i_block = 0; i_block < A.num_splits; i_block++) {
                uint_t slice = A.memory[i + j * A.m + i_block * A.m * A.n];
                uint_t new_slice = shift_value > 0 ?
                    slice << shift_value :
                    slice >> -shift_value;
                tmp |= new_slice;
                shift_value -= bits_per_slice;
            }
            C[i + j * A.m] = std::ldexp(tmp, -(int)numFracBits<fp_t>()) *
                             A.powers_vector[i];
        }
    }

    return C;
}

/* Compute exact products of slices of A and B. */
template <typename splitint_t, typename accumulator_t, typename fp_t, typename uint_t>
void computeExactIntegerGEMM(const MatrixSplit<splitint_t, fp_t, uint_t> &A, const MatrixSplit<splitint_t, fp_t, uint_t> &B, size_t i_block, size_t j_block, const size_t bits_per_slice, std::vector<fp_t> &C) {
            for (size_t i = 0; i < A.m; i++) {
                for (size_t j = 0; j < B.n; j++) {
                    accumulator_t sum = 0;
                    for (size_t k = 0; k < A.n; k++) {
                        sum += A.memory[i + k * A.m + i_block * A.m * A.n] *
                               B.memory[k + j * B.m + j_block * B.m * B.n];
                    }
            fp_t scaled_sum = std::ldexp(sum, -(i_block + 1 + j_block + 1) * bits_per_slice);
                    fp_t scaling_factor = A.powers_vector[i] * B.powers_vector[j];
                    C[i + j * A.m] += scaled_sum * scaling_factor;
                }
            }
}

/* Accumulate products using the technique in:

    Ootomo H., Ozaki K., Yokota R. DGEMM on integer matrix multiplication unit.
    Int. J. High Performance Comput. App. 2024;38(4):297-313. DOI:10.1177/10943420241239588

Each integer product is computed and accumulated in floating-point arithmetic.
*/
template <typename splitint_t, typename accumulator_t, typename fp_t, typename uint_t>
std::vector<fp_t> computeProductsWithFloatingPlointAccumulation(const MatrixSplit<splitint_t, fp_t, uint_t> &A,
                                  const MatrixSplit<splitint_t, fp_t, uint_t> &B,
                                  const size_t bits_per_slice) {

    std::vector<fp_t > C (A.m * B.n);

    for (size_t i_block = 0; i_block < A.num_splits; i_block++) {
        for (size_t j_block = 0; j_block < B.num_splits - i_block; j_block++) {
            computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t, uint_t>(A, B, i_block, j_block, bits_per_slice, C);
        }
    }

    return C;
}

/* Compute matrix vector product C += A * B, where:
 *   + A is m x p
 *   + B is p x n
 *   + C is m x n
 */
template <typename splitint_t, typename accumulator_t, typename fp_t, typename uint_t>
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const std::vector<fp_t> &B,
                         const size_t m, const size_t p, const size_t n, const size_t num_splits) {

    const size_t bits_in_accumulator = std::numeric_limits<accumulator_t>::digits;
    const size_t bits_in_integer = std::numeric_limits<splitint_t>::digits;
    assert(bits_in_integer <= bits_in_accumulator / 2);
    const size_t alpha = std::floor((bits_in_accumulator - log2(n)) / 2);
    const size_t bits_per_slice = std::min(bits_in_integer, static_cast<size_t>(alpha));

    auto splitA = splitFloatToInt<splitint_t, fp_t, uint_t>
        (A, m, p, normalisationDimension::byRows, num_splits, bits_per_slice);

    auto splitB = splitFloatToInt<splitint_t, fp_t, uint_t>
        (B, p, n, normalisationDimension::byCols, num_splits, bits_per_slice);

    return computeProductsWithFloatingPlointAccumulation<splitint_t, accumulator_t, fp_t, uint_t>(splitA, splitB, bits_per_slice);
}

template
std::vector<float> gemmi<int8_t, int32_t, float, uint32_t> (const std::vector<float> &A, const std::vector<float> &B,
                         const size_t m, const size_t p, const size_t n, const size_t num_splits);
template
std::vector<double> gemmi<int8_t, int32_t, double, uint64_t> (const std::vector<double> &A, const std::vector<double> &B,
                         const size_t m, const size_t p, const size_t n, const size_t num_splits);
