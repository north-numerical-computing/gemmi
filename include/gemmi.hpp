#include <bit>
#include <cassert>
#include <vector>

#include <iostream>

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

// TODO: I'm really unhappy with this function taking uing_t as a template
// parameter, and having to propagate this value all the way from gemmi().
// This type should be inferred from fp_t.
template <typename splitint_t, typename fp_t, typename uint_t>
struct MatrixSplit {
    size_t m;
    size_t n;
    size_t numSplits;
    normalisationDimension dimension;

    std::vector<fp_t> matrix;
    std::vector<splitint_t> memory;
    std::vector<fp_t> powersVector;
    std::vector<int> scalingExponents;

    MatrixSplit(const size_t m, const size_t n, const size_t numSplits,
                const normalisationDimension dimension,
                const std::vector<fp_t>& matrix,
                std::vector<splitint_t>& memory,
                std::vector<fp_t>& powersVector,
                std::vector<int>& scalingExponents) :
                m(m), n(n), numSplits(numSplits), dimension(dimension),
                matrix(matrix), memory(memory),
                powersVector(powersVector),
                scalingExponents(scalingExponents) {}

    MatrixSplit(const size_t m, const size_t n, const size_t numSplits,
                const normalisationDimension dimension, const std::vector<fp_t>& matrix) :
                m(m), n(n), numSplits(numSplits), dimension(dimension),
                matrix(matrix) {
                    this->memory.resize(m * n * numSplits);
                    this->powersVector.resize(this->otherDimension());
                    this->scalingExponents.resize(this->otherDimension());
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
        const size_t iStride = this->iStride();
        const size_t jStride = this->jStride();
        for (size_t i = 0; i < this->otherDimension(); i++) {
            this->powersVector[i] = 0.0;
            for (size_t j = 0; j < this->innerProductDimension(); j++) {
                const size_t index = i * iStride + j * jStride;
                this->powersVector[i] = std::max(this->powersVector[i],
                                                  std::abs(this->matrix[index]));
            }
            // Compute the smallest power of 2 that is strictly greater than the
            // maximum value in the row/column.
            // NOTE 1: This is not the same technique used in uoi24.
            // NOTE 2: I use exponents instead of powers of 2, as I need the former
            //         to shift correctly.
            frexp(this->powersVector[i], this->scalingExponents.data() + i);
            const auto largest_log = log2(this->powersVector[i]);
            auto temp = ceil(largest_log) +
                        (ceil(largest_log) == floor(largest_log) ? 1 : 0);
            this->powersVector[i] = std::ldexp(1.0, temp);
        }
    }

    void computeSplitsWithTruncation(const size_t bitsPerSlice) {
        // Compute splits one row/column at a time.
        auto nunExpBits = numExpBits<fp_t>();
        auto nunFracBits = numFracBits<fp_t>();
        auto iStride = this->iStride();
        auto jStride = this->jStride();
        std::vector<uint_t> tmp (this->innerProductDimension());
        std::vector<bool> sign (this->innerProductDimension());
        for (size_t i = 0; i < this->otherDimension(); i++) {
            // Get binary representation of normalised row/column.
            for (size_t j = 0; j < this->innerProductDimension(); j++) {
                const size_t index = i * iStride + j * jStride;
                fp_t value = this->matrix[index];             // powersVector[i];
                tmp[j] = std::bit_cast<uint_t>(value);        // To bitstring.
                sign[j] = std::signbit(value);                // Extract sign.
                tmp[j] &= (~(uint_t)(0)) >> (nunExpBits + 1); // Remove exponent.
                // Restore implicit bit for normal numbers.
                // TODO: NaNs and infs are currently not supported..
                if (std::fpclassify(value) == FP_NORMAL)
                    tmp[j] |= ((uint_t)1 << (nunFracBits - 1));
            }

            // Create bitmask.
            const uint_t smallBitmask = (1 << bitsPerSlice) - 1;
            // Perform the split.
            for (size_t j = 0; j < this->innerProductDimension(); j++) {
                int16_t shiftCounter = nunFracBits - bitsPerSlice;
                int currentExponent;
                frexp(this->matrix[i * iStride + j * jStride], &currentExponent);
                int16_t exponentDifference = scalingExponents[i] - currentExponent;
                for (size_t slice = 0; slice < numSplits; slice++) {
                    if (exponentDifference > (signed)bitsPerSlice) {
                        exponentDifference -= bitsPerSlice;
                    } else {
                        shiftCounter += exponentDifference;
                        exponentDifference = 0;
                        uint_t bitmask = shiftCounter > 0 ?
                            smallBitmask << shiftCounter :
                            smallBitmask >> -shiftCounter;
                        uint_t currentSlice = tmp[j] & bitmask;
                        uint_t current_split = shiftCounter > 0 ?
                            currentSlice >> shiftCounter :
                            currentSlice << -shiftCounter;
                        splitint_t value = (splitint_t)(current_split) * (sign[j] ? -1 : 1);
                        this->memory[i * iStride + j * jStride + slice * this->matrix.size()] = value;
                        shiftCounter -= bitsPerSlice;
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
                                              const size_t numSplits,
                                              const size_t bitsPerSlice) {
    auto splits = MatrixSplit<splitint_t, fp_t, uint_t>(m, n, numSplits, dimension, A);
    splits.computeNormalisationVectors();
    splits.computeSplitsWithTruncation(bitsPerSlice);
    //splits.computeSplitsWithTruncation(bitsPerSlice);

    return splits;
}

template <typename splitint_t, typename accumulator_t,
          typename fp_t, typename uint_t>
std::vector<fp_t> mergeFloatfromInt(const MatrixSplit<splitint_t, fp_t, uint_t> &A,
                                    const size_t bitsPerSlice) {
    std::vector<fp_t> C (A.m * A.n, 0.0);

    for (size_t i = 0; i < A.m; i++) {
        uint_t tmp = 0;
        for (size_t j = 0; j < A.n; j++) {
            int8_t shiftValue = numFracBits<fp_t>() - bitsPerSlice;
            for (size_t iBlock = 0; iBlock < A.numSplits; iBlock++) {
                uint_t slice = A.memory[i + j * A.m + iBlock * A.m * A.n];
                uint_t new_slice = shiftValue > 0 ?
                    slice << shiftValue :
                    slice >> -shiftValue;
                tmp |= new_slice;
                shiftValue -= bitsPerSlice;
            }
            C[i + j * A.m] = std::ldexp(tmp, -(int)numFracBits<fp_t>()) *
                             A.powersVector[i];
        }
    }

    return C;
}

/* Compute exact products of slices of A and B. */
template <typename splitint_t, typename accumulator_t, typename fp_t, typename uint_t>
void computeExactIntegerGEMM(const MatrixSplit<splitint_t, fp_t, uint_t> &A, 
                             const MatrixSplit<splitint_t, fp_t, uint_t> &B, std::vector<accumulator_t> &C,
                             size_t iBlock, size_t jBlock) {
    for (size_t i = 0; i < A.m; i++) {
        for (size_t j = 0; j < B.n; j++) {
            for (size_t k = 0; k < A.n; k++) {
                C[i + j * A.m] += A.memory[i + k * A.m + iBlock * A.m * A.n] *
                                  B.memory[k + j * B.m + jBlock * B.m * B.n];
            }
        }
    }
}

/* Accumulate products using the technique in:
 *
 *    Ootomo H., Ozaki K., Yokota R. DGEMM on integer matrix multiplication
 *    unit. Int. J. High Performance Comput. App. 2024;38(4):297-313.
 *    DOI: 10.1177/10943420241239588
 *
 * Integer products are accumulated in floating-point arithmetic one by one.
 */
template <typename splitint_t, typename accumulator_t, typename fp_t, typename uint_t>
std::vector<fp_t> computeProductsWithFloatingPlointAccumulation(const MatrixSplit<splitint_t, fp_t, uint_t> &A,
                                  const MatrixSplit<splitint_t, fp_t, uint_t> &B,
                                  const size_t bitsPerSlice) {

    std::vector<fp_t > C (A.m * B.n);

    for (size_t iBlock = 0; iBlock < A.numSplits; iBlock++) {
        for (size_t jBlock = 0; jBlock < B.numSplits - iBlock; jBlock++) {
            std::vector<accumulator_t> accumulator (A.m * B.n, 0.0);
            computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t, uint_t>(A, B, accumulator, iBlock, jBlock);
            for (size_t i = 0; i < A.m; i++) {
                for (size_t j = 0; j < B.n; j++) {
                    fp_t scaledSum = std::ldexp(accumulator[i + j * A.m], -(iBlock + 1 + jBlock + 1) * bitsPerSlice);
                    fp_t scalingFactor = A.powersVector[i] * B.powersVector[j];
                    C[i + j * A.m] += scaledSum * scalingFactor;
                }
            }
        }
    }

    return C;
}

/* Accumulate products using the technique in:
 *
 *    Uchino Y., Ozaki K., Imamura T. Performance enanchcement of the Ozaki
 *    scheme on integer matrix multiplication unit. arXiv:2409.13313 [cs.DC]. 2024.
 *    DOI: 10.48550/arXiv.2409.13313
 *
 * Integer products are accumulated in integer arithmetic along the diagonal, and
 * in floating-point arithmetic across diagonals.
 */
template <typename splitint_t, typename accumulator_t, typename fp_t, typename uint_t>
std::vector<fp_t> computeProductsWithIntegerAccumulation(const MatrixSplit<splitint_t, fp_t, uint_t> &A,
                                  const MatrixSplit<splitint_t, fp_t, uint_t> &B,
                                  const size_t bitsPerSlice) {

    std::vector<fp_t > C (A.m * B.n);

    // Here I'm ignoring the products below the main anti-diagonal, as done in the original
    // paper. To compute all products, the following line could be changed to:
    // size t numSplits = A.numSplits + B.numSplits - 2;
    // NOTE: this is different from previous work, as I allow a different number of splits
    // for A and B.
    // TODO: I should test that this works when A and B have very different numbers of splits
    // (for example, 2 vs 15). Are the high-diagonal products computed? In order to test this,
    // I need to modify gemmi to use different numbers of splits for A and B.
    size_t numSplits = std::max(A.numSplits, B.numSplits) - 1;
    for (size_t diagonal = 0; diagonal <= numSplits; diagonal++) {
        size_t Aindex = 0;
        int Bindex = diagonal;
        std::vector<accumulator_t> accumulator (A.m * B.n, 0.0);
        while (Aindex < A.numSplits && Bindex >= 0) {
            computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t, uint_t>(A, B, accumulator, Aindex, Bindex);
            Aindex++;
            Bindex--;
        }
        for (size_t i = 0; i < A.m; i++) {
                for (size_t j = 0; j < B.n; j++) {
                    fp_t scaledSum = std::ldexp(accumulator[i + j * A.m], -(diagonal + 2) * bitsPerSlice);
                    fp_t scalingFactor = A.powersVector[i] * B.powersVector[j];
                    C[i + j * A.m] += scaledSum * scalingFactor;
                }
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
                         const size_t m, const size_t p, const size_t n, const size_t numSplits) {

    const size_t bitsInAccumulator = std::numeric_limits<accumulator_t>::digits;
    const size_t bitsPerInteger = std::numeric_limits<splitint_t>::digits;
    assert(bitsPerInteger <= bitsInAccumulator / 2);
    const size_t alpha = std::floor((bitsInAccumulator - log2(n)) / 2);
    const size_t bitsPerSlice = std::min(bitsPerInteger, static_cast<size_t>(alpha));

    auto splitA = splitFloatToInt<splitint_t, fp_t, uint_t>
        (A, m, p, normalisationDimension::byRows, numSplits, bitsPerSlice);

    auto splitB = splitFloatToInt<splitint_t, fp_t, uint_t>
        (B, p, n, normalisationDimension::byCols, numSplits, bitsPerSlice);

    // return computeProductsWithFloatingPointAccumulation<splitint_t, accumulator_t, fp_t, uint_t>(splitA, splitB, bitsPerSlice);
    return computeProductsWithIntegerAccumulation<splitint_t, accumulator_t, fp_t, uint_t>(splitA, splitB, bitsPerSlice);
}

template
std::vector<float> gemmi<int8_t, int32_t, float, uint32_t> (const std::vector<float> &A, const std::vector<float> &B,
                         const size_t m, const size_t p, const size_t n, const size_t numSplits);
template
std::vector<double> gemmi<int8_t, int32_t, double, uint64_t> (const std::vector<double> &A, const std::vector<double> &B,
                         const size_t m, const size_t p, const size_t n, const size_t numSplits);
