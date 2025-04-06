#include <bit>
#include <cassert>
#include <vector>
#include <iostream>

/**
 * @file gemmi.hpp
 * @brief Integer matrix multiplication using the Ozaki scheme.
 */

/**
 * @brief Number of bits in the exponent field of a floating-point type.
 * @tparam fp_t Floating-point type (e.g., float, double).
 * @return Number of bits in the exponent field.
 */
template <typename fp_t> size_t computeNumExpBits() {return 0;}
template <> size_t computeNumExpBits<float>() {return 8;}
template <> size_t computeNumExpBits<double>() {return 11;}

/**
 * @brief Number of bits in the fraction field of a floating-point type.
 * @tparam fp_t Floating-point type (e.g., float, double).
 * @return Number of bits in the fraction field.
 */
template <typename fp_t> size_t computeNumFracBits() {return 0;}
template <> size_t computeNumFracBits<float>() {return 24;}
template <> size_t computeNumFracBits<double>() {return 53;}

/**
 * @brief Map floating-point type to its corresponding storage format.
 * @tparam fp_t Floating-point type (e.g., float, double).
 */
template <typename fp_t> struct get_storage_format;
template <> struct get_storage_format<float> {using storage_format = uint32_t;};
template <> struct get_storage_format<double> {using storage_format = uint64_t;};

/**
 * @brief Enum to specify the dimension used for normalization.
 */
enum class normalisationDimension {
    byRows, ///< Normalise by rows (matrix on the left of the product).
    byCols  ///< Normalise by columns (matrix on the right of the product).
};

/**
 * @brief Enum to specify the splitting strategy to use.
 */
enum class splittingStrategy {
    roundToNearest, ///< Split using round-to-nearest.
    bitMasking      ///< Split using bit masking (truncation).
};

/**
 * @brief Enum to specify the accumulation strategy to use.
 */
enum class accumulationStrategy {
    floatingPoint, ///< Accumulate products in floating-point arithmetic.
    integer        ///< Accumulate products in integer arithmetic.
};

/**
 * @brief Enum to specify the multiplication strategy to use.
 */
enum class multiplicationStrategy {
    full,       ///< Compute all products.
    reduced     ///< Only compute products above the main anti-diagonal.
};

/**
 * @brief Class to store the matrix slices for the Ozaki scheme.
 * @tparam splitint_t Type used to store the integer slices.
 * @tparam fp_t Floating-point type (e.g., float, double).
 */
template <typename splitint_t, typename fp_t>
struct MatrixSplit {
    size_t m;                         ///< Number of rows in the matrix.
    size_t n;                         ///< Number of columns in the matrix.
    splittingStrategy splitType;      ///< Splitting strategy used.
    size_t numSplits;                 ///< Number of splits to use.
    size_t bitsPerSlice;              ///< Number of bits per slice.
    normalisationDimension dimension; ///< Dimension along wich to normalize.

    std::vector<fp_t> matrix;          ///< Original matrix.
    std::vector<splitint_t> memory;    ///< Memory to store the split slices.
    std::vector<fp_t> powersVector;    ///< Normalisation vector.
    std::vector<int> scalingExponents; ///< Scaling exponents.

    using uint_t = typename get_storage_format<fp_t>::storage_format;

    /**
     * @brief Construct a MatrixSplit object and compute the splits.
     * @param matrix Original matrix.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param splitType Splitting strategy.
     * @param numSplits Number of splits.
     * @param bitsPerSlice Number of bits per slice.
     * @param dimension Normalization dimension.
     */
    MatrixSplit(const std::vector<fp_t>& matrix, const size_t m, const size_t n,
                const splittingStrategy splitType, size_t numSplits, size_t bitsPerSlice,
                const normalisationDimension dimension) :
                m(m), n(n), splitType(splitType), numSplits(numSplits), bitsPerSlice(bitsPerSlice),
                dimension(dimension), matrix(matrix) {
                    this->memory.resize(m * n * numSplits);
                    this->powersVector.resize(this->otherDimension());
                    this->scalingExponents.resize(this->otherDimension());
                    this->computeNormalisationVectors();
                    switch (splitType) {
                        case splittingStrategy::roundToNearest:
                            this->computeSplitsWithRoundToNearest();
                            break;
                        case splittingStrategy::bitMasking:
                            this->computeSplitsWithBitMasking();
                            break;
                     }
                }

    /**
     * @brief Return the dimension along which the inner product is calculated.
     * @return Inner product dimension.
     */
    size_t innerProductDimension() {
        return (dimension == normalisationDimension::byRows) ? n : m;
    }

    /**
     * @brief Return the dimension not used in the inner product.
     * @return Dimension not used in the inner product.
     */
    size_t otherDimension() {
        return (dimension == normalisationDimension::byRows) ? m : n;
    }

    /**
     * @brief Compute the stride for the i-dimension.
     * @return Stride for the i-dimension.
     */
    size_t iStride() {
        return (dimension == normalisationDimension::byRows) ? 1 : m;
    }

    /**
     * @brief Compute the stride for the j-dimension.
     * @return Stride for the j-dimension.
     */
    size_t jStride() {
        return (dimension == normalisationDimension::byRows) ? m : 1;
    }

    /**
     * @brief Compute normalization vectors for the matrix.
     */
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
            // NOTE 1: This is not the technique used in uoi24.
            // NOTE 2: I use exponents instead of powers of 2, as I need the former
            //         to shift correctly.
            frexp(this->powersVector[i], this->scalingExponents.data() + i);
            this->powersVector[i] = std::ldexp(1.0, this->scalingExponents[i]);
        }
    }

    /**
     * @brief Split the matrix using round-to-nearest.
     *
     * This is an implementation of Algorithm 8 in:
     *
     *    Uchino Y., Ozaki K., Imamura T. Performance enanchcement of the Ozaki
     *    scheme on integer matrix multiplication unit. arXiv:2409.13313 [cs.DC]. 2024.
     *    DOI: 10.48550/arXiv.2409.13313
     *
     * Integer products are accumulated in integer arithmetic along the diagonal, and in
     * floating-point arithmetic across diagonals.
     */
    void computeSplitsWithRoundToNearest() {
        this->splitType = splittingStrategy::roundToNearest;
        auto iStride = this->iStride();
        auto jStride = this->jStride();
        auto localMatrix = this->matrix;
        for (size_t slice = 0; slice < numSplits; slice++) {
            for (size_t i = 0; i < this->otherDimension(); i++) {
                fp_t sigma = ldexp(0.75, computeNumFracBits<fp_t>() - this->bitsPerSlice * slice + 1 - this->bitsPerSlice) * powersVector[i];
                for (size_t j = 0; j < this->innerProductDimension(); j++) {
                    auto value = (localMatrix[i * iStride + j * jStride] + sigma);
                    value -= sigma;
                    localMatrix[i * iStride + j * jStride] -= value;
                    value = value / powersVector[i] * ldexp(1.0, this->bitsPerSlice * slice + this->bitsPerSlice - 1);
                    this->memory[i * iStride + j * jStride + slice * this->matrix.size()] = value;
                }
            }
        }
    }

    /**
     * @brief Split the matrix using bit masking, which is equivalent to truncation.
     *
     * This is an implementation of Algorithm 4 in:
     *
     *    Ootomo H., Ozaki K., Yokota R. DGEMM on integer matrix multiplication
     *    unit. Int. J. High Performance Comput. App. 2024;38(4):297-313.
     *    DOI: 10.1177/10943420241239588
     *
     * Integer products are accumulated in floating-point arithmetic one by one.
     */
    void computeSplitsWithBitMasking() {
        this->splitType = splittingStrategy::bitMasking;
        // Compute splits one row/column at a time.
        auto numExpBits = computeNumExpBits<fp_t>();
        auto numFracBits = computeNumFracBits<fp_t>();
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
                uint_t bitmask = (~((uint_t)(0))) >> (numExpBits + 1);
                tmp[j] = tmp[j] & bitmask; // Remove exponent and sign.
                // Restore implicit bit for normal numbers.
                // NOTE: NaNs and infs are currently not supported.
                if (std::fpclassify(value) == FP_NORMAL)
                    tmp[j] |= ((uint_t)1 << (numFracBits - 1));
            }

            // Create bitmask.
            const uint_t smallBitmask = (1 << this->bitsPerSlice) - 1;
            // Perform the split.
            for (size_t j = 0; j < this->innerProductDimension(); j++) {
                int16_t shiftCounter = numFracBits - this->bitsPerSlice;
                int currentExponent;
                frexp(this->matrix[i * iStride + j * jStride], &currentExponent);
                int16_t exponentDifference = scalingExponents[i] - currentExponent;
                for (size_t slice = 0; slice < numSplits; slice++) {
                    if (exponentDifference > (signed)this->bitsPerSlice) {
                        exponentDifference -= this->bitsPerSlice;
                    } else {
                        shiftCounter += exponentDifference;
                        exponentDifference = 0;
                        uint_t bitmask = shiftCounter > 0 ?
                            smallBitmask << shiftCounter :
                            smallBitmask >> -shiftCounter;
                        uint_t currentSlice = tmp[j] & bitmask;
                        uint_t currentSplit = shiftCounter > 0 ?
                            currentSlice >> shiftCounter :
                            currentSlice << -shiftCounter;
                        splitint_t value = (splitint_t)(currentSplit) * (sign[j] ? -1 : 1);
                        this->memory[i * iStride + j * jStride + slice * this->matrix.size()] = value;
                        shiftCounter -= this->bitsPerSlice;
                    }
                }
            }
        }
    }
};

/**
 * @brief Compute the exact integer GEMM (General Matrix-Matrix Multiplication).
 * @tparam splitint_t Integer type used for splits.
 * @tparam accumulator_t Accumulator type.
 * @tparam fp_t Floating-point type (e.g., float, double).
 * @param A Slices of matrix A.
 * @param B Slices of matrix B.
 * @param C Output matrix to store the result.
 * @param iBlock Block index for matrix A.
 * @param jBlock Block index for matrix B.
 */
template <typename splitint_t, typename accumulator_t, typename fp_t>
void computeExactIntegerGEMM(const MatrixSplit<splitint_t, fp_t> &A,
                             const MatrixSplit<splitint_t, fp_t> &B, std::vector<accumulator_t> &C,
                             size_t iBlock, size_t jBlock) {
    for (size_t i = 0; i < A.m; i++) {
        for (size_t j = 0; j < B.n; j++) {
            for (size_t ell = 0; ell < A.n; ell++) {
                C[i + j * A.m] += A.memory[i + ell * A.m + iBlock * A.m * A.n] *
                                  B.memory[ell + j * B.m + jBlock * B.m * B.n];
            }
        }
    }
}

/**
 * @brief Compute the scaling constant for a given splitting strategy.
 * 
 * This function computes the scaling constant based on the splitting strategy used
 * for the matrices A and B. The scaling constant is used to adjust the final result
 * based on the number of bits used in the split.
 * @tparam splitint_t Integer type used for splits.
 * @tparam fp_t Floating-point type (e.g., float, double).
 * @param A Slices of matrix A.
 * @param B Slices of matrix B.
 * @return Scaling constant.
 */
template <typename splitint_t, typename fp_t>
fp_t computeScalingConstantForSplittingStrategy(const MatrixSplit<splitint_t, fp_t> &A,
                                                 const MatrixSplit<splitint_t, fp_t> &B) {
    // When splitting with round-to-nearest, the first slice has bitsPerSlice - 1 bits, 
    // and we need to account for this when scaling the final result.
    fp_t scalingConstant = 1.0;
    scalingConstant *= A.splitType == splittingStrategy::roundToNearest ? 2.0 : 1.0;
    scalingConstant *= B.splitType == splittingStrategy::roundToNearest ? 2.0 : 1.0;
    return scalingConstant;
}

/**
 * @brief Accumulate products in floating-point arithmetic.
 * 
 * This function accumulates the exact products of integer slices in floating-point
 * arithmetic, using the algorithm described in:
 *
 *    Ootomo H., Ozaki K., Yokota R. DGEMM on integer matrix multiplication
 *    unit. Int. J. High Performance Comput. App. 2024;38(4):297-313.
 *    DOI: 10.1177/10943420241239588
 * 
 * @tparam splitint_t Integer type used for splits.
 * @tparam accumulator_t Accumulator type.
 * @tparam fp_t Floating-point type (e.g., float, double).
 * @param A Slices of matrix A.
 * @param B Slices of matrix B.
 * @param bitsPerSlice Number of bits per slice.
 * @param numDiagonals Number of diagonals to compute.
 * @return Resulting matrix C.
 *
 */
template <typename splitint_t, typename accumulator_t, typename fp_t>
std::vector<fp_t> computeProductsWithFloatingPointAccumulation(const MatrixSplit<splitint_t, fp_t> &A,
                                  const MatrixSplit<splitint_t, fp_t> &B,
                                  const size_t bitsPerSlice,
                                  const size_t numDiagonals) {

    std::vector<fp_t > C (A.m * B.n);

    auto scalingConstant = computeScalingConstantForSplittingStrategy(A, B);

    for (size_t diagonal = 0; diagonal <= numDiagonals; diagonal++) {
        int Aindex = diagonal < A.numSplits - 1 ? diagonal : A.numSplits - 1;
        size_t Bindex = diagonal > A.numSplits - 1 ? diagonal - A.numSplits + 1 : 0;
        while (Aindex >= 0 && Bindex <= std::min(diagonal, B.numSplits - 1)) {
            std::vector<accumulator_t> accumulator (A.m * B.n, 0.0);
            computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t>(A, B, accumulator, Aindex, Bindex);
            for (size_t i = 0; i < A.m; i++) {
                for (size_t j = 0; j < B.n; j++) {
                    fp_t scaledSum = std::ldexp(accumulator[i + j * A.m], -(Aindex + 1 + Bindex + 1) * bitsPerSlice);
                    fp_t scalingFactor = A.powersVector[i] * B.powersVector[j] * scalingConstant;
                    C[i + j * A.m] += scaledSum * scalingFactor;
                }
            }
            Aindex--;
            Bindex++;
        }
    }

    return C;
}

/**
 * @brief Accumulate products in integer arithmetic.
 * 
 * This function accumulates the exact products of integer slices in integer
 * arithmetic along each anti-diagonal, and in floating-point arithmetic
 * across diagonals. It uses the algorithm described in:
 *
 *    Uchino Y., Ozaki K., Imamura T. Performance enanchcement of the Ozaki
 *    scheme on integer matrix multiplication unit. arXiv:2409.13313 [cs.DC]. 2024.
 *    DOI: 10.48550/arXiv.2409.13313
 *
 * @tparam splitint_t Integer type used for splits.
 * @tparam accumulator_t Accumulator type.
 * @tparam fp_t Floating-point type (e.g., float, double).
 * @param A Slices of matrix A.
 * @param B Slices of matrix B.
 * @param bitsPerSlice Number of bits per slice.
 * @param numDiagonals Number of diagonals to compute.
 * @return Resulting matrix C.
 */
template <typename splitint_t, typename accumulator_t, typename fp_t>
std::vector<fp_t> computeProductsWithIntegerAccumulation(const MatrixSplit<splitint_t, fp_t> &A,
                                  const MatrixSplit<splitint_t, fp_t> &B,
                                  const size_t bitsPerSlice,
                                  const size_t numDiagonals) {

    std::vector<fp_t > C (A.m * B.n);

    auto scalingConstant = computeScalingConstantForSplittingStrategy(A, B);

    for (size_t diagonal = 0; diagonal <= numDiagonals; diagonal++) {
        int Aindex = diagonal < A.numSplits - 1 ? diagonal : A.numSplits - 1;
        size_t Bindex = diagonal > A.numSplits - 1 ? diagonal - A.numSplits + 1 : 0;
        std::vector<accumulator_t> accumulator (A.m * B.n, 0.0);
        while (Aindex >= 0 && Bindex <= std::min(diagonal, B.numSplits - 1)) {
            computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t>(A, B, accumulator, Aindex, Bindex);
            Aindex--;
            Bindex++;
        }
        for (size_t i = 0; i < A.m; i++) {
                for (size_t j = 0; j < B.n; j++) {
                    fp_t scaledSum = std::ldexp(accumulator[i + j * A.m], -(diagonal + 2) * bitsPerSlice);
                    fp_t scalingFactor = A.powersVector[i] * B.powersVector[j]  * scalingConstant;
                    C[i + j * A.m] += scaledSum * scalingFactor;
                }
            }
    }

    return C;
}

/**
 * @brief Compute the matrix product C = C + A * B.
 * @tparam fp_t Floating-point type of the matrix elements.
 * @tparam splitint_t Integer type used for splits.
 * @tparam accumulator_t Accumulator type.
 * @param A Matrix A.
 * @param B Matrix B.
 * @param m Number of rows in A.
 * @param k Number of columns in A and rows in B.
 * @param n Number of columns in B.
 * @param numSplitsA Number of splits for the matrix A.
 * @param numSplitsB Number of splits for the matrix B.
 * @param splitType Splitting strategy.
 * @param multType Multiplication strategy.
 * @param accType Accumulation strategy.
 * @return Resulting matrix product.
 */
template <typename fp_t, typename splitint_t, typename accumulator_t>
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const std::vector<fp_t> &B,
                         const size_t m, const size_t k, const size_t n,
                         const size_t numSplitsA, const size_t numSplitsB,
                         const splittingStrategy splitType = splittingStrategy::roundToNearest,
                         const accumulationStrategy accType = accumulationStrategy::floatingPoint,
                         const multiplicationStrategy multType = multiplicationStrategy::reduced) {

    const size_t bitsInAccumulator = std::numeric_limits<accumulator_t>::digits;
    const size_t bitsPerInteger = std::numeric_limits<splitint_t>::digits;
    assert(bitsPerInteger <= bitsInAccumulator / 2);
    const size_t alpha = std::floor((bitsInAccumulator - log2(n)) / 2);
    const size_t bitsPerSlice = std::min(bitsPerInteger, static_cast<size_t>(alpha));

    auto splitA = MatrixSplit<splitint_t, fp_t>(A, m, k, splitType, numSplitsA, bitsPerSlice, normalisationDimension::byRows);
    auto splitB = MatrixSplit<splitint_t, fp_t>(B, k, n, splitType, numSplitsB, bitsPerSlice, normalisationDimension::byCols);

    size_t numDiagonals;
    switch (multType) {
        case multiplicationStrategy::reduced:
            // Products below the main anti-diagonal are ignored.
            numDiagonals = std::max(splitA.numSplits, splitB.numSplits) - 1;
            break;
        case multiplicationStrategy::full:
            // All products are computed.
            numDiagonals = splitA.numSplits + splitB.numSplits - 1;
            break;
        default:
            std::cerr << "Unknown multiplication strategy requested.";
            exit(1);
    }

    switch (accType) {
        case accumulationStrategy::floatingPoint:
            return computeProductsWithFloatingPointAccumulation<splitint_t, accumulator_t, fp_t>(splitA, splitB, bitsPerSlice, numDiagonals);
        case accumulationStrategy::integer:
            return computeProductsWithIntegerAccumulation<splitint_t, accumulator_t, fp_t>(splitA, splitB, bitsPerSlice, numDiagonals);
        default:
            std::cerr << "Unknown accumulation strategy requested.";
            exit(1);
    }
}
template <typename fp_t, typename splitint_t, typename accumulator_t>
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const std::vector<fp_t> &B,
                         const size_t m, const size_t k, const size_t n, const size_t numSplits) {
    return gemmi <fp_t, splitint_t, accumulator_t> (A, B, m, k, n, numSplits, numSplits);
}