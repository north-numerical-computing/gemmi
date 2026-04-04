#include <bit>
#include <cassert>
#include <cstdlib>
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
template <typename fp_t> struct getStorageFormat;
template <> struct getStorageFormat<float> {using storage_format = uint32_t;};
template <> struct getStorageFormat<double> {using storage_format = uint64_t;};

/**
 * @brief Get the exponent of a floating-point value.
 * @tparam fp_t Floating-point type (e.g., float, double).
 * @param value The floating-point value.
 * @return The value stored in the exponent field of the value, as an integer.
 */
template <typename fp_t>
int getFloatingPointExponent(fp_t value) {
    if (value == 0.0) {
        return 0; // Exponent for zero is typically represented as all zeros.
    } else {
        return std::max(std::numeric_limits<fp_t>::min_exponent,
                        std::ilogb(std::abs(value)) + 1);
    }
}

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
    bitMasking,       ///< Split using bit masking (truncation).
    unsignedEncoding, ///< Split using unsigned slice encoding.
    roundToNearest    ///< Split using round-to-nearest.
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

    using uint_t = typename getStorageFormat<fp_t>::storage_format;
    using wideint_t = std::conditional_t<(sizeof(splitint_t) < sizeof(int)), int, std::intmax_t>;

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
                        case splittingStrategy::bitMasking:
                            this->computeSplitsWithBitMasking();
                            break;
                        case splittingStrategy::unsignedEncoding:
                            this->computeSplitsWithUnsignedEncoding();
                            break;
                        case splittingStrategy::roundToNearest:
                            this->computeSplitsWithRoundToNearest();
                            break;
                     }
                }

    /**
     * @brief Return the dimension along which the inner product is calculated.
     * @return Inner product dimension.
     */
    size_t innerProductDimension() const {
        return (dimension == normalisationDimension::byRows) ? n : m;
    }

    /**
     * @brief Return the dimension not used in the inner product.
     * @return Dimension not used in the inner product.
     */
    size_t otherDimension() const {
        return (dimension == normalisationDimension::byRows) ? m : n;
    }

    /**
     * @brief Compute the stride for the i-dimension.
     * @return Stride for the i-dimension.
     */
    size_t iStride() const {
        return (dimension == normalisationDimension::byRows) ? 1 : m;
    }

    /**
     * @brief Compute the stride for the j-dimension.
     * @return Stride for the j-dimension.
     */
    size_t jStride() const {
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
                const size_t matrixIndex = i * iStride + j * jStride;
                this->powersVector[i] = std::max(this->powersVector[i],
                                                  std::abs(this->matrix[matrixIndex]));
            }
            // Compute the smallest power of 2 that is strictly greater than the
            // maximum value in the row/column.
            // NOTE 1: This is not the technique used in uoi24.
            // NOTE 2: I use exponents instead of powers of 2, as I need the former
            //         to shift correctly.
            this->scalingExponents[i] = getFloatingPointExponent(this->powersVector[i]);
            this->powersVector[i] = std::ldexp(1.0, this->scalingExponents[i]);
        }
    }

    /**
     * @brief Compute the bit offset for a given slice.
     * @param slice The slice for which to compute the bit offset.
     * @return The bit offset for the specified slice.
     */
    int computeSliceBitOffset(size_t slice) const {
        switch (splitType) {
            case splittingStrategy::bitMasking:
                // Slice k -> k * b
                return static_cast<int>((slice + 1) * bitsPerSlice);

            case splittingStrategy::roundToNearest:
                // Slice 0 -> b - 1
                // Slice k -> k * b - 1
                return static_cast<int>((slice + 1) * bitsPerSlice - 1);

            case splittingStrategy::unsignedEncoding:
                // Slice 0 -> b - 1
                // Slice k -> (b - 1) + k * (b + 1)
                return static_cast<int>((bitsPerSlice - 1) + slice * (bitsPerSlice + 1));

            // LCOV_EXCL_START
            default:
                std::abort();
            // LCOV_EXCL_STOP
        }
    }

    /**
     * @brief Compute the block fixed-point representation of a row/column of the matrix.
     * This function computes the fixed-point representation of a row/column of the matrix, which is used in the splitting algorithms. It extracts the significand and sign of each element in the row/column, and stores them in the provided vectors.
     * @param fraction Vector to store the fixed-point representation of the elements.
     * @param sign Vector to store the signs of the elements.
     * @param i The index of the row/column for which to compute the fixed-point representation.
     */
    void computeFixedPointRepresentationVector(std::vector<uint_t> &fraction, std::vector<bool> &sign, size_t i) {
        auto numFracBits = computeNumFracBits<fp_t>();
        auto iStride = this->iStride();
        auto jStride = this->jStride();
        for (size_t j = 0; j < this->innerProductDimension(); j++) {
                const size_t matrixIndex = i * iStride + j * jStride;
                fp_t value = this->matrix[matrixIndex];
                fraction[j] = std::bit_cast<uint_t>(value);
                sign[j] = std::signbit(value);
                uint_t bitmask = (1ull << (numFracBits - 1)) - 1;
                fraction[j] = fraction[j] & bitmask;
                // Restore implicit bit for normal numbers.
                // NOTE: NaNs and infs are currently not supported.
                if (std::fpclassify(value) == FP_NORMAL)
                    fraction[j] |= ((uint_t)1 << (numFracBits - 1));
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
     */
    void computeSplitsWithBitMasking() {
        this->splitType = splittingStrategy::bitMasking;
        // Compute splits one row/column at a time.
        auto numFracBits = computeNumFracBits<fp_t>();
        auto bitsPerSlice = this->bitsPerSlice;
        auto iStride = this->iStride();
        auto jStride = this->jStride();
        std::vector<uint_t> fraction (this->innerProductDimension());
        std::vector<bool> sign (this->innerProductDimension());
        for (size_t i = 0; i < this->otherDimension(); i++) {
            // Get binary representation of significands of normalised row/column.
            computeFixedPointRepresentationVector(fraction, sign, i);

            // Create bitmask.
            const uint_t smallBitmask = (1 << bitsPerSlice) - 1;

            // Perform the split.
            for (size_t j = 0; j < this->innerProductDimension(); j++) {
                // NOTE: I could have a special path for 0.
                int16_t shiftCounter = numFracBits - bitsPerSlice;
                int currentExponent = getFloatingPointExponent(this->matrix[i * iStride + j * jStride]);
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
                        uint_t currentSlice = fraction[j] & bitmask;
                        uint_t currentSplit = shiftCounter > 0 ?
                            currentSlice >> shiftCounter :
                            currentSlice << -shiftCounter;
                        splitint_t value = (splitint_t)(currentSplit) * (sign[j] ? -1 : 1);
                        this->memory[i * iStride + j * jStride + slice * this->matrix.size()] = value;
                        shiftCounter -= bitsPerSlice;
                    }
                }
            }
        }
    }

    /**
     * @brief Split the matrix using unsigned slice encoding.
     *
     * This is an implementation of the algorithm in section 3 of:
     *
     *    Schwarz A., Anders A., Brower C., Bayraktar H., Gunnels J., Clark K.,
     *    Xu R. G., Rodriguez S., Cayrols S., Tabaszewski P., Podlozhnyuk V.
     *    Guaranteed DGEMM accuracy while using reduced precision tensor cores
     *    throguh extensions of the Ozaki scheme. SCA/HPCAsia 2026.
     *    DOI: 10.1145/3773656.3773670
     */
    void computeSplitsWithUnsignedEncoding() {
        this->splitType = splittingStrategy::unsignedEncoding;
        // Compute splits one row/column at a time.
        auto numFracBits = computeNumFracBits<fp_t>();
        auto bitsPerSlice = this->bitsPerSlice;
        auto iStride = this->iStride();
        auto jStride = this->jStride();
        std::vector<uint_t> fraction (this->innerProductDimension());
        std::vector<bool> sign (this->innerProductDimension());
        for (size_t i = 0; i < this->otherDimension(); i++) {
            // Get binary representation of significands of normalised row/column.
            computeFixedPointRepresentationVector(fraction, sign, i);

            // Create bitmasks.
            // This algorithm uses bitsPerSlice bits for the first slice and bitsPerSlice + 1 bits for subsequent slices.
            // This choice is to keep the algorithm consistent with the bitmasking approach above.
            // NOTE: I am using a first slice with bitsPerSlice - 1 bits. This is not mentioned in sabb26, but it seems
            // necessary to do this, if I want to ensure that the first slice does not overflow if the second slice
            // is negative in two's complement.
            const uint_t smallBitmask = (1 << (bitsPerSlice - 1)) - 1;
            const uint_t largeBitmask = (1 << (bitsPerSlice + 1)) - 1;

            for (size_t j = 0; j < this->innerProductDimension(); j++) {

                // NOTE: I could have a special path for 0.
                int currentExponent = getFloatingPointExponent(this->matrix[i * iStride + j * jStride]);
                int16_t exponentDifference = scalingExponents[i] - currentExponent;

                // Boundary below slice 0, fully aligned to this entry.
                int16_t firstShift = numFracBits - (bitsPerSlice - 1) + exponentDifference;

                splitint_t value = 0;
                uint_t remainder = 0;

                if (!sign[j]) {
                    if (exponentDifference > (signed)(bitsPerSlice - 1)) {
                        // Too small to contribute explicit bits to slice 0.
                        value = 0;
                        remainder = fraction[j];
                    } else {
                        // Positive number: leading digit is truncation.
                        uint_t bitmask = (firstShift > 0)
                            ? (smallBitmask << firstShift)
                            : (smallBitmask >> -firstShift);

                        uint_t currentSlice = fraction[j] & bitmask;
                        uint_t currentSplit = (firstShift > 0)
                            ? (currentSlice >> firstShift)
                            : (currentSlice << -firstShift);

                        value = static_cast<splitint_t>(currentSplit);
                        remainder = fraction[j] & ((uint_t(1) << firstShift) - 1);
                    }
                } else {
                    // Negative number.
                    if (exponentDifference > (signed)(bitsPerSlice - 1)) {
                        // Too small to contribute explicit bits to slice 0:
                        // round toward -infinity gives -1.
                        value = -1;
                        remainder = ((1ull << (sizeof(uint_t) * 8 - 1)) - fraction[j]) | (uint_t(1) << (sizeof(uint_t) * 8 - 1));
                    } else {
                        // Normal truncation path for the leading slice.
                        uint_t bitmask = (firstShift > 0)
                            ? (smallBitmask << firstShift)
                            : (smallBitmask >> -firstShift);

                        uint_t currentSlice = fraction[j] & bitmask;
                        uint_t currentSplit = (firstShift > 0)
                            ? (currentSlice >> firstShift)
                            : (currentSlice << -firstShift);

                        value = -static_cast<splitint_t>(currentSplit);

                        uint_t lowMask = (uint_t(1) << firstShift) - 1;
                        uint_t lowBits = fraction[j] & lowMask;

                        if (lowBits != 0) {
                            value -= 1;  // round toward -infinity
                            remainder = (1ull << firstShift) - lowBits;
                        } else {
                            remainder = 0;
                        }
                    }
                }

                this->memory[i * iStride + j * jStride] = value;

                // From here on, the remainder is already aligned and nonnegative.
                int shiftCounter = firstShift - (bitsPerSlice + 1);
                exponentDifference = 0;

                // Split (nonnegative) remainder with unsigned slice encoding.
                for (size_t slice = 1; slice < numSplits; slice++) {
                    if (exponentDifference > (signed)(bitsPerSlice + 1)) {
                        exponentDifference -= (bitsPerSlice + 1);
                        if (sign[j]) {
                            this->memory[i * iStride + j * jStride + slice * this->matrix.size()] = static_cast<splitint_t>(largeBitmask);
                        }
                    } else {
                        shiftCounter += exponentDifference;
                        exponentDifference = 0;

                        uint_t bitmask = shiftCounter > 0 ?
                            largeBitmask << shiftCounter :
                            largeBitmask >> -shiftCounter;

                        uint_t currentSlice = remainder & bitmask;
                        uint_t currentSplit = shiftCounter > 0 ?
                            currentSlice >> shiftCounter :
                            currentSlice << -shiftCounter;

                        if (sign[j] && (shiftCounter + static_cast<int>(bitsPerSlice) + 1) > static_cast<int>(sizeof(uint_t)) * 8) {
                            int bitsUsed = sizeof(uint_t) * 8 - shiftCounter;
                            int bitsLost = (bitsPerSlice + 1) - bitsUsed;
                            uint_t maskToAdd = ((1ull << bitsLost) - 1) << bitsUsed;
                            currentSplit |= maskToAdd;
                        }

                        // Width of sub-leading slices is (bitsPerSlice + 1) bits.
                        const auto width  = bitsPerSlice + 1;
                        const auto cutoff = static_cast<wideint_t>(uint_t(1) << bitsPerSlice); // 2^b
                        const auto base   = static_cast<wideint_t>(uint_t(1) << width);        // 2^(b+1)
                        const auto digitMax = cutoff - 1;   //  2^b - 1
                        const auto digitMin = -cutoff;      // -2^b

                        // If the unsigned digit is in the upper half, propagate carry upward.
                        if (currentSplit >= static_cast<uint_t>(cutoff)) {
                            int nextSlice = static_cast<int>(slice) - 1;
                            while (true) {
                                const size_t prevIdx =
                                    i * iStride + j * jStride + nextSlice * this->matrix.size();

                                auto newValue =
                                    static_cast<wideint_t>(this->memory[prevIdx]) + 1;

                                if (newValue <= digitMax) {
                                    this->memory[prevIdx] = static_cast<splitint_t>(newValue);
                                    break;
                                } else {
                                    this->memory[prevIdx] = static_cast<splitint_t>(digitMin);
                                    nextSlice--;
                                }
                            }
                        }

                        // Store the slice as a signed (b+1)-bit value:
                        // [0, 2^b-1]      -> unchanged
                        // [2^b, 2^(b+1)-1] -> subtract 2^(b+1), yielding [-2^b, -1]
                        wideint_t signedDigit = static_cast<wideint_t>(currentSplit);
                        if (signedDigit >= cutoff) {
                            signedDigit -= base;
                        }

                        this->memory[
                            i * iStride + j * jStride + slice * this->matrix.size()
                        ] = static_cast<splitint_t>(signedDigit);

                        shiftCounter -= (bitsPerSlice + 1);
                    }
                }
            }
        }
    }

    /**
     * @brief Split the matrix using round-to-nearest.
     *
     * This is an implementation of Algorithm 8 in:
     *
     *    Uchino Y., Ozaki K., Imamura T. Performance enanchcement of the Ozaki
     *    scheme on integer matrix multiplication unit. Int. J. High Performance
     *    Comput. App. 2025;39(3):462–476. DOI: 10.1177/10943420241313064
     */
    void computeSplitsWithRoundToNearest() {
        this->splitType = splittingStrategy::roundToNearest;
        auto bitsPerSlice = this->bitsPerSlice;
        auto iStride = this->iStride();
        auto jStride = this->jStride();
        auto localMatrix = this->matrix;
        for (size_t slice = 0; slice < numSplits; slice++) {
            for (size_t i = 0; i < this->otherDimension(); i++) {
                fp_t sigma = ldexp(0.75, computeNumFracBits<fp_t>() - bitsPerSlice * slice + 1 - bitsPerSlice) * powersVector[i];
                for (size_t j = 0; j < this->innerProductDimension(); j++) {
                    auto matrixIndex = i * iStride + j * jStride;
                    auto value = (localMatrix[matrixIndex] + sigma);
                    value -= sigma;
                    localMatrix[matrixIndex] -= value;
                    value = value / powersVector[i] * ldexp(1.0, bitsPerSlice * slice + bitsPerSlice - 1);
                    this->memory[matrixIndex + slice * this->matrix.size()] = value;
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
                                  const size_t numDiagonals) {

    std::vector<fp_t > C (A.m * B.n);

    for (size_t diagonal = 0; diagonal <= numDiagonals; diagonal++) {
        int Aindex = diagonal < A.numSplits - 1 ? diagonal : A.numSplits - 1;
        size_t Bindex = diagonal > A.numSplits - 1 ? diagonal - A.numSplits + 1 : 0;
        while (Aindex >= 0 && Bindex <= std::min(diagonal, B.numSplits - 1)) {
            std::vector<accumulator_t> accumulator (A.m * B.n, 0.0);
            computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t>(A, B, accumulator, Aindex, Bindex);
            for (size_t i = 0; i < A.m; i++) {
                for (size_t j = 0; j < B.n; j++) {
                    int totalShift = A.computeSliceBitOffset(static_cast<size_t>(Aindex)) + B.computeSliceBitOffset(Bindex);
                    fp_t scaledSum = std::ldexp(static_cast<fp_t>(accumulator[i + j * A.m]), -totalShift);
                    fp_t scalingFactor = A.powersVector[i] * B.powersVector[j];
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
                                                         const size_t numDiagonals) {
    std::vector<fp_t> C(A.m * B.n);

    for (size_t diagonal = 0; diagonal <= numDiagonals; diagonal++) {
        int Aindex = diagonal < A.numSplits ? static_cast<int>(diagonal) : static_cast<int>(A.numSplits - 1);
        size_t Bindex = diagonal > A.numSplits - 1 ? diagonal - A.numSplits + 1 : 0;

        const int totalShift = A.computeSliceBitOffset(static_cast<size_t>(Aindex)) + B.computeSliceBitOffset(Bindex);

        // Compute and accumulate all products along this anti-diagonal in integer arithmetic.
        std::vector<accumulator_t> accumulator(A.m * B.n, 0);
        while (Aindex >= 0 && Bindex <= std::min(diagonal, B.numSplits - 1)) {
            // assert(A.sliceBitOffset(static_cast<size_t>(Aindex)) + B.sliceBitOffset(Bindex) == totalShift);

            computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t>(A, B, accumulator, Aindex, Bindex);
            Aindex--;
            Bindex++;
        }

        // Scale the accumulated products and accumulate in floating-point arithmetic across diagonals.
        for (size_t i = 0; i < A.m; i++) {
            for (size_t j = 0; j < B.n; j++) {
                fp_t scaledSum = std::ldexp(static_cast<fp_t>(accumulator[i + j * A.m]), -totalShift);
                fp_t scalingFactor = A.powersVector[i] * B.powersVector[j];
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
    const size_t alpha = std::floor((bitsInAccumulator - log2(k)) / 2);
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
        // LCOV_EXCL_START
        default:
            
            std::abort();
        // LCOV_EXCL_STOP
    }

    switch (accType) {
        case accumulationStrategy::floatingPoint:
            return computeProductsWithFloatingPointAccumulation<splitint_t, accumulator_t, fp_t>(splitA, splitB, numDiagonals);
        case accumulationStrategy::integer:
            return computeProductsWithIntegerAccumulation<splitint_t, accumulator_t, fp_t>(splitA, splitB, numDiagonals);
        // LCOV_EXCL_START
        default:
            std::abort();
        // LCOV_EXCL_STOP   
    }
}
template <typename fp_t, typename splitint_t, typename accumulator_t>
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const std::vector<fp_t> &B,
                         const size_t m, const size_t k, const size_t n, const size_t numSplits) {
    return gemmi <fp_t, splitint_t, accumulator_t> (A, B, m, k, n, numSplits, numSplits);
}
