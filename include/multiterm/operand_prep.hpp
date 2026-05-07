#ifndef GEMMI_MULTITERM_OPERAND_PREP_HPP
#define GEMMI_MULTITERM_OPERAND_PREP_HPP

#include "types.hpp"
#include "operand.hpp"
#include "../core/floating_point.hpp"
#include "../core/matrix_view.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace gemmi::mt {

/**
 * @brief Compute the normalisation vector for the Ozaki scheme.
 * 
 * This function computes the normalisation vector (powers of 2) for a given
 * row/column of the matrix. The normalisation vector is used to ensure that
 * all values in the row/column are properly scaled for splitting.
 * 
 * @tparam splitint_t Type used to store the integer slices.
 * @tparam fp_t Floating-point type of the matrix elements.
 * @param operand The operand to normalise.
 */
template <typename splitint_t, typename fp_t>
void computeNormalisationVectors(preparedOperand<splitint_t, fp_t>& operand) {
    // Compute normalisation vector one row/column at a time.
    for (size_t outer = 0; outer < operand.outerDimension(); outer++) {
        operand.powersVector[outer] = 0.0;
        for (size_t inner = 0; inner < operand.innerDimension(); inner++) {
            operand.powersVector[outer] = std::max(operand.powersVector[outer],
                                                std::abs(operand.operand(outer, inner)));
        }
        // Compute the smallest power of 2 that is strictly greater than the
        // maximum value in the row/column.
        // NOTE 1: This is not the technique used in uoi24.
        // NOTE 2: I use exponents instead of powers of 2, as I need the former
        //         to shift correctly.
        operand.scalingExponents[outer] = core::getStoredFloatingPointExponent(operand.powersVector[outer]);
        operand.powersVector[outer] = std::ldexp(1.0, operand.scalingExponents[outer]);
    }
}

/**
 * @brief Compute the block fixed-point representation of a row/column of the matrix.
 * 
 * This function computes the fixed-point representation of a row/column of the
 * matrix, which is used in the splitting algorithms. It extracts the significand
 * and sign of each element in the row/column, and stores them in the provided vectors.
 * 
 * The `fraction` and `sign` vectors must be pre-sized to `operand.innerDimension()`
 * before calling this function.
 *
 * @tparam splitint_t Type used to store the integer slices.
 * @tparam fp_t Floating-point type of the matrix elements.
 * @param fraction Vector to store the fixed-point representation of the elements.
 * @param sign Vector to store the signs of the elements.
 * @param i The index of the row/column for which to compute the fixed-point representation.
 */
template <typename splitint_t, typename fp_t>
void computeFixedPointRepresentationVector(std::vector<typename core::FloatingPointTraits<fp_t>::StorageType> &fraction,
                                           std::vector<bool> &sign, size_t outer,
                                           const preparedOperand<splitint_t, fp_t>& operand) {
    using uint_t = typename core::FloatingPointTraits<fp_t>::StorageType;
    constexpr size_t numSignificandBits = core::FloatingPointTraits<fp_t>::numSignificandBits;
    for (size_t inner = 0; inner < operand.innerDimension(); inner++) {
        fp_t value = operand.operand(outer, inner);
        fraction[inner] = std::bit_cast<uint_t>(value);
        sign[inner] = std::signbit(value);
        uint_t bitmask = (static_cast<uint_t>(1) << (numSignificandBits - 1)) - 1;
        fraction[inner] = fraction[inner] & bitmask;
        // Restore implicit bit for normal numbers.
        // NOTE: NaNs and infs are currently not supported.
        if (std::fpclassify(value) == FP_NORMAL)
            fraction[inner] |= (static_cast<uint_t>(1) << (numSignificandBits - 1));
    }
}

/**
 * @brief Split the matrix using truncation.
 *
 * This is an implementation of Algorithm 4 in:
 *
 *    Ootomo H., Ozaki K., Yokota R. DGEMM on integer matrix multiplication
 *    unit. Int. J. High Performance Comput. App. 2024;38(4):297-313.
 *    DOI: 10.1177/10943420241239588
 *
 * @tparam splitint_t Type used to store the integer slices.
 * @tparam fp_t Floating-point type of the matrix elements.
 * @param operand The operand to split.
 */
template <typename splitint_t, typename fp_t>
void computeSplitsWithTruncation(preparedOperand<splitint_t, fp_t>& operand) {
    using uint_t = typename core::FloatingPointTraits<fp_t>::StorageType;

    // Compute splits one row/column at a time.
    constexpr size_t numSignificandBits = core::FloatingPointTraits<fp_t>::numSignificandBits;
    auto bitsPerSlice = operand.prepConfig.bitsPerSlice;
    std::vector<uint_t> fraction (operand.innerDimension());
    std::vector<bool> sign (operand.innerDimension());
    for (size_t outer = 0; outer < operand.outerDimension(); outer++) {
        // Get binary representation of significands of normalised row/column.
        computeFixedPointRepresentationVector(fraction, sign, outer, operand);

        // Create bitmask.
        const uint_t smallBitmask = (static_cast<uint_t>(1) << bitsPerSlice) - 1;

        // Perform the split.
        for (size_t inner = 0; inner < operand.innerDimension(); inner++) {
            // NOTE: I could have a special path for 0.
            int16_t shiftCounter = numSignificandBits - bitsPerSlice;
            int currentExponent = core::getStoredFloatingPointExponent(operand.operand(outer, inner));
            int16_t exponentDifference = operand.scalingExponents[outer] - currentExponent;
            for (size_t slice = 0; slice < operand.prepConfig.numSplits; slice++) {
                if (exponentDifference > (signed)bitsPerSlice) {
                    exponentDifference -= bitsPerSlice;
                } else {
                    shiftCounter += exponentDifference;
                    exponentDifference = 0;
                    uint_t bitmask = shiftCounter > 0 ?
                        smallBitmask << shiftCounter :
                        smallBitmask >> -shiftCounter;
                    uint_t currentSlice = fraction[inner] & bitmask;
                    uint_t currentSplit = shiftCounter > 0 ?
                        currentSlice >> shiftCounter :
                        currentSlice << -shiftCounter;
                    splitint_t value = static_cast<splitint_t>(currentSplit) * (sign[inner] ? -1 : 1);
                    operand.splitValue(outer, inner, slice) = value;
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
 *
 * @tparam splitint_t Type used to store the integer slices.
 * @tparam fp_t Floating-point type of the matrix elements.
 * @param operand The operand to split.
 */
template <typename splitint_t, typename fp_t>
void computeSplitsWithUnsignedEncoding(preparedOperand<splitint_t, fp_t>& operand) {
    using uint_t = typename core::FloatingPointTraits<fp_t>::StorageType;
    using wideint_t = std::conditional_t<(sizeof(splitint_t) < sizeof(int)), int, std::intmax_t>;

    // Compute splits one row/column at a time.
    constexpr size_t numSignificandBits = core::FloatingPointTraits<fp_t>::numSignificandBits;
    auto bitsPerSlice = operand.prepConfig.bitsPerSlice;
    std::vector<uint_t> fraction (operand.innerDimension());
    std::vector<bool> sign (operand.innerDimension());
    for (size_t outer = 0; outer < operand.outerDimension(); outer++) {
        // Get binary representation of significands of normalised row/column.
        computeFixedPointRepresentationVector(fraction, sign, outer, operand);

        // Create bitmasks.
        // This algorithm uses (bitsPerSlice - 1) bits for the first slice
        // and (bitsPerSlice + 1) bits for subsequent slices. This choice
        // is to keep the algorithm consistent with the bitmasking approach
        // above.
        // NOTE: I am using a first slice with (bitsPerSlice - 1) bits. This
        // is not mentioned in sabb26, but it seems necessary to do this to
        // ensure that the first (signed) slice does not overflow if the full
        // bitwidth of splitint_t is used and the second slice is negative in
        // two's complement.
        const uint_t smallBitmask = (static_cast<uint_t>(1) << (bitsPerSlice - 1)) - 1;
        const uint_t largeBitmask = (static_cast<uint_t>(1) << (bitsPerSlice + 1)) - 1;

        for (size_t inner = 0; inner < operand.innerDimension(); inner++) {

            auto matrixIndex = operand.operandIndex(outer, inner);

            // NOTE: I could have a special path for 0.
            int16_t shiftCounter;
            int currentExponent = core::getStoredFloatingPointExponent(operand.operand(outer, inner));
            int16_t exponentDifference = operand.scalingExponents[outer] - currentExponent;

            splitint_t value = 0;
            uint_t remainder = 0;

            // Slice 0.
            if (exponentDifference > (signed)(bitsPerSlice - 1)) {
                // Value is too small to contribute explicit bits to slice 0.
                value = sign[inner] ? -1 : 0;
                remainder = sign[inner] ?
                    ((static_cast<uint_t>(1) << (sizeof(uint_t) * 8 - 1)) - fraction[inner]) |
                        (static_cast<uint_t>(1) << (sizeof(uint_t) * 8 - 1)) :
                    fraction[inner];
                exponentDifference -= (bitsPerSlice - 1);
                shiftCounter = numSignificandBits - (bitsPerSlice + 1);
            } else {
                // Truncation.
                shiftCounter = numSignificandBits - (bitsPerSlice - 1) + exponentDifference;
                exponentDifference = 0;
                uint_t bitmask = (shiftCounter > 0)
                    ? (smallBitmask << shiftCounter)
                    : (smallBitmask >> -shiftCounter);
                uint_t currentSlice = fraction[inner] & bitmask;
                uint_t currentSplit = (shiftCounter > 0)
                    ? (currentSlice >> shiftCounter)
                    : (currentSlice << -shiftCounter);

                value = (sign[inner] ? -1 : 1) * static_cast<splitint_t>(currentSplit);

                // Rounding.
                uint_t lowMask = (static_cast<uint_t>(1) << shiftCounter) - 1;
                uint_t lowBits = fraction[inner] & lowMask;

                if (lowBits != 0) {
                    value += (sign[inner] ? -1 : 0);
                }
                remainder = sign[inner] ? (static_cast<uint_t>(1) << shiftCounter) - lowBits : lowBits;
                shiftCounter -= (bitsPerSlice + 1);
            }
            operand.memory[matrixIndex] = value;

            // Remaining slices.
            const auto width  = bitsPerSlice + 1;
            const auto cutoff = static_cast<wideint_t>(static_cast<uint_t>(1) << bitsPerSlice); // 2^b
            const auto base   = static_cast<wideint_t>(static_cast<uint_t>(1) << width);        // 2^(b+1)
            const auto digitMax = cutoff - 1;   //  2^b - 1
            const auto digitMin = -cutoff;      // -2^b
            for (size_t slice = 1; slice < operand.prepConfig.numSplits; slice++) {
                if (exponentDifference > (signed)(bitsPerSlice + 1)) {
                    exponentDifference -= (bitsPerSlice + 1);
                    if (sign[inner]) {
                        operand.splitValue(outer, inner, slice - 1) = static_cast<splitint_t>(0);
                        operand.splitValue(outer, inner, slice) = static_cast<splitint_t>(-1);
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

                    if (sign[inner] && (shiftCounter + static_cast<int>(bitsPerSlice) + 1) > static_cast<int>(sizeof(uint_t)) * 8) {
                        int bitsUsed = sizeof(uint_t) * 8 - shiftCounter;
                        int bitsLost = (bitsPerSlice + 1) - bitsUsed;
                        uint_t maskToAdd = ((static_cast<uint_t>(1) << bitsLost) - 1) << bitsUsed;
                        currentSplit |= maskToAdd;
                    }

                    // Update previous slices if currentSplit is negative in splitint_t.
                    // Carry must propoagate upward until slice 0 if necessary.
                    if (currentSplit > static_cast<uint_t>(digitMax)) {
                        int prevSliceIndex = static_cast<int>(slice) - 1;
                        while (true) {
                            auto newValue = static_cast<wideint_t>(operand.splitValue(outer, inner, prevSliceIndex)) + 1;
                            if (newValue <= digitMax) {
                                operand.splitValue(outer, inner, prevSliceIndex) = static_cast<splitint_t>(newValue);
                                break;
                            } else {
                                operand.splitValue(outer, inner, prevSliceIndex) = static_cast<splitint_t>(digitMin);
                                prevSliceIndex--;
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
                    operand.splitValue(outer, inner, slice) = static_cast<splitint_t>(signedDigit);
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
 *
 * @tparam splitint_t Type used to store the integer slices.
 * @tparam fp_t Floating-point type of the matrix elements.
 * @param operand The operand to split.
 */
template <typename splitint_t, typename fp_t>
void computeSplitsWithRoundToNearest(preparedOperand<splitint_t, fp_t>& operand) {
    auto bitsPerSlice = operand.prepConfig.bitsPerSlice;
    auto localMatrix = std::vector<fp_t>(operand.matrix.data, operand.matrix.data + operand.matrix.size());
    for (size_t slice = 0; slice < operand.prepConfig.numSplits; slice++) {
        for (size_t outer = 0; outer < operand.outerDimension(); outer++) {
            // Compute exponent in signed arithmetic to avoid wraparound when
            // bitsPerSlice * (slice + 1) approaches numSignificandBits.
            int exponent = static_cast<int>(core::FloatingPointTraits<fp_t>::numSignificandBits) - static_cast<int>(bitsPerSlice * (slice + 1)) + 1;
            fp_t sigma = std::ldexp(0.75, exponent) * operand.powersVector[outer];
            for (size_t inner = 0; inner < operand.innerDimension(); inner++) {
                auto matrixIndex = operand.operandIndex(outer, inner);
                auto value = (localMatrix[matrixIndex] + sigma);
                value -= sigma;
                localMatrix[matrixIndex] -= value;
                value = value / operand.powersVector[outer] * std::ldexp(1.0, bitsPerSlice * slice + bitsPerSlice - 1);
                operand.splitValue(outer, inner, slice) = value;
            }
        }
    }
}

/**
 * @brief Prepare the operand for the multiterm emulation scheme.
 * @tparam splitint_t Integer type used for splits.
 * @tparam fp_t Floating-point type of the matrix elements.
 */
template <typename splitint_t, typename fp_t>
preparedOperand<splitint_t, fp_t> prepareOperand(core::MatrixView<const fp_t> matrix,
                    const OperandPreparationConfig& prepConfig) {
    preparedOperand<splitint_t, fp_t> operand;
    operand.matrix = matrix;
    operand.prepConfig = prepConfig;
    operand.memory.resize(operand.matrix.size() * prepConfig.numSplits);
    operand.powersVector.resize(operand.prepConfig.dimension == core::normalisationDimension::byRows ?
                           operand.rows() : operand.cols());
    operand.scalingExponents.resize(operand.powersVector.size());

    computeNormalisationVectors(operand);

    switch (prepConfig.splitType) {
        case splittingStrategy::truncation:
            computeSplitsWithTruncation(operand);
            break;
        case splittingStrategy::unsignedEncoding:
            computeSplitsWithUnsignedEncoding(operand);
            break;
        case splittingStrategy::roundToNearest:
            computeSplitsWithRoundToNearest(operand);
            break;
    }

    return operand;
}

} // namespace gemmi::mt

#endif // GEMMI_MULTITERM_OPERAND_PREP_HPP
