#ifndef GEMMI_MULTITERM_GEMMI_HPP
#define GEMMI_MULTITERM_GEMMI_HPP

#include "types.hpp"
#include "operand_prep.hpp"
#include "schedule.hpp"
#include "accumulation.hpp"
#include "../core/matrix_view.hpp"
#include <limits>
#include <stdexcept>
#include <cmath>
#include <string>

namespace gemmi::mt {

/**
 * @brief Validate input parameters for the GEMMI algorithm.
 *
 * This function performs comprehensive validation of all input parameters
 * and configuration settings before running the GEMMI multiplication algorithm.
 * Includes both compile-time type checks and runtime constraint validation.
 *
 * @tparam fp_t          Floating-point element type.
 * @tparam splitint_t    Signed integer type used to store matrix slices.
 * @tparam accumulator_t Signed integer accumulator type.
 * @param  A   View of the left-hand matrix (m x k).
 * @param  B   View of the right-hand matrix (k x n).
 * @param  config Algorithm configuration.
 * @throws std::invalid_argument if any runtime constraint is violated.
 */
template <typename fp_t, typename splitint_t, typename accumulator_t>
void validateParameters(core::MatrixView<const fp_t> A,
                        core::MatrixView<const fp_t> B,
                        const config& config) {

    // Compile-time type checks.
    static_assert(std::is_floating_point_v<fp_t>,
                  "fp_t must be a floating-point type");
    static_assert(std::is_integral_v<splitint_t> && std::is_signed_v<splitint_t>,
                  "splitint_t must be a signed integer type");
    static_assert(std::is_integral_v<accumulator_t> && std::is_signed_v<accumulator_t>,
                  "accumulator_t must be a signed integer type");

    // Matrix checks.
    auto validateMatrix = [](const auto& M, std::string_view name) {
        if (M.data == nullptr)
            throw std::invalid_argument(std::string(name) + " has a null data pointer");
        if (M.empty())
            throw std::invalid_argument(std::string(name) + " is empty (rows or cols is 0)");
    };
    validateMatrix(A, "Matrix A");
    validateMatrix(B, "Matrix B");

    if (A.cols != B.rows)
        throw std::invalid_argument(
            "Dimension mismatch: A.cols (" + std::to_string(A.cols) +
            ") != B.rows (" + std::to_string(B.rows) + ")");

    // Split counts.
    if (config.numSplitsA == 0)
        throw std::invalid_argument("numSplitsA must be >= 1");
    if (config.numSplitsB == 0)
        throw std::invalid_argument("numSplitsB must be >= 1");

    // Custom mask size.
    if (std::holds_alternative<std::vector<bool>>(config.multSpecification)) {
        const auto& mask = std::get<std::vector<bool>>(config.multSpecification);
        const size_t expected = config.numSplitsA * config.numSplitsB;
        if (mask.size() != expected)
            throw std::invalid_argument(
                "Custom mask size (" + std::to_string(mask.size()) +
                ") does not match numSplitsA * numSplitsB (" +
                std::to_string(expected) + ")");
    }

    // Type width constraints.
    constexpr size_t bitsInAccumulator = std::numeric_limits<accumulator_t>::digits;
    constexpr size_t bitsPerInteger    = std::numeric_limits<splitint_t>::digits;
    if (bitsPerInteger > bitsInAccumulator / 2)
        throw std::invalid_argument(
            "splitint_t (" + std::to_string(bitsPerInteger) + " bits) is too wide "
            "for accumulator_t (" + std::to_string(bitsInAccumulator) + " bits): "
            "require bitsPerInteger <= bitsInAccumulator / 2");
}

/**
 * @brief Compute the number of bits assigned to each split slice.
 *
 * @tparam splitint_t    Signed integer type used to store matrix slices.
 * @tparam accumulator_t Signed integer accumulator type.
 * @param innerDimension Inner dimension of the matrix product (`k`).
 * @return size_t Number of bits assigned to each split slice.
 * @throws std::invalid_argument if the computed value is zero.
 */
template <typename splitint_t, typename accumulator_t>
size_t computeBitsPerSlice(size_t innerDimension) {
    constexpr size_t bitsInAccumulator = std::numeric_limits<accumulator_t>::digits;
    constexpr size_t bitsPerInteger    = std::numeric_limits<splitint_t>::digits;

    const double log2k = (innerDimension > 1) ? std::log2(static_cast<double>(innerDimension)) : 0.0;
    const auto alpha = static_cast<size_t>(
        std::max(0.0, std::floor((static_cast<double>(bitsInAccumulator) - log2k) / 2.0)));
    const size_t bitsPerSlice = std::min(bitsPerInteger, alpha);
    if (bitsPerSlice == 0) {
        throw std::invalid_argument(
            "Computed bitsPerSlice is 0: inner dimension k=" + std::to_string(innerDimension) +
            " is too large for accumulator_t (" +
            std::to_string(bitsInAccumulator) + " bits)");
    }

    return bitsPerSlice;
}

/**
 * @brief Compute the matrix product C = C + A * B using the multiterm emulation scheme.
 * 
 * @tparam fp_t Floating-point type of the matrix elements.
 * @tparam splitint_t Integer type used for splits.
 * @tparam accumulator_t Accumulator type.
 * @param A Matrix A.
 * @param layoutA Layout of matrix A.
 * @param B Matrix B.
 * @param layoutB Layout of matrix B.
 * @param m Number of rows in A.
 * @param k Number of columns in A and rows in B.
 * @param n Number of columns in B.
 * @param layoutC Layout of the output matrix C.
 * @param config Configuration parameters.
 * @return Resulting matrix product.
 */
template <typename fp_t, typename splitint_t, typename accumulator_t>
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const core::matrixLayout layoutA,
                         const std::vector<fp_t> &B, const core::matrixLayout layoutB,
                         const size_t m, const size_t k, const size_t n,
                         const core::matrixLayout layoutC,
                         const config &config) {

    // Build matrix views.
    auto viewA = core::MatrixView<const fp_t>(A, m, k, layoutA);
    auto viewB = core::MatrixView<const fp_t>(B, k, n, layoutB);

    // Validate inputs and compute execution parameters.
    validateParameters<fp_t, splitint_t, accumulator_t>(viewA, viewB, config);
    const size_t bitsPerSlice = computeBitsPerSlice<splitint_t, accumulator_t>(k);

    // Slice operands.
    auto splitA = prepareOperand<splitint_t, fp_t>(viewA,
        OperandPreparationConfig(config.splitType, config.numSplitsA, bitsPerSlice, core::normalisationDimension::byRows));
    auto splitB = prepareOperand<splitint_t, fp_t>(viewB,
        OperandPreparationConfig(config.splitType, config.numSplitsB, bitsPerSlice, core::normalisationDimension::byCols));

    // Build multiplication schedule.
    auto schedule = makeSchedule(config);

    // Execute multiplication based on reduction type.
    if (config.redType == reductionStrategy::floatingPoint) {
        return computeProductsWithFloatingPointAccumulation<splitint_t, accumulator_t, fp_t>(
            splitA, splitB, schedule, layoutC);
    } else {
        return computeProductsWithIntegerAccumulation<splitint_t, accumulator_t, fp_t>(
            splitA, splitB, schedule, layoutC);
    }
}

/**
 * @brief Compute the matrix product with default configuration.
 *
 * Convenience overload with default values:
 * - round-to-nearest splitting,
 * - reduced multiplication schedule, and
 * - floating-point accumulation.
 * 
 * @tparam fp_t Floating-point type of the matrix elements.
 * @tparam splitint_t Integer type used for splits.
 * @tparam accumulator_t Accumulator type.
 * @param A Matrix A.
 * @param layoutA Layout of matrix A.
 * @param B Matrix B.
 * @param layoutB Layout of matrix B.
 * @param m Number of rows in A.
 * @param k Number of columns in A and rows in B.
 * @param n Number of columns in B.
 * @param layoutC Layout of the output matrix C.
 * @param numSplits Number of splits for both matrices.
 * @return Resulting matrix product.
 */
template <typename fp_t, typename splitint_t, typename accumulator_t>
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const core::matrixLayout layoutA,
                         const std::vector<fp_t> &B, const core::matrixLayout layoutB,
                         const size_t m, const size_t k, const size_t n,
                         const core::matrixLayout layoutC,
                         const size_t numSplits) {
    return gemmi <fp_t, splitint_t, accumulator_t> (A, layoutA, B, layoutB, m, k, n, layoutC,
                                                    config{numSplits, numSplits,
                                                    splittingStrategy::roundToNearest,
                                                    multiplicationStrategy::reduced,
                                                    reductionStrategy::floatingPoint});
}

} // namespace gemmi::mt

#endif // GEMMI_MULTITERM_GEMMI_HPP
