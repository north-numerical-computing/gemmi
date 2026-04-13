#ifndef GEMMI_MULTITERM_ACCUMULATION_HPP
#define GEMMI_MULTITERM_ACCUMULATION_HPP

#include "operand.hpp"
#include "schedule.hpp"
#include "../core/matrix_view.hpp"
#include <vector>
#include <cmath>

namespace gemmi::mt {

/**
 * @brief Compute the exact integer GEMM (General Matrix-Matrix Multiplication).
 * 
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
void computeExactIntegerGEMM(const preparedOperand<splitint_t, fp_t> &A,
                             const preparedOperand<splitint_t, fp_t> &B,
                             std::vector<accumulator_t> &C,
                             const core::matrixLayout layoutC,
                             size_t iBlock, size_t jBlock) {
    for (size_t row = 0; row < A.rows(); row++) {
        for (size_t col = 0; col < B.cols(); col++) {
            auto index = (layoutC == core::matrixLayout::columnMajor) ? (row + col * A.rows()) : (col + row * B.cols());
            for (size_t ell = 0; ell < A.innerDimension(); ell++) {
                C[index] += A.splitValue(row, ell, iBlock) * B.splitValue(col, ell, jBlock);
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
 * @param schedule Multiplication schedule specifying which slice products to compute.
 * @param layoutC Layout of the output matrix C.
 * @return Resulting matrix C.
 */
template <typename splitint_t, typename accumulator_t, typename fp_t>
std::vector<fp_t> computeProductsWithFloatingPointAccumulation(const preparedOperand<splitint_t, fp_t> &A,
                                                               const preparedOperand<splitint_t, fp_t> &B,
                                                               const multiplicationSchedule &schedule,
                                                               const core::matrixLayout layoutC) {
    auto numSplitsA = A.prepConfig.numSplits;
    auto numSplitsB = B.prepConfig.numSplits;
    std::vector<fp_t> C (A.rows() * B.cols(), 0.0);
    for (size_t diagonal = 0; diagonal <= numSplitsA + numSplitsB - 1; diagonal++) {
        int Aindex = diagonal < numSplitsA ? static_cast<int>(diagonal) : static_cast<int>(numSplitsA - 1);
        size_t Bindex = diagonal > numSplitsA - 1 ? diagonal - numSplitsA + 1 : 0;
        while (Aindex >= 0 && Bindex <= std::min(diagonal, numSplitsB - 1)) {
            if (schedule(Aindex, Bindex)) {
                std::vector<accumulator_t> accumulator (A.rows() * B.cols(), 0.0);
                int totalShift = A.computeSliceBitOffset(static_cast<size_t>(Aindex)) + B.computeSliceBitOffset(Bindex);
                computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t>(A, B, accumulator, layoutC, Aindex, Bindex);
                for (size_t row = 0; row < A.rows(); row++) {
                    for (size_t col = 0; col < B.cols(); col++) {
                        auto index = (layoutC == core::matrixLayout::columnMajor) ? (row + col * A.rows()) : (col + row * B.cols());
                        fp_t scaledSum = std::ldexp(static_cast<fp_t>(accumulator[index]), -totalShift);
                        fp_t scalingFactor = A.powersVector[row] * B.powersVector[col];
                        C[index] += scaledSum * scalingFactor;
                    }
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
 * @param schedule Multiplication schedule specifying which slice products to compute.
 * @param layoutC Layout of the output matrix C.
 * @return Resulting matrix C.
 */
template <typename splitint_t, typename accumulator_t, typename fp_t>
std::vector<fp_t> computeProductsWithIntegerAccumulation(const preparedOperand<splitint_t, fp_t> &A,
                                                         const preparedOperand<splitint_t, fp_t> &B,
                                                         const multiplicationSchedule &schedule,
                                                         const core::matrixLayout layoutC) {
    auto numSplitsA = A.prepConfig.numSplits;
    auto numSplitsB = B.prepConfig.numSplits;
    
    std::vector<fp_t> C (A.rows() * B.cols(), 0.0);
    for (size_t diagonal = 0; diagonal <= numSplitsA + numSplitsB - 1; diagonal++) {
        int Aindex = diagonal < numSplitsA ? static_cast<int>(diagonal) : static_cast<int>(numSplitsA - 1);
        size_t Bindex = diagonal > numSplitsA - 1 ? diagonal - numSplitsA + 1 : 0;

        const int totalShift = A.computeSliceBitOffset(static_cast<size_t>(Aindex)) + B.computeSliceBitOffset(Bindex);

        // Compute and accumulate all products along this anti-diagonal in integer arithmetic.
        std::vector<accumulator_t> accumulator(A.rows() * B.cols(), 0);
        while (Aindex >= 0 && Bindex <= std::min(diagonal, numSplitsB - 1)) {
            if (schedule(Aindex, Bindex))
                computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t>(A, B, accumulator, layoutC, Aindex, Bindex);
            Aindex--;
            Bindex++;
        }

        // Scale the accumulated products and accumulate in floating-point arithmetic across diagonals.
        for (size_t row = 0; row < A.rows(); row++) {
            for (size_t col = 0; col < B.cols(); col++) {
                auto index = (layoutC == core::matrixLayout::columnMajor) ? (row + col * A.rows()) : (col + row * B.cols());
                fp_t scaledSum = std::ldexp(static_cast<fp_t>(accumulator[index]), -totalShift);
                fp_t scalingFactor = A.powersVector[row] * B.powersVector[col];
                C[index] += scaledSum * scalingFactor;
            }
        }
    }

    return C;
}

} // namespace gemmi::mt

#endif // GEMMI_MULTITERM_ACCUMULATION_HPP
