#ifndef GEMMI_MULTITERM_OPERAND_HPP
#define GEMMI_MULTITERM_OPERAND_HPP

#include "types.hpp"
#include "../core/floating_point.hpp"
#include "../core/matrix_view.hpp"
#include <vector>
#include <stdexcept>

namespace gemmi::mt {

/**
 * @brief Class to store the matrix slices for the Ozaki scheme.
 * 
 * @tparam splitint_t Type used to store the integer slices.
 * @tparam fp_t Floating-point type of the matrix elements.
 */
template <typename splitint_t, typename fp_t>
struct preparedOperand {
    core::MatrixView<const fp_t> matrix;       ///< View of original matrix.
    OperandPreparationConfig prepConfig; ///< Configuration for operand preparation.

    std::vector<splitint_t> memory;      ///< Memory to store the split slices.
    std::vector<fp_t> powersVector;      ///< Normalisation vector.
    std::vector<int> scalingExponents;   ///< Scaling exponents.

    /**
     * @brief Return the number of rows in the original matrix.
     * @return Number of rows.
     */
    size_t rows() const {
        return matrix.rows;
    }

    /**
     * @brief Return the number of columns in the original matrix.
     * @return Number of columns.
     */
    size_t cols() const {
        return matrix.cols;
    }

    /**
     * @brief Return the dimension along which the inner product is calculated.
     * @return Inner product dimension.
     */
    size_t innerDimension() const {
        return (prepConfig.dimension == core::normalisationDimension::byRows) ?
            matrix.cols : matrix.rows;
    }

    /**
     * @brief Return the dimension not used in the inner product.
     * @return Dimension not used in the inner product.
     */
    size_t outerDimension() const {
        return (prepConfig.dimension == core::normalisationDimension::byRows) ?
            matrix.rows : matrix.cols;
    }

    /**
     * @brief Compute the index for a given operand in the matrix.
     * @param outer The outer index.
     * @param inner The inner index.
     * @return The index of the operand in the matrix.
     */
    size_t operandIndex(size_t outer, size_t inner) const {
        return (prepConfig.dimension == core::normalisationDimension::byRows) ?
            matrix.index(outer, inner) :
            matrix.index(inner, outer);
    }

    /**
     * @brief Compute the index for a given slice of an element in the matrix.
     * @param outer The outer index.
     * @param inner The inner index.
     * @param slice The slice index.
     * @return The index of the slice in the memory.
     */
    size_t splitIndex(size_t outer, size_t inner, size_t slice) const {
        return operandIndex(outer, inner) + slice * matrix.size();
    }

    /**
     * @brief Accessor methods for matrix elements and their split values.
     *
     * - `operand(outer, inner)`: Read-only access to original floating-point matrix element.
     * - `splitValue(outer, inner, slice)`: Mutable and const versions to access split values at a specific slice.
     */

    /// @brief Access the original operand at (outer, inner).
    const fp_t& operand(size_t outer, size_t inner) const {
        return matrix.linear(operandIndex(outer, inner));
    }

    /// @brief Mutable access to split value at (outer, inner, slice).
    splitint_t& splitValue(size_t outer, size_t inner, size_t slice) {
        return memory[splitIndex(outer, inner, slice)];
    }

    /// @brief Const access to split value at (outer, inner, slice).
    const splitint_t& splitValue(size_t outer, size_t inner, size_t slice) const {
        return memory[splitIndex(outer, inner, slice)];
    }

    /**
     * @brief Compute the bit offset for a given slice.
     * @param slice The slice for which to compute the bit offset.
     * @return The bit offset for the specified slice.
     */
    int computeSliceBitOffset(size_t slice) const {
        auto bitsPerSlice = prepConfig.bitsPerSlice;
        switch (prepConfig.splitType) {
            case splittingStrategy::truncation:
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
                throw std::logic_error("Unhandled splittingStrategy");
            // LCOV_EXCL_STOP
        }
    }
};

} // namespace gemmi::mt

#endif // GEMMI_MULTITERM_OPERAND_HPP
