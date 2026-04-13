#ifndef GEMMI_HPP
#define GEMMI_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

/**
 * @file gemmi.hpp
 * @brief Floating-point matrix multiplication using integer emulation.
 */

/***************************************
 * Floating-point traits and functions *
 ***************************************/

namespace fp {

/**
 * @brief Traits describing IEEE-754 layout properties for a floating-point type.
 *
 * This trait centralizes all compile-time metadata about floating-point types.
 * - StorageType: unsigned integer type for to bit-cast values.
 * - numExponentBits: number of bits in the exponent field.
 * - numSignificandBits: number of significand bits, including the implicit bit.
 *
 * @tparam fp_t Supported floating-point type.
 */
template <typename fp_t>
struct FloatingPointTraits;

/**
 * @brief IEEE-754 traits for single-precision floating point.
 */
template <>
struct FloatingPointTraits<float> {
    using StorageType = uint32_t;
    static constexpr size_t numExponentBits = 8;
    static constexpr size_t numSignificandBits = 24;
};

/**
 * @brief IEEE-754 traits for double-precision floating point.
 */
template <>
struct FloatingPointTraits<double> {
    using StorageType = uint64_t;
    static constexpr size_t numExponentBits = 11;
    static constexpr size_t numSignificandBits = 53;
};

/**
 * @brief Get the exponent of a floating-point value.
 * @tparam fp_t Floating-point type (e.g., float, double).
 * @param value The floating-point value.
 * @return The value stored in the exponent field of the value, as an integer.
 */
template <typename fp_t>
int getStoredFloatingPointExponent(fp_t value) {
    return (value == 0.0) ?
        0 :
        std::max(std::numeric_limits<fp_t>::min_exponent, std::ilogb(std::abs(value)) + 1);
}

} // namespace fp

/******************************
 * Global configuration enums *
 ******************************/

/**
 * @brief Enum to specify the dimension used for normalization.
 */
enum class normalisationDimension {
    byRows, ///< Normalise by rows (matrix on the left of the product).
    byCols  ///< Normalise by columns (matrix on the right of the product).
};

/***************
 * Matrix view *
 ***************/

namespace matrix {

/**
 * @brief Enum to specify the layout of the matrix in memory.
 */
enum class matrixLayout {
    rowMajor,    ///< Matrix stored in row-major order.
    columnMajor  ///< Matrix stored in column-major order.
};

/**
 * @brief Lightweight view of a dense matrix.
 *
 * It does not own the underlying memory but provides read and write access to it.
 *
 * @tparam value_t Element type.
 */
template <typename value_t>
struct MatrixView {
    value_t* data;        ///< Pointer to the matrix data.
    size_t rows;          ///< Number of rows in the matrix.
    size_t cols;          ///< Number of columns in the matrix.
    matrixLayout layout;  ///< Layout of the matrix in memory.

    /**
     * @brief Default constructor.
     */
    MatrixView() :
        data(nullptr), rows(0), cols(0), layout(matrixLayout::rowMajor) {}

    /**
     * @brief Construct a matrix view from raw parts.
     *
     * @param data Pointer to the first matrix element.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param layout Memory layout.
     */
    MatrixView(value_t* data, size_t rows, size_t cols, matrixLayout layout) :
        data(data), rows(rows), cols(cols), layout(layout) {}

    /**
     * @brief Construct a mutable matrix view from a vector.
     *
     * @param vec Vector containing the matrix data.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param layout Memory layout.
     */
    MatrixView(std::vector<value_t>& vec, size_t rows, size_t cols, matrixLayout layout) :
        data(vec.data()), rows(rows), cols(cols), layout(layout) {}

    /**
     * @brief Construct a read-only matrix view from a const vector.
     *
     * Only participates in overload resolution when @c value_t is const-qualified.
     *
     * @param vec Const vector containing the matrix data.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param layout Memory layout.
     */
    template <typename = std::enable_if_t<std::is_const_v<value_t>>>
    MatrixView(const std::vector<std::remove_const_t<value_t>>& vec, size_t rows, size_t cols, matrixLayout layout) :
        data(vec.data()), rows(rows), cols(cols), layout(layout) {}

    template <typename other_t,
              typename = std::enable_if_t<
                  std::is_const_v<value_t> &&
                  std::is_same_v<std::remove_const_t<value_t>, other_t>>>
    MatrixView(const MatrixView<other_t>& other) :
        data(other.data), rows(other.rows), cols(other.cols), layout(other.layout) {}

    /**
     * @brief Return the number of stored elements.
     *
     * @return Number of rows by number of columns.
     */
    size_t size() const {
        return rows * cols;
    }

    /**
     * @brief Return true if the view is empty.
     *
     * @return `true` if `rows == 0` or `cols == 0`
     */
    bool empty() const {
        return rows == 0 || cols == 0;
    }

    /**
     * @brief Compute the linear index of element (i, j).
     *
     * @param i Row index.
     * @param j Column index.
     * @return Linear index into the underlying data array.
     */
    size_t index(size_t i, size_t j) const {
        return (layout == matrixLayout::rowMajor)
            ? (i * cols + j)
            : (j * rows + i);
    }

    /**
     * @brief Access element (i, j).
     *
     * The `const` qualifier applies to the view metadata (pointer, dimensions,
     * and layout), and not to the pointed-to data. Mutability of the returned
     * reference is controlled by `value_t`. `MatrixView<const T>` should be
     * used for a read-only view.
     *
     * @param i Row index.
     * @param j Column index.
     * @return Reference to the element.
     */
    value_t& operator()(size_t i, size_t j) const {
        return data[index(i, j)];
    }

    /**
     * @brief Access element (i, j) explicitly.
     *
     * This method throws an exception if the access is out of bounds.
     * It is a safe alternative to operator(), which does not perform
     * bounds checking.
     *
     * @param i Row index.
     * @param j Column index.
     * @return Reference to the element.
     */
    value_t& at(size_t i, size_t j) const {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("MatrixView index out of range");
        }
        return data[index(i, j)];
    }

    /**
     * @brief Access element by linear index.
     * @param idx Linear index.
     * @return Reference to the element.
     */
    value_t& linear(size_t idx) const {
        return data[idx];
    }
};

} // namespace matrix

/***********************
 * Multiterm emulation *
 ***********************/

namespace multiterm {

/**
 * @brief Enum to specify the splitting strategy to use.
 */
enum class splittingStrategy {
    truncation,       ///< Split using truncation.
    unsignedEncoding, ///< Split using unsigned slice encoding.
    roundToNearest    ///< Split using round-to-nearest.
};

/**
 * @brief Enum to specify the multiplication strategy to use.
 */
enum class multiplicationStrategy {
    full,       ///< Compute all products.
    reduced     ///< Only compute products above the main anti-diagonal.
};

/**
 * @brief Enum to specify the accumulation strategy to use.
 */
enum class reductionStrategy {
    floatingPoint, ///< Accumulate products in floating-point arithmetic.
    integer        ///< Accumulate products in integer arithmetic.
};

using multiplicationSpecification = std::variant<multiplicationStrategy, std::vector<bool>>;

/**
 * @brief Configuration object for the multiterm emulation scheme.
 *
 * The `config` struct specifies *all user‑selectable settings* for the
 * multiterm algorithm.
 *
 * The user may specify the multiplication schedule as either:
 *
 *   **(1) a predefined strategy **, using a `multiplicationStrategy` value, or
 *
 *   **(2) a custom mask**, using an explicit std::vector<bool> with
 *       `numSplitsA * numSplitsB` elements.
 *
 * If the custom mask is used, the vector is interpreted as a matrix stored in
 * row-major order, where each column represents a slice of `A` and each row
 * represents a slice of `B`. Products are computed only for those slice pairs
 * (i,j) where the mask is true.
 *
 *
 * ### Examples
 *
 * To use the reduced slice-product rule with round-to-nearest splitting
 * and floating-point accumulation, one can use:
 *
 * \code
 * multiterm::config config;
 * config.numSplitsA        = 8;
 * config.numSplitsB        = 8;
 * config.splitType         = multiterm::splitStrategy::roundToNearest;
 * config.redType           = multiterm::reductionStrategy::floatingPoint;
 * config.multSpecification = multiterm::multiplicationStrategy::reduced;
 *
 * auto C = gemmi<double, std::int8_t, std::int32_t>(
 *     A, layoutA, B, layoutB, m, k, n, layoutC, config
 * );
 * \endcode
 *
 * To use a multiplication where the first two slices of A and
 * B are multiplied by all slices of the other matrix, one can use the
 * custom mask:
 *
 * \code
 * config.multSpecification = std::vector<bool>{
 *     1,1,1,1,
 *     1,1,1,1,
 *     1,1,0,0,
 *     1,1,0,0
 * };
 * \endcode
 *
 * Only products at positions marked `true' are computed.
 *
 */
struct config {
    size_t numSplitsA;                             ///< Number of slices for matrix A.
    size_t numSplitsB;                             ///< Number of slices for matrix B.
    splittingStrategy splitType;                   ///< Slice computation strategy.
    multiplicationSpecification multSpecification; ///< Multiplication specification.
    reductionStrategy redType;                     ///< Reduction strategy.
};

/**
 * @brief Validate inputs for the multiterm scheme.
 *
 * Check that the two matrix views and the algorithm configuration are mutually
 * consistent and that the template types satisfy the precision requirements of
 * the accumulation scheme. The function performs both compile-time
 * (static_assert) and runtime checks:
 *  - `fp_t` is a floating-point type (compile-time);
 *  - `splitint_t` is a signed integer type (compile-time);
 *  - `accumulator_t` is a signed integer type (compile-time);
 *  - neither matrix view has a null data pointer;
 *  - neither matrix view is empty;
 *  - the matrices are conformable for multiplication(`A.cols() == B.rows()`);
 *  - `numSplitsA` and `numSplitsB` are both strictly positive;
 *  - the custom mask size (if used) equals `numSplitsA * numSplitsB`;
 *  - `splitint_t` is not too wide for `accumulator_t`.
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
void validateParameters(matrix::MatrixView<const fp_t> A,
                        matrix::MatrixView<const fp_t> B,
                        const config& config) {

    // Compile-time type checks.
    static_assert(std::is_floating_point_v<fp_t>,
                  "fp_t must be a floating-point type");
    static_assert(std::is_integral_v<splitint_t> && std::is_signed_v<splitint_t>,
                  "splitint_t must be a signed integer type");
    static_assert(std::is_integral_v<accumulator_t> && std::is_signed_v<accumulator_t>,
                  "accumulator_t must be a signed integer type");

    // Matrix checks.
    if (A.data == nullptr)
        throw std::invalid_argument("Matrix A has a null data pointer");
    if (B.data == nullptr)
        throw std::invalid_argument("Matrix B has a null data pointer");
    if (A.empty())
        throw std::invalid_argument("Matrix A is empty (rows or cols is 0)");
    if (B.empty())
        throw std::invalid_argument("Matrix B is empty (rows or cols is 0)");

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
 * @tparam fp_t          Floating-point element type.
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

/***********************
 * Operand preparation *
 ***********************/

/**
 * @brief Configuration for operand preparation.
 */
struct OperandPreparationConfig {
    splittingStrategy splitType;      ///< Splitting strategy to use.
    size_t numSplits;                 ///< Number of splits to use.
    size_t bitsPerSlice;              ///< Number of bits per slice.
    normalisationDimension dimension; ///< Dimension along which to normalize.
};

/**
 * @brief Class to store the matrix slices for the Ozaki scheme.
 * 
 * @tparam splitint_t Type used to store the integer slices.
 * @tparam fp_t Floating-point type of the matrix elements.
 */
template <typename splitint_t, typename fp_t>
struct preparedOperand {
    matrix::MatrixView<const fp_t> matrix;       ///< View of original matrix.
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
        return (prepConfig.dimension == normalisationDimension::byRows) ?
            matrix.cols : matrix.rows;
    }

    /**
     * @brief Return the dimension not used in the inner product.
     * @return Dimension not used in the inner product.
     */
    size_t outerDimension() const {
        return (prepConfig.dimension == normalisationDimension::byRows) ?
            matrix.rows : matrix.cols;
    }

    /**
     * @brief Compute the index for a given operand in the matrix.
     * @param outer The outer index.
     * @param inner The inner index.
     * @return The index of the operand in the matrix.
     */
    size_t operandIndex(size_t outer, size_t inner) const {
        return (prepConfig.dimension == normalisationDimension::byRows) ?
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
     * @brief Access the operand at the given outer and inner indices.
     * @param outer The outer index.
     * @param inner The inner index.
     * @return Reference to the operand at the specified indices.
     */
    const fp_t& operand(size_t outer, size_t inner) const {
        return matrix.linear(operandIndex(outer, inner));
    }

    /**
     * @brief Access the split value at the given indices.
     * @param outer The outer index.
     * @param inner The inner index.
     * @param slice The slice index.
     * @return Mutable reference to the split value at the specified indices.
     */
    splitint_t& splitValue(size_t outer, size_t inner, size_t slice) {
        return memory[splitIndex(outer, inner, slice)];
    }

    /**
     * @brief Access the split value at the given indices (const version).
     * @param outer The outer index.
     * @param inner The inner index.
     * @param slice The slice index.
     * @return Read-only reference to the split value at the specified indices.
     */
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

/**
 * @brief Compute normalization vectors for the matrix.
 */
template <typename splitint_t, typename fp_t>
void computeNormalisationVectors(preparedOperand<splitint_t, fp_t>& operand) {
    // Compute normalisation vector.
    for (size_t outer = 0; outer < operand.outerDimension(); outer++) {
        operand.powersVector[outer] = 0.0;
        for (size_t j = 0; j < operand.innerDimension(); j++) {
            operand.powersVector[outer] = std::max(operand.powersVector[outer],
                                                std::abs(operand.operand(outer, j)));
        }
        // Compute the smallest power of 2 that is strictly greater than the
        // maximum value in the row/column.
        // NOTE 1: This is not the technique used in uoi24.
        // NOTE 2: I use exponents instead of powers of 2, as I need the former
        //         to shift correctly.
        operand.scalingExponents[outer] = fp::getStoredFloatingPointExponent(operand.powersVector[outer]);
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
void computeFixedPointRepresentationVector(std::vector<typename fp::FloatingPointTraits<fp_t>::StorageType> &fraction,
                                           std::vector<bool> &sign, size_t outer,
                                           const preparedOperand<splitint_t, fp_t>& operand) {
    using uint_t = typename fp::FloatingPointTraits<fp_t>::StorageType;
    constexpr size_t numSignificandBits = fp::FloatingPointTraits<fp_t>::numSignificandBits;
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
    using uint_t = typename fp::FloatingPointTraits<fp_t>::StorageType;

    // Compute splits one row/column at a time.
    constexpr size_t numSignificandBits = fp::FloatingPointTraits<fp_t>::numSignificandBits;
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
            int currentExponent = fp::getStoredFloatingPointExponent(operand.operand(outer, inner));
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
    using uint_t = typename fp::FloatingPointTraits<fp_t>::StorageType;
    using wideint_t = std::conditional_t<(sizeof(splitint_t) < sizeof(int)), int, std::intmax_t>;

    // Compute splits one row/column at a time.
    constexpr size_t numSignificandBits = fp::FloatingPointTraits<fp_t>::numSignificandBits;
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
            int currentExponent = fp::getStoredFloatingPointExponent(operand.operand(outer, inner));
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
            int exponent = static_cast<int>(fp::FloatingPointTraits<fp_t>::numSignificandBits) - static_cast<int>(bitsPerSlice * (slice + 1)) + 1;
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
preparedOperand<splitint_t, fp_t> prepareOperand(matrix::MatrixView<const fp_t> matrix,
                    const OperandPreparationConfig& prepConfig) {
    preparedOperand<splitint_t, fp_t> operand;
    operand.matrix = matrix;
    operand.prepConfig = prepConfig;
    operand.memory.resize(operand.matrix.size() * prepConfig.numSplits);
    operand.powersVector.resize(operand.prepConfig.dimension == normalisationDimension::byRows ?
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

/*************************************
 * Perform the matrix multiplication *
 *************************************/

/**
 * @brief Class to store the multiplication schedule for the Ozaki scheme.
 * 
 * The multiplication schedule specifies which products of slices of A and B
 * should be computed. It can be constructed from a config object.
 */
 struct multiplicationSchedule {
    size_t numSplitsA;           // number of slices for A
    size_t numSplitsB;           // number of slices for B

    std::vector<bool> mask;

    /**
     * @brief Constructor for the multiplication schedule.
     * @param numSplitsA Number of splits for matrix A.
     * @param numSplitsB Number of splits for matrix B.
     */
    multiplicationSchedule(size_t numSplitsA, size_t numSplitsB) :
        numSplitsA(numSplitsA), numSplitsB(numSplitsB), mask(numSplitsA * numSplitsB, false) {}

    inline std::vector<bool>::reference operator()(size_t i, size_t j) {
        return mask[i * numSplitsB + j];
    }
    inline bool operator()(size_t i, size_t j) const {
        return mask[i * numSplitsB + j];
    }
};

/**
 * @brief Create a multiplication schedule from a config object.
 * 
 * This function creates a multiplication schedule based on the multiplication
 * specification in the config object.
 * 
 * @param config The config object containing the multiplication specification.
 * @return multiplicationSchedule The resulting multiplication schedule.
 * @throws std::invalid_argument if the multiplication specification is invalid.
 */
inline multiplicationSchedule makeSchedule(const config& config) {
    multiplicationSchedule schedule(config.numSplitsA, config.numSplitsB);

    std::visit([&](auto&& spec) {
        using T = std::decay_t<decltype(spec)>;

        if constexpr (std::is_same_v<T, multiplicationStrategy>) {
            if (spec == multiplicationStrategy::full) {
                std::fill(schedule.mask.begin(), schedule.mask.end(), true);
            } else {
                size_t limit = std::max(config.numSplitsA, config.numSplitsB) - 1;
                for (size_t i = 0; i < config.numSplitsA; ++i)
                    for (size_t j = 0; j < config.numSplitsB; ++j)
                        schedule(i, j) = i + j <= limit;
            }
        } else if constexpr (std::is_same_v<T, std::vector<bool>>) {

            const auto& mask = spec;

            if (mask.size() != config.numSplitsA * config.numSplitsB)
                throw std::invalid_argument("Mask size mismatch.");

            schedule.mask = mask; // row‑major copy
        }

    }, config.multSpecification);

    return schedule;
}

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
                             const matrix::matrixLayout layoutC,
                             size_t iBlock, size_t jBlock) {
    for (size_t row = 0; row < A.rows(); row++) {
        for (size_t col = 0; col < B.cols(); col++) {
            auto index = (layoutC == matrix::matrixLayout::columnMajor) ? (row + col * A.rows()) : (col + row * B.cols());
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
                                                               const matrix::matrixLayout layoutC) {
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
                        auto index = (layoutC == matrix::matrixLayout::columnMajor) ? (row + col * A.rows()) : (col + row * B.cols());
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
                                                         const matrix::matrixLayout layoutC) {
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
                auto index = (layoutC == matrix::matrixLayout::columnMajor) ? (row + col * A.rows()) : (col + row * B.cols());
                fp_t scaledSum = std::ldexp(static_cast<fp_t>(accumulator[index]), -totalShift);
                fp_t scalingFactor = A.powersVector[row] * B.powersVector[col];
                C[index] += scaledSum * scalingFactor;
            }
        }
    }

    return C;
}

} // namespace multiterm

/**
 * @brief Compute the matrix product C = C + A * B.
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
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const matrix::matrixLayout layoutA,
                         const std::vector<fp_t> &B, const matrix::matrixLayout layoutB,
                         const size_t m, const size_t k, const size_t n,
                         const matrix::matrixLayout layoutC,
                         const multiterm::config &config) {

    // Build matrix views.
    auto viewA = matrix::MatrixView<const fp_t>(A, m, k, layoutA);
    auto viewB = matrix::MatrixView<const fp_t>(B, k, n, layoutB);

    // Validate inputs and compute execution parameters.
    multiterm::validateParameters<fp_t, splitint_t, accumulator_t>(viewA, viewB, config);
    const size_t bitsPerSlice = multiterm::computeBitsPerSlice<splitint_t, accumulator_t>(k);

    // Slice operands.
    auto splitA = multiterm::prepareOperand<splitint_t, fp_t>(viewA,
        multiterm::OperandPreparationConfig(config.splitType, config.numSplitsA, bitsPerSlice, normalisationDimension::byRows));
    auto splitB = multiterm::prepareOperand<splitint_t, fp_t>(viewB,
        multiterm::OperandPreparationConfig(config.splitType, config.numSplitsB, bitsPerSlice, normalisationDimension::byCols));

    // Build multiplication schedule.
    auto schedule = multiterm::makeSchedule(config);

    // Execute multiplication based on reduction type.
    if (config.redType == multiterm::reductionStrategy::floatingPoint) {
        return multiterm::computeProductsWithFloatingPointAccumulation<splitint_t, accumulator_t, fp_t>(
            splitA, splitB, schedule, layoutC);
    } else {
        return multiterm::computeProductsWithIntegerAccumulation<splitint_t, accumulator_t, fp_t>(
            splitA, splitB, schedule, layoutC);
    }

}

template <typename fp_t, typename splitint_t, typename accumulator_t>
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const matrix::matrixLayout layoutA,
                         const std::vector<fp_t> &B, const matrix::matrixLayout layoutB,
                         const size_t m, const size_t k, const size_t n, const size_t numSplits) {
    return gemmi <fp_t, splitint_t, accumulator_t> (A, layoutA, B, layoutB, m, k, n,
                                                    matrix::matrixLayout::columnMajor,
                                                    multiterm::config{numSplits, numSplits,
                                                    multiterm::splittingStrategy::roundToNearest,
                                                    multiterm::multiplicationStrategy::reduced,
                                                    multiterm::reductionStrategy::floatingPoint});
}

#endif // GEMMI_HPP