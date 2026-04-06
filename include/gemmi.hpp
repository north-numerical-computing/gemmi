#ifndef GEMMI_HPP
#define GEMMI_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

/**
 * @file gemmi.hpp
 * @brief Floating-point matrix multiplication using integer emulation.
 */

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
 * It does not own the underlying memory but provides read and write to it.
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
     * @brief Contruct a matrix biew from raw parts.
     *
     * @param data Pointer to the first matrix element.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param layout Memory layout.
     */
    MatrixView(value_t* data, size_t rows, size_t cols, matrixLayout layout) :
        data(data), rows(rows), cols(cols), layout(layout) {}

    /**
     * @brief Return the number of stored elements.
     *
     * @return rows * cols
     */
    size_t size() const {
        return rows * cols;
    }

    /**
     * @brief Return true if the view is empty.
     *
     * @return true if rows == 0 or cols == 0
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
};

/**
 * @brief Create a mutable matrix view from raw pointer.
 *
 * @tparam value_t Element type.
 * @param data Pointer to the first matrix element.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param layout Memory layout.
 * @return MatrixView<value_t>
 */
template <typename value_t>
MatrixView<value_t> makeMatrixView(value_t* data,
                                   size_t rows,
                                   size_t cols,
                                   matrixLayout layout) {
    return MatrixView<value_t>(data, rows, cols, layout);
}

/**
 * @brief Create a read-only matrix view from raw pointer.
 *
 * @tparam value_t Element type.
 * @param data Pointer to the first matrix element.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param layout Memory layout.
 * @return MatrixView<const value_t>
 */
template <typename value_t>
MatrixView<const value_t> makeMatrixView(const value_t* data,
                                         size_t rows,
                                         size_t cols,
                                         matrixLayout layout) {
    return MatrixView<const value_t>(data, rows, cols, layout);
}

/**
 * @brief Create a mutable matrix view from a std::vector.
 *
 * @tparam value_t Element type.
 * @param matrix Matrix storage.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param layout Memory layout.
 * @return MatrixView<value_t>
 */
template <typename value_t>
MatrixView<value_t> makeMatrixView(std::vector<value_t>& matrix,
                                   size_t rows,
                                   size_t cols,
                                   matrixLayout layout) {
    return MatrixView<value_t>(matrix.data(), rows, cols, layout);
}

/**
 * @brief Create a read-only matrix view from a const std::vector.
 *
 * @tparam value_t Element type.
 * @param matrix Matrix storage.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param layout Memory layout.
 * @return MatrixView<const value_t>
 */
template <typename value_t>
MatrixView<const value_t> makeMatrixView(const std::vector<value_t>& matrix,
                                         size_t rows,
                                         size_t cols,
                                         matrixLayout layout) {
    return MatrixView<const value_t>(matrix.data(), rows, cols, layout);
}

/**
 * @brief Create a read-only matrix view from a mutable MatrixView.
 *
 * This is useful to pass a mutable view into code that expects a read-only
 * view without rebuilding it manually.
 *
 * @tparam value_t Element type.
 * @param view Mutable matrix view.
 * @return MatrixView<const value_t>
 */
template <typename value_t>
MatrixView<const value_t> makeConstMatrixView(MatrixView<value_t> view) {
    return MatrixView<const value_t>(view.data, view.rows, view.cols, view.layout);
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
    truncation,       ///< Split using truncation.
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
    MatrixView<const fp_t> matrix;    ///< View of original matrix.
    splittingStrategy splitType;      ///< Splitting strategy used.
    size_t numSplits;                 ///< Number of splits to use.
    size_t bitsPerSlice;              ///< Number of bits per slice.
    normalisationDimension dimension; ///< Dimension along wich to normalize.

    std::vector<splitint_t> memory;    ///< Memory to store the split slices.
    std::vector<fp_t> powersVector;    ///< Normalisation vector.
    std::vector<int> scalingExponents; ///< Scaling exponents.

    using uint_t = typename FloatingPointTraits<fp_t>::StorageType;
    using wideint_t = std::conditional_t<(sizeof(splitint_t) < sizeof(int)), int, std::intmax_t>;

    /**
     * @brief Construct a MatrixSplit object and compute the splits.
     * @param matrix View of original matrix.
     * @param splitType Splitting strategy.
     * @param numSplits Number of splits.
     * @param bitsPerSlice Number of bits per slice.
     * @param dimension Normalization dimension.
     */
    MatrixSplit(const MatrixView<const fp_t>& matrix,
                const splittingStrategy splitType, size_t numSplits, size_t bitsPerSlice,
                const normalisationDimension dimension) :
                matrix(matrix), splitType(splitType), numSplits(numSplits), bitsPerSlice(bitsPerSlice),
                dimension(dimension) {
        this->memory.resize(matrix.rows * matrix.cols * numSplits);
        this->powersVector.resize(this->outerDimension());
        this->scalingExponents.resize(this->outerDimension());
    }

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
        return (dimension == normalisationDimension::byRows) ?
            matrix.cols : matrix.rows;
    }

    /**
     * @brief Return the dimension not used in the inner product.
     * @return Dimension not used in the inner product.
     */
    size_t outerDimension() const {
        return (dimension == normalisationDimension::byRows) ?
            matrix.rows : matrix.cols;
    }

    /**
     * @brief Compute the index for a given operand in the matrix.
     * @param outer The outer index.
     * @param inner The inner index.
     * @return The index of the operand in the matrix.
     */
    size_t operandIndex(size_t outer, size_t inner) const {
        return (dimension == normalisationDimension::byRows) ?
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
     * @brief Compute normalization vectors for the matrix.
     */
    void computeNormalisationVectors() {
        // Compute normalisation vector.
        for (size_t outer = 0; outer < this->outerDimension(); outer++) {
            this->powersVector[outer] = 0.0;
            for (size_t j = 0; j < this->innerDimension(); j++) {
                const size_t matrixIndex = operandIndex(outer, j);
                this->powersVector[outer] = std::max(this->powersVector[outer],
                                                  std::abs(this->matrix.data[matrixIndex]));
            }
            // Compute the smallest power of 2 that is strictly greater than the
            // maximum value in the row/column.
            // NOTE 1: This is not the technique used in uoi24.
            // NOTE 2: I use exponents instead of powers of 2, as I need the former
            //         to shift correctly.
            this->scalingExponents[outer] = getStoredFloatingPointExponent(this->powersVector[outer]);
            this->powersVector[outer] = std::ldexp(1.0, this->scalingExponents[outer]);
        }
    }

    /**
     * @brief Compute the bit offset for a given slice.
     * @param slice The slice for which to compute the bit offset.
     * @return The bit offset for the specified slice.
     */
    int computeSliceBitOffset(size_t slice) const {
        switch (splitType) {
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

    /**
     * @brief Compute the block fixed-point representation of a row/column of the matrix.
     * This function computes the fixed-point representation of a row/column of the matrix, which is used in the splitting algorithms. It extracts the significand and sign of each element in the row/column, and stores them in the provided vectors.
     * @param fraction Vector to store the fixed-point representation of the elements.
     * @param sign Vector to store the signs of the elements.
     * @param i The index of the row/column for which to compute the fixed-point representation.
     */
    void computeFixedPointRepresentationVector(std::vector<uint_t> &fraction, std::vector<bool> &sign, size_t outer) {
        constexpr size_t numSignificandBits = FloatingPointTraits<fp_t>::numSignificandBits;
        for (size_t inner = 0; inner < this->innerDimension(); inner++) {
                const size_t matrixIndex = operandIndex(outer, inner);
                fp_t value = this->matrix.data[matrixIndex];
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
     */
    void computeSplitsWithTruncation() {
        this->splitType = splittingStrategy::truncation;
        // Compute splits one row/column at a time.
        constexpr size_t numSignificandBits = FloatingPointTraits<fp_t>::numSignificandBits;
        auto bitsPerSlice = this->bitsPerSlice;
        std::vector<uint_t> fraction (this->innerDimension());
        std::vector<bool> sign (this->innerDimension());
        for (size_t outer = 0; outer < this->outerDimension(); outer++) {
            // Get binary representation of significands of normalised row/column.
            computeFixedPointRepresentationVector(fraction, sign, outer);

            // Create bitmask.
            const uint_t smallBitmask = (static_cast<uint_t>(1) << bitsPerSlice) - 1;

            // Perform the split.
            for (size_t inner = 0; inner < this->innerDimension(); inner++) {
                // NOTE: I could have a special path for 0.
                int16_t shiftCounter = numSignificandBits - bitsPerSlice;
                int currentExponent = getStoredFloatingPointExponent(this->matrix.data[operandIndex(outer, inner)]);
                int16_t exponentDifference = scalingExponents[outer] - currentExponent;
                for (size_t slice = 0; slice < numSplits; slice++) {
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
                        splitint_t value = (splitint_t)(currentSplit) * (sign[inner] ? -1 : 1);
                        this->memory[splitIndex(outer, inner, slice)] = value;
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
        constexpr size_t numSignificandBits = FloatingPointTraits<fp_t>::numSignificandBits;
        auto bitsPerSlice = this->bitsPerSlice;
        std::vector<uint_t> fraction (this->innerDimension());
        std::vector<bool> sign (this->innerDimension());
        for (size_t outer = 0; outer < this->outerDimension(); outer++) {
            // Get binary representation of significands of normalised row/column.
            computeFixedPointRepresentationVector(fraction, sign, outer);

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

            for (size_t inner = 0; inner < this->innerDimension(); inner++) {

                auto matrixIndex = operandIndex(outer, inner);

                // NOTE: I could have a special path for 0.
                int16_t shiftCounter;
                int currentExponent = getStoredFloatingPointExponent(this->matrix.data[matrixIndex]);
                int16_t exponentDifference = scalingExponents[outer] - currentExponent;

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
                this->memory[matrixIndex] = value;

                // Remaining slices.
                const auto width  = bitsPerSlice + 1;
                const auto cutoff = static_cast<wideint_t>(static_cast<uint_t>(1) << bitsPerSlice); // 2^b
                const auto base   = static_cast<wideint_t>(static_cast<uint_t>(1) << width);        // 2^(b+1)
                const auto digitMax = cutoff - 1;   //  2^b - 1
                const auto digitMin = -cutoff;      // -2^b
                for (size_t slice = 1; slice < numSplits; slice++) {
                    if (exponentDifference > (signed)(bitsPerSlice + 1)) {
                        exponentDifference -= (bitsPerSlice + 1);
                        if (sign[inner]) {
                            this->memory[splitIndex(outer, inner, slice - 1)] = static_cast<splitint_t>(0);
                            this->memory[splitIndex(outer, inner, slice)] = static_cast<splitint_t>(-1);
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
                            int nextSlice = static_cast<int>(slice) - 1;
                            while (true) {
                                const size_t prevIdx = splitIndex(outer, inner, nextSlice);
                                auto newValue = static_cast<wideint_t>(this->memory[prevIdx]) + 1;

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

                        this->memory[splitIndex(outer, inner, slice)] = static_cast<splitint_t>(signedDigit);
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
        auto localMatrix = std::vector<fp_t>(this->matrix.data, this->matrix.data + this->matrix.size());
        for (size_t slice = 0; slice < numSplits; slice++) {
            for (size_t outer = 0; outer < this->outerDimension(); outer++) {
                fp_t sigma = ldexp(0.75,
                                   FloatingPointTraits<fp_t>::numSignificandBits - bitsPerSlice * slice + 1 - bitsPerSlice) *
                             powersVector[outer];
                for (size_t inner = 0; inner < this->innerDimension(); inner++) {
                    auto matrixIndex = operandIndex(outer, inner);
                    auto value = (localMatrix[matrixIndex] + sigma);
                    value -= sigma;
                    localMatrix[matrixIndex] -= value;
                    value = value / powersVector[outer] * ldexp(1.0, bitsPerSlice * slice + bitsPerSlice - 1);
                    this->memory[splitIndex(outer, inner, slice)] = value;
                }
            }
        }
    }

    void prepare() {
        this->computeNormalisationVectors();
        switch (this->splitType) {
            case splittingStrategy::truncation:
                this->computeSplitsWithTruncation();
                break;
            case splittingStrategy::unsignedEncoding:
                this->computeSplitsWithUnsignedEncoding();
                break;
            case splittingStrategy::roundToNearest:
                this->computeSplitsWithRoundToNearest();
                break;
        }
    }
};

/*************************************
 * Perform the matrix multiplication *
 *************************************/

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
                             const MatrixSplit<splitint_t, fp_t> &B,
                             std::vector<accumulator_t> &C,
                             const matrixLayout layoutC,
                             size_t iBlock, size_t jBlock) {
    for (size_t row = 0; row < A.rows(); row++) {
        for (size_t col = 0; col < B.cols(); col++) {
            for (size_t ell = 0; ell < A.innerDimension(); ell++) {
                auto index = (layoutC == matrixLayout::columnMajor) ? (row + col * A.rows()) : (col + row * B.cols());
                C[index] += A.memory[A.splitIndex(row, ell, iBlock)] * B.memory[B.splitIndex(col, ell, jBlock)];
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
                                                               const matrixLayout layoutC,
                                                               const size_t numDiagonals) {
    std::vector<fp_t> C (A.rows() * B.cols(), 0.0);
    for (size_t diagonal = 0; diagonal <= numDiagonals; diagonal++) {
        int Aindex = diagonal < A.numSplits - 1 ? diagonal : A.numSplits - 1;
        size_t Bindex = diagonal > A.numSplits - 1 ? diagonal - A.numSplits + 1 : 0;
        while (Aindex >= 0 && Bindex <= std::min(diagonal, B.numSplits - 1)) {
            std::vector<accumulator_t> accumulator (A.rows() * B.cols(), 0.0);
            computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t>(A, B, accumulator, layoutC, Aindex, Bindex);
            for (size_t row = 0; row < A.rows(); row++) {
                for (size_t col = 0; col < B.cols(); col++) {
                    int totalShift = A.computeSliceBitOffset(static_cast<size_t>(Aindex)) + B.computeSliceBitOffset(Bindex);
                    auto index = (layoutC == matrixLayout::columnMajor) ? (row + col * A.rows()) : (col + row * B.cols());
                    fp_t scaledSum = std::ldexp(static_cast<fp_t>(accumulator[index]), -totalShift);
                    fp_t scalingFactor = A.powersVector[row] * B.powersVector[col];
                    C[index] += scaledSum * scalingFactor;
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
                                                         const matrixLayout layoutC,
                                                         const size_t numDiagonals) {
    
    std::vector<fp_t> C (A.rows() * B.cols(), 0.0);
    for (size_t diagonal = 0; diagonal <= numDiagonals; diagonal++) {
        int Aindex = diagonal < A.numSplits ? static_cast<int>(diagonal) : static_cast<int>(A.numSplits - 1);
        size_t Bindex = diagonal > A.numSplits - 1 ? diagonal - A.numSplits + 1 : 0;

        const int totalShift = A.computeSliceBitOffset(static_cast<size_t>(Aindex)) + B.computeSliceBitOffset(Bindex);

        // Compute and accumulate all products along this anti-diagonal in integer arithmetic.
        std::vector<accumulator_t> accumulator(A.rows() * B.cols(), 0);
        while (Aindex >= 0 && Bindex <= std::min(diagonal, B.numSplits - 1)) {
            computeExactIntegerGEMM<splitint_t, accumulator_t, fp_t>(A, B, accumulator, layoutC, Aindex, Bindex);
            Aindex--;
            Bindex++;
        }

        // Scale the accumulated products and accumulate in floating-point arithmetic across diagonals.
        for (size_t row = 0; row < A.rows(); row++) {
            for (size_t col = 0; col < B.cols(); col++) {
                auto index = (layoutC == matrixLayout::columnMajor) ? (row + col * A.rows()) : (col + row * B.cols());
                fp_t scaledSum = std::ldexp(static_cast<fp_t>(accumulator[index]), -totalShift);
                fp_t scalingFactor = A.powersVector[row] * B.powersVector[col];
                C[index] += scaledSum * scalingFactor;
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
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const matrixLayout layoutA,
                         const std::vector<fp_t> &B, const matrixLayout layoutB,
                         const size_t m, const size_t k, const size_t n,
                         const size_t numSplitsA, const size_t numSplitsB,
                         const matrixLayout layoutC = matrixLayout::columnMajor,
                         const splittingStrategy splitType = splittingStrategy::roundToNearest,
                         const accumulationStrategy accType = accumulationStrategy::floatingPoint,
                         const multiplicationStrategy multType = multiplicationStrategy::reduced) {

    const size_t bitsInAccumulator = std::numeric_limits<accumulator_t>::digits;
    const size_t bitsPerInteger = std::numeric_limits<splitint_t>::digits;
    if (bitsPerInteger > bitsInAccumulator / 2) {
        throw std::invalid_argument("Split integer type is too wide for the chosen accumulator type");
    }
    const size_t alpha = std::floor((bitsInAccumulator - log2(k)) / 2);
    const size_t bitsPerSlice = std::min(bitsPerInteger, static_cast<size_t>(alpha));

    auto viewA = makeMatrixView(A, m, k, layoutA);
    auto viewB = makeMatrixView(B, k, n, layoutB);

    auto splitA = MatrixSplit<splitint_t, fp_t>(viewA, splitType, numSplitsA, bitsPerSlice, normalisationDimension::byRows);
    auto splitB = MatrixSplit<splitint_t, fp_t>(viewB, splitType, numSplitsB, bitsPerSlice, normalisationDimension::byCols);

    splitA.prepare();
    splitB.prepare();

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
            throw std::logic_error("Unhandled multiplicationStrategy");
        // LCOV_EXCL_STOP
    }

    switch (accType) {
        case accumulationStrategy::floatingPoint:
            return computeProductsWithFloatingPointAccumulation<splitint_t, accumulator_t, fp_t>(splitA, splitB, layoutC, numDiagonals);
        case accumulationStrategy::integer:
            return computeProductsWithIntegerAccumulation<splitint_t, accumulator_t, fp_t>(splitA, splitB, layoutC, numDiagonals);
        // LCOV_EXCL_START
        default:
            throw std::logic_error("Unhandled accumulationStrategy");
        // LCOV_EXCL_STOP   
    }
}

template <typename fp_t, typename splitint_t, typename accumulator_t>
std::vector<fp_t> gemmi (const std::vector<fp_t> &A, const matrixLayout layoutA,
                         const std::vector<fp_t> &B, const matrixLayout layoutB,
                         const size_t m, const size_t k, const size_t n, const size_t numSplits) {
    return gemmi <fp_t, splitint_t, accumulator_t> (A, layoutA, B, layoutB, m, k, n, numSplits, numSplits);
}

#endif // GEMMI_HPP