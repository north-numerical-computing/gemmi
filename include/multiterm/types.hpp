#ifndef GEMMI_MULTITERM_TYPES_HPP
#define GEMMI_MULTITERM_TYPES_HPP

#include <vector>
#include <variant>
#include <string>
#include <limits>

namespace gemmi::mt {

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
 *   **(1) a predefined strategy**, using a `multiplicationStrategy` value, or
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
 * config.splitType         = gemmi::multiterm::splittingStrategy::roundToNearest;
 * config.redType           = gemmi::multiterm::reductionStrategy::floatingPoint;
 * config.multSpecification = gemmi::multiterm::multiplicationStrategy::reduced;
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
 * @brief Configuration for operand preparation.
 */
struct OperandPreparationConfig {
    splittingStrategy splitType;      ///< Splitting strategy to use.
    size_t numSplits;                 ///< Number of splits to use.
    size_t bitsPerSlice;              ///< Number of bits per slice.
    normalisationDimension dimension; ///< Dimension along which to normalize.
};

} // namespace gemmi::mt

#endif // GEMMI_MULTITERM_TYPES_HPP
