#ifndef GEMMI_CORE_TYPES_HPP
#define GEMMI_CORE_TYPES_HPP

namespace gemmi::core {

/**
 * @brief Enum to specify the dimension used for normalization.
 */
enum class normalisationDimension {
    byRows, ///< Normalise by rows (matrix on the left of the product).
    byCols  ///< Normalise by columns (matrix on the right of the product).
};

} // namespace gemmi::core

#endif // GEMMI_CORE_TYPES_HPP
