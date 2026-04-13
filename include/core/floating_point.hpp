#ifndef GEMMI_CORE_FLOATING_POINT_HPP
#define GEMMI_CORE_FLOATING_POINT_HPP

#include <cstdint>
#include <limits>
#include <cmath>

namespace gemmi::core {

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
    auto minFPExponent = std::numeric_limits<fp_t>::min_exponent;
    auto actualExponent = std::ilogb(std::abs(value)) + 1;
    return (value == 0.0) ? 0 : std::max(minFPExponent, actualExponent);
}

} // namespace gemmi::core

#endif // GEMMI_CORE_FLOATING_POINT_HPP
