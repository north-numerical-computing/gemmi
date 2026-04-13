#ifndef GEMMI_MULTITERM_SCHEDULE_HPP
#define GEMMI_MULTITERM_SCHEDULE_HPP

#include "types.hpp"
#include <vector>
#include <variant>
#include <algorithm>
#include <stdexcept>

namespace gemmi::mt {

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

    /// @brief Mutable access to schedule entry at (i, j).
    inline std::vector<bool>::reference operator()(size_t i, size_t j) {
        return mask[i * numSplitsB + j];
    }

    /// @brief Const access to schedule entry at (i, j).
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

            schedule.mask = mask; // row-major copy
        }

    }, config.multSpecification);

    return schedule;
}

} // namespace gemmi::mt

#endif // GEMMI_MULTITERM_SCHEDULE_HPP
