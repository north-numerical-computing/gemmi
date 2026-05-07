#ifndef GEMMI_HPP
#define GEMMI_HPP

/*
 * @brief GEMMI: General Matrix Multiplication using Integer emulation.
 *
 * This header provides a header-only C++ library for emulating matrix
 * multiplication using integer arithmetic. The library supports multiple
 * splitting and reduction strategies.
 */

// Core utilities.
#include "core/types.hpp"
#include "core/floating_point.hpp"
#include "core/matrix_view.hpp"

// Multiterm strategy.
#include "multiterm/types.hpp"
#include "multiterm/operand.hpp"
#include "multiterm/operand_prep.hpp"
#include "multiterm/schedule.hpp"
#include "multiterm/accumulation.hpp"
#include "multiterm/gemmi.hpp"

#endif // GEMMI_HPP