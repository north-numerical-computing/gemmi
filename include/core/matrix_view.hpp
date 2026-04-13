#ifndef GEMMI_CORE_MATRIX_VIEW_HPP
#define GEMMI_CORE_MATRIX_VIEW_HPP

#include <vector>
#include <stdexcept>
#include <type_traits>

namespace gemmi::core {

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

} // namespace gemmi::core

#endif // GEMMI_CORE_MATRIX_VIEW_HPP
