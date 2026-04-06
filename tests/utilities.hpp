#ifndef UTILITIES_HPP
#define UTILITIES_HPP
#include <bitset>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

template <typename fp_t>
size_t matrixIndex(size_t i, size_t j,
                   size_t rows, size_t cols,
                   matrixLayout layout) {
    return (layout == matrixLayout::rowMajor)
        ? (i * cols + j)
        : (j * rows + i);
}

template <typename fp_t>
std::vector<fp_t> referenceGemm(
    const std::vector<fp_t>& A, matrixLayout layoutA,
    const std::vector<fp_t>& B, matrixLayout layoutB,
    size_t m, size_t k, size_t n,
    matrixLayout layoutC)
{
    std::vector<fp_t> C(m * n, fp_t{0});

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            fp_t sum = 0;
            for (size_t ell = 0; ell < k; ++ell) {
                sum += A[matrixIndex<fp_t>(i, ell, m, k, layoutA)]
                    * B[matrixIndex<fp_t>(ell, j, k, n, layoutB)];
            }
            C[matrixIndex<fp_t>(i, j, m, n, layoutC)] = sum;
        }
    }

    return C;
}

template <typename input_type, typename output_type>
output_type frobenius_norm(const std::vector<input_type> &A) {
    output_type sum = (output_type)0;
    for (size_t i = 0; i < A.size(); i++) {
        sum += A[i] * A[i];
    }
    return std::sqrt(sum);
}

template <typename element_type>
std::vector<element_type> operator-(const std::vector<element_type> &A,
                                    const std::vector<element_type> &B) {
    std::vector<element_type> C (A.size());
    for (size_t i = 0; i < A.size(); i++) {
        C[i] = A[i] - B[i];
    }
    return C;
}

#endif // UTILITIES_HPP