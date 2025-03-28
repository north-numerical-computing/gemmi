#ifndef UTILITIES_HPP
#define UTILITIES_HPP
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

template <typename T>
void print_matrix(std::vector<T> A, const size_t m, const size_t n,
                  const std::string id_string) {
    std::cout << id_string << std::endl;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++)
            std::cout << std::setprecision(15) << A[i + j * m] << " ";
        std::cout << std::endl;
    }
}

template <typename fp_t>
std::vector<fp_t> reference_gemm (const std::vector<fp_t> &A,
                                  const std::vector<fp_t> &B,
                         const size_t m, const size_t p, const size_t n) {
    std::vector<fp_t> C (m * n);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            C[i + j * m] = 0;
            for (size_t k = 0; k < p; k++) {
                C[i + j * m] += A[i + k * m] * B[k + j * p];
            }
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