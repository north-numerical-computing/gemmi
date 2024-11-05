#include <random>
#include <bit>
#include <cassert>
#include <vector>

#include "gemmi.hpp"
#include "utilities.hpp"

int main() {

/*     typedef float my_fp_type;
    typedef uint32_t my_int_type;
    const size_t n_exp_bits = 8;
    const size_t n_frac_bits = 24; */

    typedef double my_fp_type;
    typedef uint64_t my_int_type;

    // Genereate a 5 x 3 matrix.
    for (size_t m = 50; m <= 200; m += 50) {
        for (size_t p = 50; p <= 200; p += 50) {
            for (size_t n = 50; n <= 200; n += 50) {
                std::vector<my_fp_type> A(m * p);
                std::vector<my_fp_type> B(p * n);

                std::cout << "m: " << m << ", p: " << p << ", n: " << n << std::endl;

                // Initalize matrix with random values between -10 and 10.
                std::default_random_engine generator(std::random_device{}());
                std::uniform_real_distribution<double> distribution(-100000.0, 100000.0);
                for (size_t i = 0; i < m; i++)
                    for (size_t j = 0; j < p; j++)
                        A[i * m + j] = distribution(generator);
                for (size_t i = 0; i < p; i++)
                    for (size_t j = 0; j < n; j++)
                        B[i * n + j] = distribution(generator);
                A[0] = 1;

                auto C = gemmi<int8_t, int32_t, my_fp_type, my_int_type>(A, B, m, p, n, 15);
                auto C_ref = reference_gemm(A, B, m, p, n);

                double relative_error = frobenius_norm<my_fp_type, double>(C - C_ref) / frobenius_norm<my_fp_type, double>(C);

                assert(relative_error < 1e-15);

                std::cout << "Relative error: " << relative_error << std::endl;
            }
        }
    }

    //std::cout << std::hex << std::bit_cast<my_int_type>(C[0]) << std::endl;
    //std::cout << std::hex << std::bit_cast<my_int_type>(C_ref[0]) << std::endl;

    return 0;
}
