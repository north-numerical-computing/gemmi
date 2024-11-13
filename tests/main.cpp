#include <random>
#include <bit>
#include <cassert>
#include <vector>

#include "gemmi.hpp"
#include "utilities.hpp"

int main() {

    // typedef float my_fp_type;
    // typedef uint32_t my_int_type;

    typedef double my_fp_type;

    size_t ms = 2, ns = 2, ps = 2;
    std::vector<my_fp_type> As(ms * ps);
    std::vector<my_fp_type> Bs(ps * ns);
    std::default_random_engine generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(-100000.0, 100000.0);
    for (size_t i = 0; i < ms; i++)
        for (size_t j = 0; j < ps; j++)
            As[i * ms + j] = distribution(generator);
    for (size_t i = 0; i < ps; i++)
        for (size_t j = 0; j < ns; j++)
            Bs[i * ns + j] = distribution(generator);
    As[0] = 1;

    auto Cs = gemmi<my_fp_type, int8_t, int32_t>(As, Bs, ms, ps, ns, 10);
    auto Cs_ref = reference_gemm(As, Bs, ms, ps, ns);

    double relErr = frobenius_norm<my_fp_type, double>(Cs - Cs_ref) / frobenius_norm<my_fp_type, double>(Cs);

    std::cout << "Relative error: " << relErr << std::endl;
    assert(relErr < 1e-15);

    return 0;

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

                auto C = gemmi<my_fp_type, int8_t, int32_t>(A, B, m, p, n, 10);
                auto C_ref = reference_gemm(A, B, m, p, n);

                double relative_error = frobenius_norm<my_fp_type, double>(C - C_ref) / frobenius_norm<my_fp_type, double>(C);

                std::cout << "Relative error: " << relative_error << std::endl;
                assert(relative_error < 1e-15);
            }
        }
    }

    //std::cout << std::hex << std::bit_cast<my_int_type>(C[0]) << std::endl;
    //std::cout << std::hex << std::bit_cast<my_int_type>(C_ref[0]) << std::endl;

    return 0;
}
