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

    // Test different sizes.
    for (size_t numSplitA : { 1, 2, 10 }) {
        for (size_t numSplitB : { 1, 2, 10 }) {
            for (size_t m = 10; m <= 50; m += 10) {
                for (size_t p = 10; p <= 50; p += 10) {
                    for (size_t n = 10; n <= 50; n += 10) {
                        std::vector<my_fp_type> A(m * p);
                        std::vector<my_fp_type> B(p * n);

                        std::cout << "m: " << m << ", p: " << p << ", n: " << n << std::endl;

                        // Initalize matrix with random values.
                        std::default_random_engine generator(std::random_device{}());
                        std::uniform_real_distribution<double> distribution(-100000.0, 100000.0);
                        for (auto & element : A)
                            element = numSplitA < 10 ? ldexp(1.0, 2 * numSplitA) - 1 : distribution(generator);
                        for (auto & element : B)
                            element = numSplitB < 10 ? ldexp(1.0, 2 * numSplitB) - 1 : distribution(generator);
                        auto C = gemmi<my_fp_type, int8_t, int32_t>(A, B, m, p, n, numSplitA, numSplitB);
                        auto C_ref = reference_gemm(A, B, m, p, n);

                        double relative_error = frobenius_norm<my_fp_type, double>(C - C_ref) / frobenius_norm<my_fp_type, double>(C);

                        std::cout << "Relative error: " << relative_error << std::endl;
                        assert(relative_error < 1e-15);
                    }
                }
            }
        }
    }

    return 0;
}
