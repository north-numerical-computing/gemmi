#include <catch2/catch_test_macros.hpp>
#include <random>
#include <bit>
#include <vector>

#include "gemmi.hpp"
#include "utilities.hpp"

TEST_CASE("GEMMI accuracy", "[gemmi]") {

    typedef double my_fp_type;

    // Test different sizes
    for (size_t numSplitA : { 1, 2, 10 }) {
        for (size_t numSplitB : { 1, 2, 10 }) {
            for (size_t m = 10; m <= 50; m += 10) {
                for (size_t p = 10; p <= 50; p += 10) {
                    for (size_t n = 10; n <= 50; n += 10) {
                        std::vector<my_fp_type> A(m * p);
                        std::vector<my_fp_type> B(p * n);

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

                        REQUIRE(relative_error < 1e-15);
                    }
                }
            }
        }
    }
}