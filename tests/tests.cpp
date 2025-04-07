#include <catch2/catch_test_macros.hpp>
#include <random>
#include <bit>
#include <vector>
#include <iostream>

#include "gemmi.hpp"
#include "utilities.hpp"

template <typename fp_t> double tolerance() {return 0;}
template <> double tolerance<float>() {return 1e-6;}
template <> double tolerance<double>() {return 1e-15;}

template <typename fp_t>
void runTest() {
    for (auto splitType : {splittingStrategy::bitMasking, splittingStrategy::roundToNearest}) {
        for (auto accumulationType : {accumulationStrategy::floatingPoint, accumulationStrategy::integer}) {
            for (auto multiplicationType : {multiplicationStrategy::reduced, multiplicationStrategy::full}) {
                for (size_t numSplitA : { 1, 2, 10 }) {
                    for (size_t numSplitB : { 1, 2, 10 }) {
                        for (size_t m = 10; m <= 50; m += 10) {
                            for (size_t k = 10; k <= 50; k += 10) {
                                for (size_t n = 10; n <= 50; n += 10) {
                                    std::vector<fp_t> A(m * k);
                                    std::vector<fp_t> B(k * n);

                                    // Initalize matrix with random values.
                                    std::default_random_engine generator(std::random_device{}());
                                    std::uniform_real_distribution<fp_t> distribution(-100000.0, 100000.0);
                                    for (auto & element : A)
                                        element = numSplitA < 10 ? ldexp(1.0, 2 * numSplitA) - 1 : distribution(generator);
                                    for (auto & element : B)
                                        element = numSplitB < 10 ? ldexp(1.0, 2 * numSplitB) - 1 : distribution(generator);

                                    auto C = gemmi<fp_t, int8_t, int32_t>(A, B, m, k, n, numSplitA, numSplitB, splitType, accumulationType, multiplicationType);
                                    auto C_ref = reference_gemm(A, B, m, k, n);

                                    double relative_error = frobenius_norm<fp_t, double>(C - C_ref) / frobenius_norm<fp_t, double>(C);

                                    REQUIRE(relative_error < tolerance<fp_t>());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("GEMMI accuracy binary64", "[gemmi]") {
    runTest<double>();
}

TEST_CASE("GEMMI accuracy binary32", "[gemmi]") {
    runTest<float>();
}