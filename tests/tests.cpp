#include <catch2/catch_test_macros.hpp>

#include <random>
#include <string_view>

#include "gemmi.hpp"
#include "utilities.hpp"

/*********************************
 * Helper functions for testing. *
 *********************************/
template <typename fp_t>
constexpr double tolerance() = delete;
template <>
constexpr double tolerance<float>() { return 1e-5; }
template <>
constexpr double tolerance<double>() { return 1e-14; }

template <typename fp_t>
using storage_t = typename FloatingPointTraits<fp_t>::StorageType;

template <typename fp_t>
std::string hexBits(fp_t value) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::uppercase
        << std::bit_cast<storage_t<fp_t>>(value);
    return oss.str();
}

template <typename fp_t>
bool bitwiseEqual(fp_t a, fp_t b) {
    return std::bit_cast<storage_t<fp_t>>(a) == std::bit_cast<storage_t<fp_t>>(b);
}

template <typename fp_t>
void requireBitwiseIdenticalVectors(const std::vector<fp_t> &actual,
                                    const std::vector<fp_t> &expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        INFO("index = " << i);
        INFO("actual = " << actual[i] << " (" << hexBits(actual[i]) << ")");
        INFO("expected = " << expected[i] << " (" << hexBits(expected[i]) << ")");
        REQUIRE(bitwiseEqual(actual[i], expected[i]));
    }
}

template <typename Fn>
void requireInvalidArgumentContains(Fn&& fn, std::string_view messageFragment) {
        try {
                fn();
                FAIL("Expected std::invalid_argument to be thrown");
        } catch (const std::invalid_argument& ex) {
                const std::string message = ex.what();
                REQUIRE(message.find(messageFragment) != std::string::npos);
        }
}

/***********************
 * Testcase generators *
 ***********************/

template <typename fp_t>
inline void pushWithBothSigns(std::vector<fp_t>& out,
                                typename FloatingPointTraits<fp_t>::StorageType bits) {
    using uint_t = typename FloatingPointTraits<fp_t>::StorageType;
        const size_t totalBits = sizeof(uint_t) * 8;
        const uint_t signMask  = static_cast<uint_t>(1) << (totalBits - 1);
        out.push_back(std::bit_cast<fp_t>(bits));
        out.push_back(std::bit_cast<fp_t>(bits | signMask));
}

template <typename fp_t>
std::vector<fp_t> generateValuesWithSignificand(typename FloatingPointTraits<fp_t>::StorageType pattern,
                                                int expMin, int expMax) {
    using uint_t = typename FloatingPointTraits<fp_t>::StorageType;

    constexpr size_t significandBits = FloatingPointTraits<fp_t>::numSignificandBits;
    constexpr size_t expBits  = FloatingPointTraits<fp_t>::numExponentBits;

        const uint_t fractionMask = (static_cast<uint_t>(1) << significandBits) - 1;
        const uint_t expMask  = (static_cast<uint_t>(1) << expBits) - 1;

        const uint_t fraction = pattern & fractionMask;
        const int bias = (static_cast<int>(1) << (expBits - 1)) - 1;

        std::vector<fp_t> result;
        result.reserve((expMax - expMin + 1) * 2);

        for (int e = expMin; e <= expMax; e++) {
            int stored = e + bias;
            // LCOV_EXCL_START
            if (stored <= 0 || stored >= int(expMask))
                continue;
            // LCOV_EXCL_STOP
            uint_t bits = (static_cast<uint_t>(stored) << (significandBits - 1)) | fraction;
            pushWithBothSigns<fp_t>(result, bits);
        }

        return result;
}

template <typename fp_t>
std::vector<fp_t> generateTestValues(int targetExponent) {
    using uint_t = typename FloatingPointTraits<fp_t>::StorageType;
    constexpr size_t significandBits = FloatingPointTraits<fp_t>::numSignificandBits;
    constexpr size_t expBits = FloatingPointTraits<fp_t>::numExponentBits;

    const int bias = (static_cast<int>(1) << (expBits - 1)) - 1;
        const uint_t expField = static_cast<uint_t>(targetExponent + bias) << (significandBits - 1);

        std::vector<fp_t> result;
        result.reserve(4 * (significandBits - 1) + 2);

        // Add powers of 2 and preceding values.
        for (size_t i = 1; i < significandBits; ++i) {
                uint_t frac = static_cast<uint_t>(1) << i;
                pushWithBothSigns(result, expField | (frac - 1));
                pushWithBothSigns(result, expField | frac);
        }

    // Add largest subnormal values in magnitude.
    uint_t largestPositiveSubnormal = expField | ((static_cast<uint_t>(1) << significandBits) - 1);
    pushWithBothSigns(result, largestPositiveSubnormal);

        return result;
}

template<typename fp_t>
std::vector<fp_t> generateTestSubnormals() {
    return generateTestValues<fp_t>(std::numeric_limits<fp_t>::min_exponent - 2);
}

template <typename fp_t>
std::vector<fp_t> makeRandomMatrix(size_t rows, size_t cols,
                                   std::uint64_t seed) {
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<fp_t> dist(fp_t(-100000000.0), fp_t(100000000.0));

        std::vector<fp_t> matrix(rows * cols);
        for (auto &x : matrix) {
            x = dist(gen);
        }
        return matrix;
}

/************************
 * Tests matrix slicing *
 ************************/

template <typename splitint_t, typename fp_t>
std::vector<fp_t> reconstructFromMultitermDecomposition(const multiterm::preparedOperand<splitint_t, fp_t> &split) {
    std::vector<fp_t> reconstructed(split.matrix.size(), fp_t{0});
    for (size_t i = 0; i < split.outerDimension(); ++i) {
        for (size_t j = 0; j < split.innerDimension(); ++j) {
        const size_t index = split.operandIndex(i, j);
        fp_t value = 0.0;
        for (size_t slice = 0; slice < split.prepConfig.numSplits; ++slice) {
            const auto digit =
                static_cast<fp_t>(split.memory[index + slice * split.matrix.size()]);
            value += std::ldexp(digit, -split.computeSliceBitOffset(slice));
        }
        reconstructed[index] = value * split.powersVector[i];
        }
    }

    return reconstructed;
}

template <typename fp_t>
constexpr size_t numSlicesForExactRoundTrip() = delete;
template <>
constexpr size_t numSlicesForExactRoundTrip<float>() { return 16; }
template <>
constexpr size_t numSlicesForExactRoundTrip<double>() { return 32; }

template <typename fp_t>
void runSplitRoundTripTests(const size_t bitsPerSlice, const std::vector<fp_t> testValues) {
        const size_t numSplits = numSlicesForExactRoundTrip<fp_t>();

        // Generate an even number of subnormals.
        const size_t testCount = testValues.size();

        struct Shape { size_t m, n; };
        std::vector<Shape> shapes = {
            {1, testCount},
            {2, testCount / 2},
            {testCount / 2, 2},
        };

        for (auto [m, n] : shapes) {
        for (auto layout : {matrixLayout::rowMajor,
                            matrixLayout::columnMajor}) {
            for (auto strategy : {multiterm::splittingStrategy::truncation,
                                  multiterm::splittingStrategy::unsignedEncoding,
                                  multiterm::splittingStrategy::roundToNearest}) {
                for (auto dim : {normalisationDimension::byRows,
                                 normalisationDimension::byCols}) {
                    auto config = multiterm::OperandPreparationConfig(strategy, numSplits, bitsPerSlice, dim);
                    auto split = multiterm::prepareOperand<int8_t>(
                        makeMatrixView(testValues, m, n, layout),
                        config);

                    const auto recon = reconstructFromMultitermDecomposition(split);
                    requireBitwiseIdenticalVectors(recon, testValues);
                }
            }
        }
        }
}

/*******************************
 * Tests matrix multiplication *
 *******************************/

template <typename fp_t>
void runGemmiAccuracyTests() {
    for (auto layoutA : {matrixLayout::rowMajor,
                         matrixLayout::columnMajor}) {
        for (auto layoutB : {matrixLayout::rowMajor,
                             matrixLayout::columnMajor}) {
            for (auto layoutC : {matrixLayout::rowMajor,
                                 matrixLayout::columnMajor}) {
                for (size_t m : {1u, 2u, 3u, 5u, 10u}) { //, 4u, 5u, 10u, 19u, 50u}) {
                    for (size_t k : {1u, 2u, 3u, 5u, 10u, 19u, 50u}) { // , 4u, 5u, 10u, 19u, 50u}) {
                        for (size_t n : {1u, 2u, 3u, 5u, 10u}) { //, 4u, 5u, 10u, 19u, 50u}) {
                            for (auto splitType : {multiterm::splittingStrategy::truncation,
                                                   multiterm::splittingStrategy::unsignedEncoding,
                                                   multiterm::splittingStrategy::roundToNearest}) {
                                for (auto accumulationType : {multiterm::reductionStrategy::floatingPoint,
                                                              multiterm::reductionStrategy::integer}) {
                                    for (auto multiplicationType : {multiterm::multiplicationStrategy::reduced,
                                                                    multiterm::multiplicationStrategy::full}) {
                                        for (size_t numSplitA : {10u, 15u}) {
                                            for (size_t numSplitB : {10u, 15u}) {
                                                const auto A = makeRandomMatrix<fp_t>(m, k, 127);
                                                const auto B = makeRandomMatrix<fp_t>(k, n, 255);
                                                const auto config = multiterm::config{
													numSplitA, numSplitB,
													splitType, multiplicationType, accumulationType
												};

                                                const auto C = gemmi<fp_t, int8_t, int32_t>(A, layoutA,
                                                                                            B, layoutB,
                                                                                            m, k, n,
                                                                                            layoutC,
                                                                                            config);
                                                const auto C_ref = referenceGemm<fp_t>(A, layoutA, B, layoutB, m, k, n, layoutC);

                                                const double relative_error =
                                                    frobenius_norm<fp_t, double>(C - C_ref) /
                                                    frobenius_norm<fp_t, double>(C_ref);
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
        }
    }
}

template <typename fp_t>
void runBitmaskScheduleEquivalenceTests() {
    struct Shape {
        size_t m, k, n;
    };
    const std::vector<Shape> shapes = {
        {1, 1, 1},
        {2, 3, 4},
        {5, 4, 3},
    };

    struct SliceCounts {
        size_t a, b;
    };
    const std::vector<SliceCounts> splitCounts = {
        {10, 10},
        {10, 15},
        {15, 10},
    };

    for (auto layoutA : {matrixLayout::rowMajor, matrixLayout::columnMajor}) {
        for (auto layoutB : {matrixLayout::rowMajor, matrixLayout::columnMajor}) {
            for (auto layoutC : {matrixLayout::rowMajor, matrixLayout::columnMajor}) {
                for (auto splitType : {multiterm::splittingStrategy::truncation,
                                       multiterm::splittingStrategy::unsignedEncoding,
                                       multiterm::splittingStrategy::roundToNearest}) {
                    for (auto accumulationType : {multiterm::reductionStrategy::floatingPoint,
                                                  multiterm::reductionStrategy::integer}) {
                        for (auto [m, k, n] : shapes) {
                            const auto A = makeRandomMatrix<fp_t>(m, k, 127);
                            const auto B = makeRandomMatrix<fp_t>(k, n, 255);

                            for (auto [numSplitA, numSplitB] : splitCounts) {
                                const auto makeConfig = [&](const multiterm::multiplicationSpecification& spec) {
                                    return multiterm::config{numSplitA, numSplitB, splitType,
                                                             spec, accumulationType};
                                };

                                std::vector<bool> fullMask(numSplitA * numSplitB, true);
                                std::vector<bool> reducedMask(numSplitA * numSplitB, false);
                                const size_t limit = std::max(numSplitA, numSplitB) - 1;
                                for (size_t i = 0; i < numSplitA; ++i) {
                                    for (size_t j = 0; j < numSplitB; ++j) {
                                        reducedMask[i * numSplitB + j] = (i + j <= limit);
                                    }
                                }

                                const auto run = [&](const multiterm::multiplicationSpecification& spec) {
                                    return gemmi<fp_t, int8_t, int32_t>(A, layoutA,
                                                                        B, layoutB,
                                                                        m, k, n,
                                                                        layoutC,
                                                                        makeConfig(spec));
                                };

                                REQUIRE(multiterm::makeSchedule(makeConfig(multiterm::multiplicationStrategy::full)).mask == fullMask);
                                REQUIRE(multiterm::makeSchedule(makeConfig(multiterm::multiplicationStrategy::reduced)).mask == reducedMask);
                                requireBitwiseIdenticalVectors(run(multiterm::multiplicationStrategy::full), run(fullMask));
                                requireBitwiseIdenticalVectors(run(multiterm::multiplicationStrategy::reduced), run(reducedMask));
                            }
                        }
                    }
                }
            }
        }
    }
}

/**************
 * Test cases *
 **************/

 TEST_CASE("deriveParameters exceptions on invalid inputs", "[deriveParameters][errors]") {
        using fp_t = double;

        std::vector<fp_t> Adata(2 * 3, 1.0);
        std::vector<fp_t> Bdata(3 * 2, 1.0);

        const auto A = makeConstMatrixView(makeMatrixView(Adata, 2, 3, matrixLayout::rowMajor));
        const auto B = makeConstMatrixView(makeMatrixView(Bdata, 3, 2, matrixLayout::rowMajor));

        const auto validConfig = multiterm::config{
            2,
            2,
            multiterm::splittingStrategy::roundToNearest,
            multiterm::multiplicationStrategy::full,
            multiterm::reductionStrategy::integer
        };

        SECTION("null A pointer") {
            const auto nullA = makeMatrixView(static_cast<const fp_t*>(nullptr), 2, 3, matrixLayout::rowMajor);
            requireInvalidArgumentContains([&] {
                    (void)multiterm::deriveParameters<fp_t, int8_t, int32_t>(nullA, B, validConfig);
                },
            "Matrix A has a null data pointer");
        }

        SECTION("null B pointer") {
            const auto nullB = makeMatrixView(static_cast<const fp_t*>(nullptr), 3, 2, matrixLayout::rowMajor);
            requireInvalidArgumentContains([&] {
                    (void)multiterm::deriveParameters<fp_t, int8_t, int32_t>(A, nullB, validConfig);
                },
                "Matrix B has a null data pointer");
        }

        SECTION("empty A") {
            const auto emptyA = makeMatrixView(static_cast<const fp_t*>(Adata.data()), 0, 3, matrixLayout::rowMajor);
            requireInvalidArgumentContains([&] {
                    (void)multiterm::deriveParameters<fp_t, int8_t, int32_t>(emptyA, B, validConfig);
                },
                "Matrix A is empty");
        }

        SECTION("empty B") {
            const auto emptyB = makeMatrixView(static_cast<const fp_t*>(Bdata.data()), 3, 0, matrixLayout::rowMajor);
            requireInvalidArgumentContains([&] {
                    (void)multiterm::deriveParameters<fp_t, int8_t, int32_t>(A, emptyB, validConfig);
                },
                "Matrix B is empty");
        }

        SECTION("dimension mismatch") {
            std::vector<fp_t> badBData(4 * 2, 1.0);
            const auto badB = makeConstMatrixView(makeMatrixView(badBData, 4, 2, matrixLayout::rowMajor));
            requireInvalidArgumentContains([&] {
                    (void)multiterm::deriveParameters<fp_t, int8_t, int32_t>(A, badB, validConfig);
                },
                "Dimension mismatch");
        }

        SECTION("numSplitsA must be >= 1") {
            auto cfg = validConfig;
            cfg.numSplitsA = 0;
            requireInvalidArgumentContains([&] {
                    (void)multiterm::deriveParameters<fp_t, int8_t, int32_t>(A, B, cfg);
                },
                "numSplitsA must be >= 1");
        }

        SECTION("numSplitsB must be >= 1") {
            auto cfg = validConfig;
            cfg.numSplitsB = 0;
            requireInvalidArgumentContains([&] {
                    (void)multiterm::deriveParameters<fp_t, int8_t, int32_t>(A, B, cfg);
                },
                "numSplitsB must be >= 1");
        }

        SECTION("custom mask size mismatch") {
            auto cfg = validConfig;
            cfg.multSpecification = std::vector<bool>{true, false, true};
            requireInvalidArgumentContains([&] {
                    (void)multiterm::deriveParameters<fp_t, int8_t, int32_t>(A, B, cfg);
                },
                "Custom mask size");
        }

        SECTION("split integer type too wide for accumulator") {
            requireInvalidArgumentContains([&] {
                    (void)multiterm::deriveParameters<fp_t, int32_t, int32_t>(A, B, validConfig);
                },
                "splitint_t");
        }

        SECTION("bitsPerSlice evaluates to zero") {
            fp_t value = 1.0;
            constexpr size_t hugeK = (size_t{1} << 31);
            const auto hugeA = makeMatrixView(static_cast<const fp_t*>(&value), 1, hugeK, matrixLayout::rowMajor);
            const auto hugeB = makeMatrixView(static_cast<const fp_t*>(&value), hugeK, 1, matrixLayout::rowMajor);

            requireInvalidArgumentContains([&] {
	                (void)multiterm::deriveParameters<fp_t, int8_t, int32_t>(hugeA, hugeB, validConfig);
                },
                "Computed bitsPerSlice is 0");
        }
}

TEST_CASE("Bitmask schedule validation", "[schedule][mask]") {
    const auto config = multiterm::config{3, 2,
                                          multiterm::splittingStrategy::roundToNearest,
                                          std::vector<bool>{true},
                                          multiterm::reductionStrategy::integer};
    requireInvalidArgumentContains([&] { (void)multiterm::makeSchedule(config); },
                                   "Mask size mismatch");
}

TEST_CASE("Bitmask schedule equivalence binary32", "[schedule][mask][float]") {
    runBitmaskScheduleEquivalenceTests<float>();
}

TEST_CASE("Bitmask schedule equivalence binary64", "[schedule][mask][double]") {
    runBitmaskScheduleEquivalenceTests<double>();
}

TEST_CASE("Split round-trip binary32 – subnormals", "[split][subnormals][float]") {
    runSplitRoundTripTests<float>(6, generateTestSubnormals<float>());
    runSplitRoundTripTests<float>(7, generateTestSubnormals<float>());
    auto res = generateTestValues<float>(0);
}

TEST_CASE("Split round-trip binary64 – subnormals", "[split][subnormals][double]") {
    runSplitRoundTripTests<double>(6, generateTestSubnormals<double>());
    runSplitRoundTripTests<double>(7, generateTestSubnormals<double>());
}

TEST_CASE("Split round-trip binary32 – normals in [1, 2)", "[split][roundtrip][float]") {
    runSplitRoundTripTests<float>(6, generateTestValues<float>(0));
    runSplitRoundTripTests<float>(7, generateTestValues<float>(0));
}

TEST_CASE("Split round-trip binary64 – normals in [1, 2)", "[split][roundtrip][double]") {
    runSplitRoundTripTests<double>(6, generateTestValues<double>(0));
    runSplitRoundTripTests<double>(7, generateTestValues<double>(0));
}

TEST_CASE("Split round-trip binary32 – wide range", "[split][roundtrip][float]") {
    runSplitRoundTripTests<float>(6, generateValuesWithSignificand<float>(0xFFFFFF, -10, 10));
    runSplitRoundTripTests<float>(7, generateValuesWithSignificand<float>(0xFFFFFF, -10, 10));
}

TEST_CASE("Split round-trip binary64 – wide range", "[split][roundtrip][double]") {
    runSplitRoundTripTests<double>(6, generateValuesWithSignificand<double>(0xFFFFFFFFFFFFF, -10, 10));
    runSplitRoundTripTests<double>(7, generateValuesWithSignificand<double>(0xFFFFFFFFFFFFF, -10, 10));
}

TEST_CASE("GEMMI accuracy binary32", "[gemmi][float]") {
    runGemmiAccuracyTests<float>();
}

TEST_CASE("GEMMI accuracy binary64", "[gemmi][double]") {
    runGemmiAccuracyTests<double>();
}