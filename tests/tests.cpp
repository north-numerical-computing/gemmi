#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "gemmi.hpp"
#include "utilities.hpp"

/*********************************
 * Helper functions for testing. *
 *********************************/
template <typename fp_t>
constexpr double tolerance() = delete;
template <>
constexpr double tolerance<float>() { return 1e-6; }
template <>
constexpr double tolerance<double>() { return 1e-15; }

std::string toString(splittingStrategy strategy) {
	switch (strategy) {
	case splittingStrategy::bitMasking:
		return "bitMasking";
	case splittingStrategy::unsignedEncoding:
		return "unsignedEncoding";
	case splittingStrategy::roundToNearest:
		return "roundToNearest";
	}
	return "unknown";
}

std::string toString(accumulationStrategy strategy) {
	switch (strategy) {
	case accumulationStrategy::floatingPoint:
		return "floatingPoint";
	case accumulationStrategy::integer:
		return "integer";
	}
	return "unknown";
}

std::string toString(multiplicationStrategy strategy) {
	switch (strategy) {
	case multiplicationStrategy::reduced:
		return "reduced";
	case multiplicationStrategy::full:
		return "full";
	}
	return "unknown";
}

template <typename fp_t>
using storage_t = typename get_storage_format<fp_t>::storage_format;

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

/***********************
 * Testcase generators *
 ***********************/

template <typename fp_t>
std::vector<fp_t> makeRoundTripValues();

template <>
std::vector<float> makeRoundTripValues<float>() {
    return {
		std::bit_cast<float>(0x00000000u), //    0.0
		std::bit_cast<float>(0x41000000u), //    8.0
		std::bit_cast<float>(0x41000001u), //    8.00000095367431640625
		std::bit_cast<float>(0x41234567u), //   10.20444393157958984375
		std::bit_cast<float>(0x417FFFFFu), //   15.99999904632568359375
		std::bit_cast<float>(0x42000000u), //   16.0
		std::bit_cast<float>(0xC1000000u), //  - 8.0
		std::bit_cast<float>(0xC1000001u), //  - 8.00000095367431640625
		std::bit_cast<float>(0xC1234567u), //  -10.20444393157958984375
		std::bit_cast<float>(0xC17FFFFFu), //  -15.99999904632568359375
		std::bit_cast<float>(0xC2000000u), //  -16.0
		std::bit_cast<float>(0x4149F2CAu)  //   12.6217746734619140625
    };
}

template <>
std::vector<double> makeRoundTripValues<double>() {
	return {
		std::bit_cast<double>(0x0000000000000000ull), //   0.0
		std::bit_cast<double>(0x4040000000000000ull), //  32.0
		std::bit_cast<double>(0x4040000000000001ull), //  32.00000000000001
		std::bit_cast<double>(0x404123456789ABCDull), //  34.27555555555555
		std::bit_cast<double>(0x404FFFFFFFFFFFFFull), //  63.99999999999999
		std::bit_cast<double>(0x4050000000000000ull), //  64.0
		std::bit_cast<double>(0xC040000000000000ull), // -32.0
		std::bit_cast<double>(0xC040000000000001ull), // -32.00000000000001
		std::bit_cast<double>(0xC04123456789ABCDull), // -34.27555555555555
		std::bit_cast<double>(0xC04FFFFFFFFFFFFFull), // -63.99999999999999
		std::bit_cast<double>(0xC050000000000000ull), // -64.0
		std::bit_cast<double>(0x404921FB54442D18ull)  //  50.26548245743669
	};
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
std::vector<fp_t> reconstructFromSplit(const MatrixSplit<splitint_t, fp_t> &split) {
	std::vector<fp_t> reconstructed(split.matrix.size(), fp_t{0});
	for (size_t i = 0; i < split.otherDimension(); ++i) {
		for (size_t j = 0; j < split.innerProductDimension(); ++j) {
		const size_t index = i * split.iStride() + j * split.jStride();
		fp_t value = 0.0;
		for (size_t slice = 0; slice < split.numSplits; ++slice) {
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
void runSplitRoundTripTests() {
	constexpr size_t bitsPerSlice = 6;
	constexpr size_t m = 3;
	constexpr size_t n = 4;

	const auto matrix = makeRoundTripValues<fp_t>();
	const size_t numSplits = numSlicesForExactRoundTrip<fp_t>();

	for (auto strategy :
		{splittingStrategy::bitMasking, splittingStrategy::unsignedEncoding,
			splittingStrategy::roundToNearest}) {
		DYNAMIC_SECTION("type="
						<< (std::is_same_v<fp_t, float> ? "float" : "double")
						<< ", strategy=" << toString(strategy)
						<< ", dimension=byRows") {
		MatrixSplit<int8_t, fp_t> split(matrix, m, n, strategy, numSplits,
										bitsPerSlice,
										normalisationDimension::byRows);
		const auto reconstructed = reconstructFromSplit(split);
		requireBitwiseIdenticalVectors(reconstructed, matrix);
		}

		DYNAMIC_SECTION("type="
						<< (std::is_same_v<fp_t, float> ? "float" : "double")
						<< ", strategy=" << toString(strategy)
						<< ", dimension=byCols") {
		MatrixSplit<int8_t, fp_t> split(matrix, m, n, strategy, numSplits,
										bitsPerSlice,
										normalisationDimension::byCols);
		const auto reconstructed = reconstructFromSplit(split);
		requireBitwiseIdenticalVectors(reconstructed, matrix);
		}
	}
}

/*******************************
 * Tests matrix multiplication *
 *******************************/

template <typename fp_t>
void runGemmiAccuracyTests() {
	for (size_t m : {1u, 2u, 3u, 4u, 5u, 10u, 19u, 50u}) {
		for (size_t k : {1u, 2u, 3u, 4u, 5u, 10u, 19u, 50u}) {
			for (size_t n : {1u, 2u, 3u, 4u, 5u, 10u, 19u, 50u}) {
				for (auto splitType : {splittingStrategy::bitMasking,
									   splittingStrategy::unsignedEncoding,
									   splittingStrategy::roundToNearest}) {
					for (auto accumulationType : {accumulationStrategy::floatingPoint,
												  accumulationStrategy::integer}) {
						for (auto multiplicationType : {multiplicationStrategy::reduced,
														multiplicationStrategy::full}) {
							for (size_t numSplitA : {10u, 15u}) {
								for (size_t numSplitB : {10u, 15u}) {
									DYNAMIC_SECTION(
										"type="
										<< (std::is_same_v<fp_t, float> ? "float" : "double")
										<< ", split=" << toString(splitType)
										<< ", accumulation=" << toString(accumulationType)
										<< ", multiplication=" << toString(multiplicationType)
										<< ", numSplitA=" << numSplitA
										<< ", numSplitB=" << numSplitB) {
										const auto A = makeRandomMatrix<fp_t>(m, k, 127);
										const auto B = makeRandomMatrix<fp_t>(k, n, 255);

										const auto C = gemmi<fp_t, int8_t, int32_t>(
											A, B, m, k, n, numSplitA, numSplitB, splitType,
											accumulationType, multiplicationType);

										const auto C_ref = reference_gemm(A, B, m, k, n);

										const double relative_error =
											frobenius_norm<fp_t, double>(C - C_ref) /
											frobenius_norm<fp_t, double>(C);

										INFO("relative_error = " << relative_error);
										INFO("tolerance = " << tolerance<fp_t>());
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

/**************
 * Test cases *
 **************/
TEST_CASE("Split round-trip binary32", "[split][roundtrip][float]") {
  runSplitRoundTripTests<float>();
}

TEST_CASE("Split round-trip binary64", "[split][roundtrip][double]") {
  runSplitRoundTripTests<double>();
}

TEST_CASE("GEMMI accuracy binary32", "[gemmi][float]") {
  runGemmiAccuracyTests<float>();
}

TEST_CASE("GEMMI accuracy binary64", "[gemmi][double]") {
  runGemmiAccuracyTests<double>();
}
