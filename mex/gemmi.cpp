#include "mex.hpp"
#include "mexAdapter.hpp"
#include "../include/gemmi.hpp"

typedef struct {
    gemmi::mt::splittingStrategy splitType;
    gemmi::mt::multiplicationSpecification multSpec;
    gemmi::mt::reductionStrategy accType;
} algorithmOptions;
static std::unique_ptr<algorithmOptions> options = nullptr;

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {

        if (options == nullptr) {
            options = std::make_unique<algorithmOptions>();
            options->splitType = gemmi::mt::splittingStrategy::roundToNearest;
            options->accType = gemmi::mt::reductionStrategy::integer;
            options->multSpec = gemmi::mt::multiplicationStrategy::reduced;
        }

        // Validate input.
        validateInput(inputs, outputs);

        size_t numSplitsA = std::move(inputs[2][0]);
        size_t numSplitsB = inputs.size() == 3 ? numSplitsA : std::move(inputs[3][0]);

        if (inputs[0].getType() == matlab::data::ArrayType::DOUBLE &&
            inputs[1].getType() == matlab::data::ArrayType::DOUBLE) {
            matlab::data::TypedArray<double> Amatlab = std::move(inputs[0]);
            matlab::data::TypedArray<double> Bmatlab = std::move(inputs[1]);
            outputs[0] = std::move(executeOperation(Amatlab, Bmatlab, numSplitsA, numSplitsB));
        } else if (inputs[0].getType() == matlab::data::ArrayType::SINGLE &&
            inputs[1].getType() == matlab::data::ArrayType::SINGLE) {
            matlab::data::TypedArray<float> Amatlab = std::move(inputs[0]);
            matlab::data::TypedArray<float> Bmatlab = std::move(inputs[1]);
            outputs[0] = std::move(executeOperation(Amatlab, Bmatlab, numSplitsA, numSplitsB));
        } else {
            std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
            matlab::data::ArrayFactory factory;
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Unsupported combination of data type.") }));
        }

        if (outputs.size() == 2) {
            matlab::data::ArrayFactory factory;
            matlab::data::StructArray S = factory.createStructArray({1, 1}, {"split", "acc", "mult"});
            switch (options->splitType) {
                case gemmi::mt::splittingStrategy::truncation:
                    S[0]["split"] = factory.createCharArray("t");
                    break;
                case gemmi::mt::splittingStrategy::unsignedEncoding:
                    S[0]["split"] = factory.createCharArray("u");
                    break;
                case gemmi::mt::splittingStrategy::roundToNearest:
                    S[0]["split"] = factory.createCharArray("n");
                    break;
                default:
                    S[0]["split"] = factory.createCharArray("unknown");
                    break;
            }
            S[0]["acc"] = factory.createCharArray(options->accType == gemmi::mt::reductionStrategy::floatingPoint ? "f" : "i");
            if (std::holds_alternative<gemmi::mt::multiplicationStrategy>(options->multSpec)) {
                const auto multType = std::get<gemmi::mt::multiplicationStrategy>(options->multSpec);
                S[0]["mult"] = factory.createCharArray(multType == gemmi::mt::multiplicationStrategy::full ? "f" : "r");
            } else {
                matlab::data::TypedArray<bool> multMask = factory.createArray<bool>({numSplitsA, numSplitsB});
                const auto& mask = std::get<std::vector<bool>>(options->multSpec);
                for (size_t row = 0; row < numSplitsA; ++row) {
                    for (size_t col = 0; col < numSplitsB; ++col) {
                        multMask[row][col] = mask[row * numSplitsB + col];
                    }
                }
                S[0]["mult"] = multMask;
            }
            outputs[1] = std::move(S);
        }
    }

private:
    matlab::data::TypedArray<double> executeOperation(const matlab::data::TypedArray<double> &Amatlab,
                          const matlab::data::TypedArray<double> &Bmatlab,
                          const size_t numSplitsA, const size_t numSplitsB) {
        const std::vector<double> A(Amatlab.begin(), Amatlab.end());
        const std::vector<double> B(Bmatlab.begin(), Bmatlab.end());
        auto A_size = Amatlab.getDimensions();
        auto B_size = Bmatlab.getDimensions();

        auto C = gemmi::mt::gemmi<double, int8_t, int32_t>(A, gemmi::core::matrixLayout::columnMajor,
                                                B, gemmi::core::matrixLayout::columnMajor,
                                                A_size[0], A_size[1], B_size[1],
                                                gemmi::core::matrixLayout::columnMajor,
                                                gemmi::mt::config{numSplitsA, numSplitsB,
                                                                  options->splitType,
                                                                  options->multSpec,
                                                                  options->accType});

        matlab::data::ArrayFactory factory;
        return factory.createArray({A_size[0], B_size[1]}, C.begin(), C.end());;
    }

    matlab::data::TypedArray<float> executeOperation(const matlab::data::TypedArray<float> &Amatlab,
                          const matlab::data::TypedArray<float> &Bmatlab,
                          const size_t numSplitsA, const size_t numSplitsB) {
        const std::vector<float> A(Amatlab.begin(), Amatlab.end());
        const std::vector<float> B(Bmatlab.begin(), Bmatlab.end());
        auto A_size = Amatlab.getDimensions();
        auto B_size = Bmatlab.getDimensions();

        auto C = gemmi::mt::gemmi<float, int8_t, int32_t>(A, gemmi::core::matrixLayout::columnMajor, B, gemmi::core::matrixLayout::columnMajor,
                                               A_size[0], A_size[1], B_size[1],
                                               gemmi::core::matrixLayout::columnMajor,
                                               gemmi::mt::config{numSplitsA, numSplitsB,
                                                                 options->splitType,
                                                                 options->multSpec,
                                                                 options->accType});

        matlab::data::ArrayFactory factory;
        return factory.createArray({A_size[0], B_size[1]}, C.begin(), C.end());;
    }

    void validateInput(matlab::mex::ArgumentList inputs, matlab::mex::ArgumentList outputs) {
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
        matlab::data::ArrayFactory factory;

        if (outputs.size() > 2) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("This function accepts at most two output arguments.") }));
        }

        size_t numArgs = inputs.size();

        if ( numArgs < 3 || numArgs > 5) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Three to five inputs expected.") }));
        }

        auto A_size = inputs[0].getDimensions();
        auto B_size = inputs[1].getDimensions();

        if (A_size.size() != 2 || B_size.size() != 2) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("The first two inputs must be matrices.") }));
        }

        if (A_size[1] != B_size[0]) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input matrices must be conformable.") }));
        }

        if (inputs[0].getType() != matlab::data::ArrayType::DOUBLE &&
            inputs[0].getType() != matlab::data::ArrayType::SINGLE) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("First input must be a real matrix.") }));
        }

        if (inputs[1].getType() != matlab::data::ArrayType::DOUBLE &&
            inputs[1].getType() != matlab::data::ArrayType::SINGLE) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Second input must be a real matrix.") }));
        }

        if (inputs[0].getType() != inputs[1].getType()) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input matrices must have the same data type.") }));
        }

        if (inputs[2].getNumberOfElements() != 1 || std::round((double)inputs[2][0][0]) != (double)inputs[2][0][0]) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("The third input must be a scalar integer.") }));
        }

        if (numArgs == 4) {
            if (inputs[3].getNumberOfElements() != 1 || std::round((double)inputs[3][0][0]) != (double)inputs[3][0][0]) {
                matlabPtr->feval(u"error",
                    0, std::vector<matlab::data::Array>({ factory.createScalar("The fourth input must be a scalar integer.") }));
            }
        }

        if (numArgs == 5) {
            if(!inputs[4].isEmpty() && !(inputs[4].getType() == matlab::data::ArrayType::STRUCT)) {
                matlabPtr->feval(u"error",
                    0, std::vector<matlab::data::Array>({ factory.createScalar("The fifth input must be a struct.") }));
            }

            const size_t numSplitsA = static_cast<size_t>((double)inputs[2][0][0]);
            const size_t numSplitsB = numArgs == 3 ? numSplitsA : static_cast<size_t>((double)inputs[3][0][0]);

            matlab::data::StructArray inStruct(inputs[4]);
            if (inStruct.getNumberOfFields() > 3) {
                matlabPtr->feval(u"error",
                    0, std::vector<matlab::data::Array>({ factory.createScalar("The fifth input must have at most three fields.") }));
            }
            auto fields = inStruct.getFieldNames();
            std::vector<matlab::data::MATLABFieldIdentifier> fieldNames(fields.begin(), fields.end());
            for (auto field : fieldNames) {
                if (std::string(field) != "split" && std::string(field) != "mult" && std::string(field) != "acc") {
                    matlabPtr->feval(u"error",
                        0, std::vector<matlab::data::Array>({ factory.createScalar("The fifth input's fields can only be named 'split', 'mult', or 'acc'.") }));
                } else {
                    if (std::string(field) == "split") {
                        if (inStruct[0][field].getNumberOfElements() != 1 || inStruct[0][field].getType() != matlab::data::ArrayType::CHAR) {
                            matlabPtr->feval(u"error",
                                0, std::vector<matlab::data::Array>({ factory.createScalar("Each field of the struct should be a single character.") }));
                        }
                        const matlab::data::TypedArrayRef<char16_t> data = inStruct[0][field];
                        switch ((char)data[0]) {
                            case 't':
                                options->splitType = gemmi::mt::splittingStrategy::truncation;
                                break;
                            case 'u':
                                options->splitType = gemmi::mt::splittingStrategy::unsignedEncoding;
                                break;
                            case 'n':
                                options->splitType = gemmi::mt::splittingStrategy::roundToNearest;
                                break;
                            default:
                                matlabPtr->feval(u"error",
                                    0, std::vector<matlab::data::Array>({ factory.createScalar("Specified 'split' is invalid.") }));
                                break;
                        }
                    } else if (std::string(field) == "acc") {
                        if (inStruct[0][field].getNumberOfElements() != 1 || inStruct[0][field].getType() != matlab::data::ArrayType::CHAR) {
                            matlabPtr->feval(u"error",
                                0, std::vector<matlab::data::Array>({ factory.createScalar("Each field of the struct should be a single character.") }));
                        }
                        const matlab::data::TypedArrayRef<char16_t> data = inStruct[0][field];
                        switch ((char)data[0]) {
                            case 'f':
                                options->accType = gemmi::mt::reductionStrategy::floatingPoint;
                                break;
                            case 'i':
                                options->accType = gemmi::mt::reductionStrategy::integer;
                                break;
                            default:
                                matlabPtr->feval(u"error",
                                    0, std::vector<matlab::data::Array>({ factory.createScalar("Specified 'acc' is invalid.") }));
                                break;
                        }
                    } else if (std::string(field) == "mult") {
                        if (inStruct[0][field].getType() == matlab::data::ArrayType::CHAR) {
                            if (inStruct[0][field].getNumberOfElements() != 1) {
                                matlabPtr->feval(u"error",
                                    0, std::vector<matlab::data::Array>({ factory.createScalar("Each field of the struct should be a single character.") }));
                            }
                            const matlab::data::TypedArrayRef<char16_t> data = inStruct[0][field];
                            switch ((char)(data[0])) {
                                case 'f':
                                    options->multSpec = gemmi::mt::multiplicationStrategy::full;
                                    break;
                                case 'r':
                                    options->multSpec = gemmi::mt::multiplicationStrategy::reduced;
                                    break;
                                default:
                                    matlabPtr->feval(u"error",
                                        0, std::vector<matlab::data::Array>({ factory.createScalar("Specified 'mult' is invalid.") }));
                                    break;
                            }
                        } else if (inStruct[0][field].getType() == matlab::data::ArrayType::LOGICAL) {
                            matlab::data::TypedArray<bool> inMask = inStruct[0][field];
                            const auto mask_dims = inMask.getDimensions();
                            if (mask_dims.size() != 2 || mask_dims[0] != numSplitsA || mask_dims[1] != numSplitsB) {
                                matlabPtr->feval(u"error",
                                    0, std::vector<matlab::data::Array>({ factory.createScalar("The 'mult' logical matrix must have size [numSplitsA, numSplitsB].") }));
                            }

                            std::vector<bool> mask;
                            mask.reserve(numSplitsA * numSplitsB);
                            // Convert from column-major (MATLAB) to row-major (C order)
                            // mask order expected by gemmi::mt::config.
                            for (size_t row = 0; row < numSplitsA; ++row) {
                                for (size_t col = 0; col < numSplitsB; ++col) {
                                    mask.push_back(inMask[row][col]);
                                }
                            }
                            options->multSpec = mask;
                        } else {
                            matlabPtr->feval(u"error",
                                0, std::vector<matlab::data::Array>({ factory.createScalar("The 'mult' field must be either a single character or a logical matrix.") }));
                        }
                    }
                }
            }
        }

    }
};
