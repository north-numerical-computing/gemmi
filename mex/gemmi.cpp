#include "mex.hpp"
#include "mexAdapter.hpp"
#include "../include/gemmi.hpp"

typedef struct {
    splittingStrategy splitType;
    accumulationStrategy accType;
} algorithmOptions;
static std::unique_ptr<algorithmOptions> options = nullptr;

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {

        if (options == nullptr) {
            options = std::make_unique<algorithmOptions>();
            options->splitType = splittingStrategy::roundToNearest;
            options->accType = accumulationStrategy::integer;
        }

        // Validate input.
        validateInput(inputs);

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
    }

private:
    matlab::data::TypedArray<double> executeOperation(const matlab::data::TypedArray<double> &Amatlab,
                          const matlab::data::TypedArray<double> &Bmatlab,
                          const size_t numSplitsA, const size_t numSplitsB) {
        const std::vector<double> A(Amatlab.begin(), Amatlab.end());
        const std::vector<double> B(Bmatlab.begin(), Bmatlab.end());
        auto A_size = Amatlab.getDimensions();
        auto B_size = Bmatlab.getDimensions();

        auto C = gemmi<double, int8_t, int32_t>(A, B, A_size[0], A_size[1], B_size[1],
                                                numSplitsA, numSplitsB, options->splitType, options->accType);

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

        auto C = gemmi<float, int8_t, int32_t>(A, B, A_size[0], A_size[1], B_size[1],
                                               numSplitsA, numSplitsB, options->splitType, options->accType);

        matlab::data::ArrayFactory factory;
        return factory.createArray({A_size[0], B_size[1]}, C.begin(), C.end());;
    }

    void validateInput(matlab::mex::ArgumentList inputs) {
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
        matlab::data::ArrayFactory factory;

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
            matlab::data::StructArray inStruct(inputs[4]);
            if (inStruct.getNumberOfFields() > 2) {
                matlabPtr->feval(u"error",
                    0, std::vector<matlab::data::Array>({ factory.createScalar("The fifth input must have at most two fields.") }));
            }
            auto fields = inStruct.getFieldNames();
            std::vector<matlab::data::MATLABFieldIdentifier> fieldNames(fields.begin(), fields.end());
            for (auto field : fieldNames) {
                if (std::string(field) != "split" && std::string(field) != "acc") {
                    matlabPtr->feval(u"error",
                        0, std::vector<matlab::data::Array>({ factory.createScalar("The fifth input's fields can only be named 'split' or 'acc'.") }));
                } else {
                    if (inStruct[0][field].getNumberOfElements() != 1 || inStruct[0][field].getType() != matlab::data::ArrayType::CHAR) {
                        matlabPtr->feval(u"error",
                            0, std::vector<matlab::data::Array>({ factory.createScalar("The field of the struct should be single characters.") }));
                    }
                    const matlab::data::TypedArray<char16_t> data = inStruct[0][field];
                    if (std::string(field) == "split") {
                        switch (data[0]) {
                            case 'n':
                                options->splitType = splittingStrategy::roundToNearest;
                                break;
                            case 'b':
                                options->splitType = splittingStrategy::bitMasking;
                                break;
                            default:
                                matlabPtr->feval(u"error",
                                    0, std::vector<matlab::data::Array>({ factory.createScalar("Specified 'split' is invalid.") }));
                                break;
                        }
                    } else if (std::string(field) == "acc") {
                        switch (data[0]) {
                            case 'f':
                                options->accType = accumulationStrategy::floatingPoint;
                                break;
                            case 'i':
                                options->accType = accumulationStrategy::integer;
                                break;
                            default:
                                matlabPtr->feval(u"error",
                                    0, std::vector<matlab::data::Array>({ factory.createScalar("Specified 'acc' is invalid.") }));
                                break;
                        }
                    }
                }
            }
        }

    }
};
