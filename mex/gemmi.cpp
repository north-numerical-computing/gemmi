#include "mex.hpp"
#include "mexAdapter.hpp"
#include "../include/gemmi.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {

        // Validate input.
        validateInput(inputs);

        size_t num_splits = std::move(inputs[2][0]);
        if (inputs[0].getType() == matlab::data::ArrayType::DOUBLE &&
            inputs[1].getType() == matlab::data::ArrayType::DOUBLE) {
            matlab::data::TypedArray<double> A_matlab = std::move(inputs[0]);
            matlab::data::TypedArray<double> B_matlab = std::move(inputs[1]);
            outputs[0] = std::move(executeOperation(A_matlab, B_matlab, num_splits));
        } else if (inputs[0].getType() == matlab::data::ArrayType::SINGLE &&
            inputs[1].getType() == matlab::data::ArrayType::SINGLE) {
            matlab::data::TypedArray<float> A_matlab = std::move(inputs[0]);
            matlab::data::TypedArray<float> B_matlab = std::move(inputs[1]);
            outputs[0] = std::move(executeOperation(A_matlab, B_matlab, num_splits));
        } else {
            std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
            matlab::data::ArrayFactory factory;
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Unsupported combination of data type.") }));
        }
    }

private:
    matlab::data::TypedArray<double> executeOperation(const matlab::data::TypedArray<double> &A_matlab,
                          const matlab::data::TypedArray<double> &B_matlab,
                          const size_t num_splits) {
        const std::vector<double> A(A_matlab.begin(), A_matlab.end());
        const std::vector<double> B(B_matlab.begin(), B_matlab.end());
        auto A_size = A_matlab.getDimensions();
        auto B_size = B_matlab.getDimensions();

        auto C = gemmi<int8_t, int32_t, double, uint64_t, 11, 53>(A, B, A_size[0], A_size[1], B_size[1], (size_t)num_splits);

        matlab::data::ArrayFactory factory;
        return factory.createArray({A_size[0], B_size[1]}, C.begin(), C.end());;
    }

    matlab::data::TypedArray<float> executeOperation(const matlab::data::TypedArray<float> &A_matlab,
                          const matlab::data::TypedArray<float> &B_matlab,
                          const size_t num_splits) {
        const std::vector<float> A(A_matlab.begin(), A_matlab.end());
        const std::vector<float> B(B_matlab.begin(), B_matlab.end());
        auto A_size = A_matlab.getDimensions();
        auto B_size = B_matlab.getDimensions();

        auto C = gemmi<int8_t, int32_t, float, uint32_t, 8, 24>(A, B, A_size[0], A_size[1], B_size[1], (size_t)num_splits);

        matlab::data::ArrayFactory factory;
        return factory.createArray({A_size[0], B_size[1]}, C.begin(), C.end());;
    }

    void validateInput(matlab::mex::ArgumentList inputs) {
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
        matlab::data::ArrayFactory factory;

        if (inputs.size() != 3) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Three inputs expected.") }));
        }

        auto A_size = inputs[0].getDimensions();
        auto B_size = inputs[1].getDimensions();

        if (A_size.size() != 2 || B_size.size() != 2) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("The first two inputs must be matrices.") }));
        }

        if (inputs[2].getNumberOfElements() != 1 || std::round((double)inputs[2][0][0]) != (double)inputs[2][0][0]) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("The third input must be a scalar integer.") }));
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
    }
};
