#pragma once

#include "definitions.hpp"
#include "activation.hpp"

namespace layer {

class DenseLayer {
    protected:
        Matrix W;
        Matrix W_grad;
        RowVector b;
        RowVector b_grad;
        Matrix inputs;
        Matrix outputs;
        Matrix delta;
        const activation::ActivationFunction* actFunc;

    public:

        DenseLayer(int inputs, int outputs, const activation::ActivationFunction& actFunc);
        const Matrix& forward(const MatrixRef& x);
        Matrix backward(const MatrixRef& error);

        // Getters
        const Matrix& getOutputs();
};

inline const Matrix& DenseLayer::getOutputs() {
    return outputs;
}

}
