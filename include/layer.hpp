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
        const Matrix& getOutputs() const;
        const Matrix& getWeightsGradient() const;
        const RowVector& getBiasGradient() const;
        unsigned getNumberOfParameter() const;
};

inline const Matrix& DenseLayer::getOutputs() const {
    return outputs;
}
inline unsigned DenseLayer::getNumberOfParameter() const {
    return W.size() + b.size();
}
inline const Matrix& DenseLayer::getWeightsGradient() const {
    return W_grad;
}
inline const RowVector& DenseLayer::getBiasGradient() const {
    return b_grad;
}

}
