#include "layer.hpp"

namespace layer {

DenseLayer::DenseLayer(int inputs, int outputs, const activation::ActivationFunction& actFunc)
    :
        W(Matrix::Random(inputs, outputs)),
        W_grad(W),
        b(RowVector::Random(outputs)),
        b_grad(b),
        actFunc(&actFunc)
{
    // TODO: Initialize W and b some sensible way.
}

const Matrix& DenseLayer::forward(const MatrixRef& x)
{
    inputs = x;
    Matrix z = (inputs * W).rowwise() + b;
    return outputs = actFunc->evaluate(z);
}

Matrix DenseLayer::backward(const MatrixRef& error)
{
    delta = error.cwiseProduct(actFunc->derivative(outputs));

    W_grad = inputs.transpose() * delta;
    b_grad = delta.colwise().mean();

    return delta * W.transpose();
}

}
