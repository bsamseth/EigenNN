#include <cassert>
#include "dnn.hpp"

void Dnn::addLayer(layer::DenseLayer layer) {
    layers.push_back(layer);
    paramCount += layer.getNumberOfParameter();

}

const Matrix& Dnn::evaluate(const MatrixRef& x) {
    forward(x);
    return layers[layers.size() - 1].getOutputs();
}

const Vector& Dnn::parameterGradient(const MatrixRef& x) {
    forward(x);
    backward();
    unsigned k = 0;
    for (const auto& layer : layers) {
        const auto& W_grad = layer.getWeightsGradient();
        for (unsigned i = 0; i < W_grad.size(); ++i)
            paramGradient(k++) = W_grad.data()[i];
        const auto& b_grad = layer.getBiasGradient();
        for (unsigned i = 0; i < W_grad.size(); ++i)
            paramGradient(k++) = b_grad[i];
    }
    assert(k == paramCount);
    return paramGradient;
}


void Dnn::forward(const MatrixRef& x) {
    assert(layers.size() > 0);

    auto layerIterator = layers.begin();
    Matrix y = layerIterator->forward(x);
    for (++layerIterator; layerIterator != layers.end(); ++layerIterator) {
        y = layerIterator->forward(y);
    }
}

void Dnn::backward() {
    const auto& output = layers[layers.size() - 1].getOutputs();
    Matrix y = Matrix::Ones(output.rows(), output.cols());
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        y = it->backward(y);
    }
}

