#include <cassert>
#include "dnn.hpp"

void Dnn::addLayer(layer::DenseLayer layer) {
    layers.push_back(layer);
}

const Matrix& Dnn::evaluate(const MatrixRef& x) {
    assert(layers.size() > 0);

    auto layerIterator = layers.begin();
    Matrix y = layerIterator->forward(x);
    for (++layerIterator; layerIterator != layers.end(); ++layerIterator) {
        y = layerIterator->forward(y);
    }
    return layers[layers.size() - 1].getOutputs();
}

void Dnn::backward() {
    const auto& output = layers[layers.size() - 1].getOutputs();
    Matrix y = Matrix::Ones(output.rows(), output.cols());
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        y = it->backward(y);
    }
}

