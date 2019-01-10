#include <gtest/gtest.h>

#include "layer.hpp"

using namespace layer;

TEST(DenseLayer, forward_backward_accepts_arguments) {
    DenseLayer<5, 3, activation::Identity> layer1;
    DenseLayer<3, 4, activation::Identity> layer2;

    Matrix x = Matrix::Random(2, 5);
    Matrix output1 = layer1.forward(x);
    Matrix output2 = layer2.forward(output1);
    layer1.backward(output2 * layer2.W.transpose());
}

