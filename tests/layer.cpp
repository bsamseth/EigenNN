#include <gtest/gtest.h>

#include "layer.hpp"

using namespace layer;

TEST(DenseLayer, forward_backward_accepts_arguments) {
    DenseLayer layer1 {5, 3, activation::identity};
    DenseLayer layer2 {3, 4, activation::identity};

    Matrix x = Matrix::Random(2, 5);
    Matrix error = Matrix::Random(2, 3);
    Matrix output1 = layer1.forward(x);
    Matrix output2 = layer2.forward(output1);
    Matrix result = layer1.backward(error);
}

