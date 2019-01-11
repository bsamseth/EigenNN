#include <gtest/gtest.h>

#include "layer.hpp"
#include "dnn.hpp"

using namespace layer;

TEST(Dnn, forward) {
    Dnn<
        DenseLayer<5, 3, activation::Relu>,
        DenseLayer<3, 4, activation::Identity>
    > dnn;

    Matrix x = Matrix::Random(2, 5);
    dnn.forward(x);
}

