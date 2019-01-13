#include <gtest/gtest.h>

#include "layer.hpp"
#include "dnn.hpp"

using namespace layer;

TEST(Dnn, forward_runs) {
    Dnn dnn;

    dnn.addLayer(DenseLayer{5, 3, activation::relu});
    dnn.addLayer(DenseLayer{3, 4, activation::identity});

    Matrix x = Matrix::Random(2, 5);
    dnn.evaluate(x);
}

