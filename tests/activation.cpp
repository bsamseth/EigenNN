#include <gtest/gtest.h>

#include "activation.hpp"

using namespace activation;

TEST(Activation, identity) {
    Matrix rand = Matrix::Random(5, 5);
    Matrix zeros = Matrix::Zero(5, 5);
    Matrix ones = Matrix::Ones(5, 5);

    EXPECT_TRUE(rand.isApprox(activation::identity.evaluate(rand)));
    EXPECT_TRUE(ones.isApprox(activation::identity.derivative(rand)));
    EXPECT_TRUE(zeros.isApprox(activation::identity.dblDerivative(rand)));
}

TEST(Activation, relu) {
    Array rand = Array::Random(5, 5);
    Array zeros = Array::Zero(5, 5);
    Array ones = Array::Ones(5, 5);
    Array eval_expect = (rand.array() > 0).select(rand, zeros);
    Array deriv_expect = (rand.array() > 0).select(ones, zeros);
    Array dblDeriv_expect = zeros;

    EXPECT_TRUE(eval_expect.isApprox(activation::relu.evaluate(rand).array()));
    EXPECT_TRUE(deriv_expect.isApprox(activation::relu.derivative(rand).array()));
    EXPECT_TRUE(dblDeriv_expect.isApprox(activation::relu.dblDerivative(rand).array()));
}

TEST(Activation, sigmoid) {
    Array rand = Array::Random(5, 5);
    Array zeros = Array::Zero(5, 5);
    Array ones = Array::Ones(5, 5);
    Array eval_expect = 1 / (1 + (-rand).exp());
    Array deriv_expect = eval_expect * (1 - eval_expect);
    Array dblDeriv_expect = deriv_expect * (1 - 2 * eval_expect);

    Array eval = activation::sigmoid.evaluate(rand);

    EXPECT_TRUE(eval_expect.isApprox(eval));
    EXPECT_TRUE(deriv_expect.isApprox(activation::sigmoid.derivative(eval).array()));
    EXPECT_TRUE(dblDeriv_expect.isApprox(activation::sigmoid.dblDerivative(eval).array()));
}
