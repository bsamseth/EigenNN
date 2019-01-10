#include <gtest/gtest.h>

#include "activation.hpp"

using namespace activation;

TEST(Activation, identity) {
    Array rand = Array::Random(5, 5);
    Array zeros = Array::Zero(5, 5);
    Array ones = Array::Ones(5, 5);

    EXPECT_TRUE(rand.isApprox(activation::Identity{}.evaluate(rand)));
    EXPECT_TRUE(ones.isApprox(activation::Identity{}.derivative(rand)));
    EXPECT_TRUE(zeros.isApprox(activation::Identity{}.dblDerivative(rand)));
}

TEST(Activation, relu) {
    Array rand = Array::Random(5, 5);
    Array zeros = Array::Zero(5, 5);
    Array ones = Array::Ones(5, 5);
    Array eval_expect = (rand > 0).select(rand, zeros);
    Array deriv_expect = (rand > 0).select(ones, zeros);
    Array dblDeriv_expect = zeros;

    EXPECT_TRUE(eval_expect.isApprox(activation::Relu{}.evaluate(rand)));
    EXPECT_TRUE(deriv_expect.isApprox(activation::Relu{}.derivative(rand)));
    EXPECT_TRUE(dblDeriv_expect.isApprox(activation::Relu{}.dblDerivative(rand)));
}

TEST(Activation, sigmoid) {
    Array rand = Array::Random(5, 5);
    Array zeros = Array::Zero(5, 5);
    Array ones = Array::Ones(5, 5);
    Array eval_expect = 1 / (1 + (-rand).exp());
    Array deriv_expect = eval_expect * (1 - eval_expect);
    Array dblDeriv_expect = deriv_expect * (1 - 2 * eval_expect);

    Array eval = activation::Sigmoid{}.evaluate(rand);

    EXPECT_TRUE(eval_expect.isApprox(eval));
    EXPECT_TRUE(deriv_expect.isApprox(activation::Sigmoid{}.derivative(eval)));
    EXPECT_TRUE(dblDeriv_expect.isApprox(activation::Sigmoid{}.dblDerivative(eval)));
}
