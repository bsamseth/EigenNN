#pragma once

#include "definitions.hpp"
#include "activation.hpp"

namespace layer {

template<unsigned Inputs, unsigned Outputs, typename ActFunc>
class DenseLayer {
    public:
        Eigen::Matrix<Real, Inputs, Outputs, Eigen::RowMajor> W;
        Eigen::Matrix<Real, Inputs, Outputs, Eigen::RowMajor> W_grad;
        Eigen::Matrix<Real, 1, Outputs, Eigen::RowMajor> b;
        Eigen::Matrix<Real, 1, Outputs, Eigen::RowMajor> b_grad;
        Eigen::Matrix<Real, Eigen::Dynamic, Inputs, Eigen::RowMajor> inputs;
        Eigen::Matrix<Real, Eigen::Dynamic, Outputs, Eigen::RowMajor> outputs;
        Eigen::Matrix<Real, Eigen::Dynamic, Outputs, Eigen::RowMajor> delta;
        ActFunc actFunc;


        template<typename Derived>
        auto& forward(const Eigen::MatrixBase<Derived>& x) {
            inputs = x;
            return outputs = actFunc.evaluate((inputs * W).rowwise() + b);
        }

        template<typename Derived>
        auto backward(const Eigen::MatrixBase<Derived>& error) {
            delta = error.cwiseProduct(actFunc.derivative(outputs));

            W_grad = inputs.transpose() * delta;
            b_grad = delta.colwise().mean();

            return delta * W.transpose();
        }
};

}
