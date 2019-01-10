#pragma once

#include "definitions.hpp"
#include "activation.hpp"

namespace layer {

template<unsigned Inputs, unsigned Outputs, typename ActFunc>
class DenseLayer {
    public:
        Eigen::Matrix<Real, Inputs, Outputs, Eigen::RowMajor> W;
        Eigen::Matrix<Real, 1, Outputs, Eigen::RowMajor> b;
        Eigen::Matrix<Real, Eigen::Dynamic, Outputs, Eigen::RowMajor> a;
        Eigen::Matrix<Real, Eigen::Dynamic, Outputs, Eigen::RowMajor> delta;
        ActFunc actFunc;


        template<typename Derived>
        auto& forward(const Eigen::MatrixBase<Derived>& x) {
            return a = actFunc.evaluate((x * W).rowwise() + b);
        }

        template<typename Derived>
        auto backward(const Eigen::MatrixBase<Derived>& error) {
            delta = error.cwiseProduct(actFunc.derivative(a));
            return delta * W.transpose();
        }
};

}
