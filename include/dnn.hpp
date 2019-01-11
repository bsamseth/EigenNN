#pragma once

#include "definitions.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "helpers.hpp"

template<typename... Layers>
class Dnn {
    public:
        std::tuple<Layers...> layers;
        using Depth = std::tuple_size<decltype(layers)>;

        template<typename Derived>
        auto& forward(const Eigen::MatrixBase<Derived>& input) {
            Matrix x = input;
            for_each(layers,
                    [&](auto& layer)
                    {
                        x = layer.forward(x);
                    }
            );
            return std::get<Depth::value - 1>(layers).outputs;
        }

        /**
         * Calculate derivative of output w.r.t. each component of the DNN.
         *
         * This is standard backprop. with the alteration of not propagating
         * the derivative of a cost function, but rather that of the output it self.
         * The backprop. algorithm can be used by letting the output be the cost,
         * and so the cost_gradient is simply all 1's.
         */
        template<typename Derived>
        void backward() {
            const auto& output = std::get<Depth::value - 1>(layers).outputs;
            Matrix y = Matrix::Ones(output.rows(), output.cols());
            for_each_reverse(layers,
                    [&](auto& layer)
                    {
                        y = layer.backward(y);
                    }
            );
        }

};

