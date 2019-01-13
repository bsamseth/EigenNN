#pragma once
#include <vector>
#include "definitions.hpp"
#include "activation.hpp"
#include "layer.hpp"

class Dnn {
    protected:

        std::vector<layer::DenseLayer> layers;

    public:
        /**
         * Calculate derivative of output w.r.t. each component of the DNN.
         *
         * This is standard backprop. with the alteration of not propagating
         * the derivative of a cost function, but rather that of the output it self.
         * The backprop. algorithm can be used by letting the output be the cost,
         * and so the cost_gradient is simply all 1's.
         */
        void backward();

    public:

        void addLayer(layer::DenseLayer layer);
        const Matrix& evaluate(const MatrixRef& x);
};

