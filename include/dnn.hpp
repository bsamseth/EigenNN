#pragma once
#include <vector>
#include "definitions.hpp"
#include "activation.hpp"
#include "layer.hpp"

class Dnn {
    protected:

        unsigned paramCount = 0;
        std::vector<layer::DenseLayer> layers;
        Vector paramGradient;


        /**
         * Forward propagation of input (evaluation of network).
         */
        void forward(const MatrixRef& x);

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
        const Vector& parameterGradient(const MatrixRef& x);
};

