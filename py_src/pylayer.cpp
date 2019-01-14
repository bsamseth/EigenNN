#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "layer.hpp"

namespace py = pybind11;

void init_layer(py::module& main) {
    auto m = main.def_submodule("layer");
    m.doc() = R"pbdoc(
        Network Layers
        -----------------------
        .. currentmodule:: layer
        .. autosummary::
           :toctree: _generate
           DenseLayer
    )pbdoc";

    py::class_<layer::DenseLayer>(m, "DenseLayer")
        .def(py::init<int, int, const activation::ActivationFunction&>(),
                py::arg("inputs"), py::arg("outputs"), py::arg("activation") = activation::identity)
        .def("evaluate", &layer::DenseLayer::forward);
}
