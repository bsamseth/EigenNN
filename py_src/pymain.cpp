#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

void init_activation(py::module&);
void init_layer(py::module&);

PYBIND11_MODULE(qflow, m) {
    m.doc() = R"pbdoc(
        QFlow - Quantum Variational Monte Carlo Framework
        -----------------------
        .. currentmodule:: qflow
        .. autosummary::
           :toctree: _generate
           activation
           layer
           Dnn
    )pbdoc";

    init_activation(m);
    init_layer(m);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
