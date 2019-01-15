#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_activation(py::module&);
void init_layer(py::module&);
void init_dnn(py::module&);

PYBIND11_MODULE(EigenNN, m) {
    m.doc() = R"pbdoc(
        EigenNN - Neural Networks for Quantum Variational Monte Carlo
        -----------------------
    )pbdoc";

    init_activation(m);
    init_layer(m);
    init_dnn(m);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
