#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "include/space.h"

namespace py = pybind11;

PYBIND11_MODULE(simplegrad, m) {
    m.doc() = "Auto gradient module";  // Module docstring
    py::class_<Space>(m, "Space")
        .def(py::init<float>())
        .def(py::self + py::self)
        .def(py::self * py::self)
        .def("__pow__", &Space::pow)
        .def("__neg__", &Space::neg)
        .def("__sub__", &Space::sub)
        .def("__truediv__", &Space::truediv)
        .def("__radd__", &Space::radd)
        .def("__rsub__", &Space::rsub)
        .def("__rmul__", &Space::rmul)
        .def("__rtruediv__", &Space::rtruediv)
        .def("__repr__", &Space::print)
        .def("relu", &Space::relu)
        .def("backward", &Space::backward)
        .def("data", (float(Space::*)()) & Space::data)
        .def("grad", (float(Space::*)()) & Space::grad)
        .def("op", (const std::string& (Space::*)()) & Space::op);
}