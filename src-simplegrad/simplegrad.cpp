#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/node.h"

namespace py = pybind11;

PYBIND11_MODULE(simplegrad, m) {
    m.doc() = "Automatic differentation module written in C++";  // Module docstring
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<float>())
        .def("__repr__", &Node::print)
        .def("backward", &Node::backward)
        .def("data", &Node::get_data)
        .def("grad", &Node::get_grad)
        .def("prev", &Node::get_prev)
        .def("op", &Node::get_op)
        .def(py::self + py::self)
        .def(py::self + float())
        .def(float() + py::self);
    /*
    .def(py::self * py::self)
    .def("__pow__", &Node::pow)
    .def("__neg__", &Node::neg)
    .def("__sub__", &Node::sub)
    .def("__truediv__", &Node::rdiv)
    .def("__radd__", &Node::radd)
    .def("__rsub__", &Node::rsub)
    .def("__rmul__", &Node::rmul)
    .def("__rtruediv__", &Node::rtruediv)
    .def("relu", &Node::relu)
    */
}