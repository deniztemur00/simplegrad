#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/net.h"
#include "include/node.h"
#include "include/mlp.h"

namespace py = pybind11;

PYBIND11_MODULE(_simplegrad, m) {
    m.doc() = "Automatic differentation module written in C++";
    py::class_<Node, NodePtr>(m, "Node")
        .def(py::init<float>())
        .def("__repr__", &Node::print)
        .def("backward", &Node::backward)
        .def("relu", &Node::relu)
        .def("data", &Node::get_data)
        .def("clear_prev", &Node::clear_prev)
        .def("grad", &Node::get_grad)
        .def("prev", &Node::get_prev)
        .def("op", &Node::get_op)
        .def(py::self + py::self)
        .def(py::self + float())
        .def(float() + py::self)
        .def(py::self * py::self)
        .def(py::self * float())
        .def(float() * py::self)
        .def("__rmul__", &Node::rmul)
        .def("__radd__", &Node::radd)
        .def("__neg__", &Node::operator-)
        .def("__sub__", &Node::sub)
        .def("__rsub__", &Node::rsub)
        .def("__pow__", static_cast<NodePtr (Node::*)(float) const>(&Node::pow))
        .def("__pow__", static_cast<NodePtr (Node::*)(const Node&) const>(&Node::pow))
        .def("__truediv__", &Node::rdiv)
        .def("__rtruediv__", &Node::rtruediv);

    // Bind Module base class
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("zero_grad", &Module::zero_grad)
        .def("parameters", &Module::parameters);

    // Bind Neuron class
    py::class_<Neuron, Module, std::shared_ptr<Neuron>>(m, "Neuron")
        .def(py::init<int, bool>(),
             py::arg("nin"),
             py::arg("nonlin") = true)
        .def("__call__", &Neuron::operator())
        .def("parameters", &Neuron::parameters)
        .def("clear_weights", &Neuron::clear_weights)
        .def("__repr__", &Neuron::display_params);

    // Bind Layer class
    py::class_<Layer, Module, std::shared_ptr<Layer>>(m, "Layer")
        .def(py::init<int, int>())
        .def("__call__", &Layer::operator())
        .def("parameters", &Layer::parameters)
        .def("clear_neurons", &Layer::clear_neurons)
        .def("size", &Layer::size)
        .def("input_size", &Layer::input_size)
        .def("__repr__", &Layer::display_params);

    // Bind MLP class
    py::class_<MLP, Module, std::shared_ptr<MLP>>(m, "MLP")
        .def(py::init<int, std::vector<int>>())
        .def("__call__", static_cast<NodePtrVec& (MLP::*)(NodePtrVec&)>(&MLP::operator()))
        .def("__call__", static_cast<NodePtrVec (MLP::*)(const py::array_t<float>&)>(&MLP::operator()))
        .def("step", &MLP::step,
             py::arg("lr"),
             "Performs a step of gradient descent.")
        .def("parameters", &MLP::parameters)
        .def("__repr__", &MLP::display_params);
}