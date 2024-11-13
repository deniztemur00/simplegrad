#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

PYBIND11_MODULE(simplegrad, m) {
    m.doc() = "Math operations module"; // Module docstring

    m.def("add",
         &add,
         py::arg("a"), py::arg("b"),
         R"pbdoc(
         Add two integers.
         
         Parameters:
         a (int): The first integer to add.
         b (int): The second integer to add.
         
         Returns:
         int: The sum of a and b.
         )pbdoc");

    m.def("subtract",
         &subtract,
         py::arg("a"), py::arg("b"),
         R"pbdoc(
         Subtract two integers.
         
         Parameters:
         a (int): The integer to subtract from.
         b (int): The integer to subtract.
         
         Returns:
         int: The difference of a and b.
         )pbdoc");
}