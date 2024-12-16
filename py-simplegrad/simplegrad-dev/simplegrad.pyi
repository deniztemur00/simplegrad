"""Automatic differentiation library for basic arithmetic operations"""
from typing import List, Union


class Node:
    def __init__(self, value: float) -> None:
        """
        Initializes a new node with the given value.

        Parameters:
            value (float): The numeric value to store in the node.
        """
        ...

    def data(self) -> float:
        """
        Retrieves the data stored in the node.

        Returns:
            float: The data value of the node.
        """
        ...

    def grad(self) -> float:
        """
        Retrieves the gradient associated with the node.

        Returns:
            float: The gradient value of the node.
        """
        ...

    def backward(self) -> None:
        """
        Performs backpropagation to compute gradients of all nodes in the computational graph.
        """
        ...

    def relu(self) -> "Node":
        """
        Applies the ReLU activation function to the node.

        Returns:
            Node: A new node after applying the ReLU function.
        """
        ...

    def __add__(self, other: Union["Node", float]) -> "Node":
        """
        Adds this node with another node or scalar.

        Parameters:
            other (Union[Node, float]): The node or scalar to add.

        Returns:
            Node: The result of the addition.
        """
        ...

    def __radd__(self, other: Union["Node", float]) -> "Node":
        """
        Adds another node or scalar to this node (reversed operands).

        Parameters:
            other (Union[Node, float]): The node or scalar to add.

        Returns:
            Node: The result of the addition.
        """
        ...

    def __mul__(self, other: Union["Node", float]) -> "Node":
        """
        Multiplies this node with another node or scalar.

        Parameters:
            other (Union[Node, float]): The node or scalar to multiply.

        Returns:
            Node: The result of the multiplication.
        """
        ...

    def __rmul__(self, other: Union["Node", float]) -> "Node":
        """
        Multiplies another node or scalar with this node (reversed operands).

        Parameters:
            other (Union[Node, float]): The node or scalar to multiply.

        Returns:
            Node: The result of the multiplication.
        """
        ...

    def __sub__(self, other: Union["Node", float]) -> "Node":
        """
        Subtracts another node or scalar from this node.

        Parameters:
            other (Union[Node, float]): The node or scalar to subtract.

        Returns:
            Node: The result of the subtraction.
        """
        ...

    def __rsub__(self, other: Union["Node", float]) -> "Node":
        """
        Subtracts this node from another node or scalar (reversed operands).

        Parameters:
            other (Union[Node, float]): The node or scalar to subtract from.

        Returns:
            Node: The result of the subtraction.
        """
        ...

    def __truediv__(self, other: Union["Node", float]) -> "Node":
        """
        Divides this node by another node or scalar.

        Parameters:
            other (Union[Node, float]): The node or scalar to divide by.

        Returns:
            Node: The result of the division.
        """
        ...

    def __rtruediv__(self, other: Union["Node", float]) -> "Node":
        """
        Divides another node or scalar by this node (reversed operands).

        Parameters:
            other (Union[Node, float]): The node or scalar to divide.

        Returns:
            Node: The result of the division.
        """
        ...

class Module:
    def zero_grad(self) -> None:
        """
        Resets the gradients of all parameters to zero.
        """
        ...

    def parameters(self) -> List[Node]:
        """
        Retrieves all parameter nodes within the module.

        Returns:
            List[Node]: A list of parameter nodes.
        """
        ...

class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True) -> None:
        """
        Initializes a neuron with a specified number of inputs.

        Parameters:
            nin (int): Number of input features.
            nonlin (bool): If True, applies a non-linear activation function (ReLU).
        """
        ...

    def __call__(self, x: List[Node]) -> Node:
        """
        Performs the forward pass of the neuron.

        Parameters:
            x (List[Node]): A list of input nodes.

        Returns:
            Node: The output node after applying weights, bias, and activation.
        """
        ...

    def parameters(self) -> List[Node]:
        """
        Retrieves the parameters (weights and bias) of the neuron.

        Returns:
            List[Node]: A list containing the weight nodes and bias node.
        """
        ...

class Layer(Module):
    def __init__(self, nin: int, nout: int) -> None:
        """
        Initializes a layer composed of multiple neurons.

        Parameters:
            nin (int): Number of input features.
            nout (int): Number of neurons in the layer.
        """
        ...

    def __call__(self, x: List[Node]) -> List[Node]:
        """
        Performs the forward pass of the layer.

        Parameters:
            x (List[Node]): A list of input nodes.

        Returns:
            List[Node]: A list of output nodes from each neuron in the layer.
        """
        ...

    def parameters(self) -> List[Node]:
        """
        Retrieves the parameters from all neurons in the layer.

        Returns:
            List[Node]: A list of parameter nodes from all neurons.
        """
        ...

class MLP(Module):
    def __init__(self, nin: int, nouts: List[int]) -> None:
        """
        Initializes a Multi-Layer Perceptron (MLP) model.

        Parameters:
            nin (int): Number of input features.
            nouts (List[int]): A list specifying the number of neurons in each layer.
        """
        ...

    def __call__(self, x: Union[List[Node], List[float]]) -> List[Node]:
        """
        Performs the forward pass of the MLP.

        Parameters:
            x (Union[List[Node], List[float]]): Input data as a list of nodes or raw values.

        Returns:
            List[Node]: Output nodes after passing through the network.
        """
        ...

    def parameters(self) -> List[Node]:
        """
        Retrieves the parameters from all layers of the MLP.

        Returns:
            List[Node]: A list of parameter nodes from all layers.
        """
        ...

    def step(self, lr: float) -> None:
        """
        Updates the parameters using gradient descent.

        Parameters:
            lr (float): The learning rate for parameter updates.
        """
        ...

