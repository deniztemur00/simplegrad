# SimpleGrad

SimpleGrad is a lightweight automatic differentiation library written in C++ with Python bindings.

## Prerequisites

- Python 3.10 or higher
- g++/gcc with C++17 support
- CMake 3.12 or higher
- pybind11

## Build & Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simplegrad.git
cd simplegrad
```
2. Build with makefile:
```bash
make build
```
## Features

- Easy-to-understand implementation
- Supports basic arithmetic operations
- Gradient computation


## Usage

Here's a quick example of how to use SimpleGrad:

```python
from simplegrad import Variable

# Create variables
x = Variable(2.0)
y = Variable(3.0)

# Perform operations
z = x * y + y

# Compute gradients
z.backward()

# Print gradients
print(x)  # Should print the node with updated gradient
print(y)  # Should print the node with updated gradient
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project was inspired by the [micrograd](https://github.com/karpathy/micrograd) library.
```