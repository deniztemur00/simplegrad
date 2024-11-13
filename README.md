# SimpleGrad

SimpleGrad is a lightweight automatic differentiation library for educational purposes.

## Features

- Easy-to-understand implementation
- Supports basic arithmetic operations
- Gradient computation

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/simplegrad.git
cd simplegrad
```

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
print(x.grad)  # Should print the gradient of z with respect to x
print(y.grad)  # Should print the gradient of z with respect to y
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.