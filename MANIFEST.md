# SimpleGrad

SimpleGrad is a lightweight automatic differentiation library written in C++ with Python bindings.

### Compatible Operating Systems:
- Linux (x86_64 architecture only)

## Installation
```bash
pip install simplegrad
```

## Features

- Multi-layer perceptron (MLP) which can be used for regression and classification tasks
- Supports basic arithmetic operations
- Lightweight and easy to use
- Gradient computation
- Backpropagation
- Numpy compatibility


## Usage

Here's a quick example of how to use MLP in SimpleGrad:

```python
from simplegrad import MLP, Node
from sklearn import datasets

# Define the model
X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=42,  # for reproducibility
    )
lr = 0.01
batch_size = 16
epochs = 10

# Define the model
model = MLP(
    10, [16, 1]
)  # 2 input nodes, 2 hidden layers with arbitrary sizes, 1 output node

# Training data
n_batches = (len(X) + batch_size - 1) // batch_size  # Ceiling division

for epoch in range(epochs):
    epoch_loss = 0.0
    for i in range(0, len(X), batch_size):
        batch_X = X[i : i + batch_size]
        batch_y = y[i : i + batch_size]
        current_batch_size = len(batch_X)  # Handle last batch

        batch_loss = 0.0
        #model.zero_grad()  # gradients are automatically reset after step function

        # Accumulate gradients over batch
        for x, y_true in zip(batch_X, batch_y):
            y_hat = model(x)[0]
            y_true = Node(y_true)
            loss = (y_hat - y_true) ** 2
            loss = loss * (1.0 / current_batch_size)  # Normalize loss
            batch_loss += loss.data()
            loss.backward()

        model.step(lr)  # Update weights using accumulated gradients
        del loss, y_hat, y_true  # Clean up
        epoch_loss += batch_loss

    # Average loss over all batches
    print(f"Epoch {epoch+1}, Average Loss: {epoch_loss/n_batches:.3f}")
```



