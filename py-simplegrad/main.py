from simplegrad import Node, MLP
from sklearn import datasets


def print_layer_values(nodes):
    """Helper to extract values from Node objects"""
    return [node.data() for node in nodes]


def train_loop():
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
        10, [12, 1]
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
            #model.zero_grad()  

            # Accumulate gradients over batch
            for x, y_true in zip(batch_X, batch_y):
                y_hat = model(x)[0]
                y_true = Node(y_true)
                loss = (y_hat - y_true) ** 2
                loss = loss * (1.0 / current_batch_size)  # Normalize loss
                batch_loss += loss.data()
                loss.backward()

            model.step(lr)  # Update weights using accumulated gradients
            epoch_loss += batch_loss

        # Average loss over all batches
        print(f"Epoch {epoch+1}, Average Loss: {epoch_loss/n_batches:.3f}")


def test_multiplication_backward():
    a = Node(5.0)
    b = Node(3.0)
    c = a * b
    c.backward()
    print(a.grad())
    print(b.grad())
    assert abs(a.grad() - 3.0) < 1e-6
    assert abs(b.grad() - 5.0) < 1e-6
    c = c * a
    c.backward()
    print(c)


def main():
    train_loop()


if __name__ == "__main__":
    main()
