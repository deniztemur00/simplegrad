from simplegrad import Node, MLP
from sklearn import datasets
#from memory_profiler import profile

def print_layer_values(nodes):
    """Helper to extract values from Node objects"""
    return [node.data() for node in nodes]


def make_training_data():
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        random_state=42,  # for reproducibility
    )
    return X, y


def train_loop(mlp, X, y, lr=0.01, batch_size=16, epochs=100):
    n_batches = (len(X) + batch_size - 1) // batch_size  # Ceiling division

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(X), batch_size):
            batch_X = X[i : i + batch_size]
            batch_y = y[i : i + batch_size]
            current_batch_size = len(batch_X)  # Handle last batch

            batch_loss = 0.0
            mlp.zero_grad()  # Reset gradients before batch

            # Accumulate gradients over batch
            for x, y_true in zip(batch_X, batch_y):
                y_hat = mlp(x)[0]
                y_true = Node(y_true)
                loss = (y_hat - y_true) ** 2
                loss = loss * (1.0 / current_batch_size)  # Normalize loss
                batch_loss += loss.data()
                loss.backward()
                break
        

            mlp.step(lr)  # Update weights using accumulated gradients
            del loss, y_hat, y_true  # Clean up
            epoch_loss += batch_loss
        print(f"Epoch {epoch+1}, Average Loss: {epoch_loss/n_batches:.3f}")
        break

        # Average loss over all batches


def main():
    a = Node(1.2)
    model = MLP(10, [16, 1])
    print(a)


    """
    # X, y = make_training_data()
    # mlp = MLP(10, [32, 16, 1])
    # train_loop(mlp, X, y, lr=0.01, batch_size=16, epochs=10)

    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=42,  # for reproducibility
    )
    lr = 0.01
    batch_size = 100
    epochs = 1

    model = MLP(
        10, [2,1]
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
            model.zero_grad()  # Reset gradients before batch

            # Accumulate gradients over batch
            for x, y_true in zip(batch_X, batch_y):
                y_hat = model(x)[0]
                y_true = Node(y_true)
                loss = (y_hat - y_true) ** 2
                loss = loss * (1.0 / current_batch_size)  # Normalize loss
                batch_loss += loss.data()
                loss.backward()
                

            model.step(lr)  # Update weights using accumulated gradients
            #del loss, y_hat, y_true  # Clean up
            epoch_loss += batch_loss

        # Average loss over all batches
        print(f"Epoch {epoch+1}, Average Loss: {epoch_loss/n_batches:.3f}"),
        print(model)
"""

if __name__ == "__main__":
    main()
