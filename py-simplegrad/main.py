from simplegrad import Node, MLP
from sklearn import datasets
from memory_profiler import profile
import gc

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
    batch_size = 32
    epochs = 20

    model = MLP(
        10, [2,1]
    )  # 2 input nodes, 2 hidden layers with arbitrary sizes, 1 output node
    
    model.zero_grad()
    # Training data
    n_batches = (len(X) + batch_size - 1) // batch_size  # Ceiling division


    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(X), batch_size):
            batch_X = X[i : i + batch_size]
            batch_y = y[i : i + batch_size]
            current_batch_size = len(batch_X)

            batch_loss = 0.0
            model.zero_grad()

            for x, y_true in zip(batch_X, batch_y):
                y_hat = model([Node(feature) for feature in x])[0]
                y_true_node = Node(y_true)
                loss = (y_hat - y_true_node) ** 2
                loss = loss * (1.0 / current_batch_size)
                batch_loss += loss.data()
                loss.backward()
                #del loss, y_hat, y_true_node
                

            model.step(lr)
            epoch_loss += batch_loss

        print(f"Epoch {epoch+1}, Average Loss: {epoch_loss/n_batches:.3f}")

        


def main():
    train_loop()

    


if __name__ == "__main__":
    main()
