from simplegrad import Node,Neuron,Layer, MLP




def main():
    # Create a simple MLP model
    mlp = MLP(3, [2,1,4])
    print(mlp)




if __name__ == "__main__":
    main()
