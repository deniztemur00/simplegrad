#include "../include/mlp.h"

static int n_instances = 0;

MLP::MLP(int nin, std::vector<int> nouts) : n_params(0) {
    layers.reserve(nouts.size() + 1);
    layers.emplace_back(Layer(nin, nouts[0]));
    n_params += nin * nouts[0];
    for (size_t i = 1; i < nouts.size(); i++) {
        Layer l{nouts[i - 1], nouts[i]};
        layers.emplace_back(l);
        n_params += nouts[i - 1] * nouts[i];
    }
}

NodePtrVec MLP::operator()(NodePtrVec& x) {
    for (auto& layer : layers) {
        x = layer(x);
        // std::cout << "Layer output: " << x.size() << "\n";
    }

    return x;
}

NodePtrVec MLP::operator()(const pybind11::array_t<float>& x) {
    auto buf = x.unchecked<1>();
    NodePtrVec inputs;
    inputs.reserve(buf.shape(0));
    for (pybind11::ssize_t i = 0; i < buf.shape(0); i++) {
        inputs.emplace_back(std::make_shared<Node>(buf(i)));
    }

    return operator()(inputs);
}

NodePtrVec MLP::parameters() {
    NodePtrVec params;
    params.reserve(n_params + 1);

    for (auto& layer : layers) {
        for (auto& param : layer.parameters()) {
            params.emplace_back(param);
        }
    }

    return params;
}

std::string MLP::display_params() {
    std::stringstream ss;
    ss << "MLP(\n";
    for (auto& layer : layers) {
        ss << layer.display_params();
    }
    ss << "))\nTotal Parameters: " << n_params;
    return ss.str();
}

void MLP::step(float lr) {
    for (auto& layer : layers) {
        for (auto& neurons : layer) {  // implement begin
            for (auto& node : neurons) {
                std::cout << "Inside step function loop of MLP: " << ++n_instances << "\n";
                // std::cout << "Parameter: " << parameter->get_data() << "\n";
                // std::cout << "Gradient: " << parameter->get_grad() << "\n";

                auto out = node->get_data() - lr * node->get_grad();
                node->set_data(out);
                //std::cout << "Parameter: " << node->print() << "\n";
                
            }
        }
    }
}