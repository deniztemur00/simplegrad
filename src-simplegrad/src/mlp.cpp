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

NodePtrVec& MLP::parameters() {
    static NodePtrVec params;
    params.clear();
    params.reserve(n_params + 1);

    for (auto& layer : layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
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
    for (auto& param : parameters()) {
        float grad = param->get_grad();
        float old_data = param->get_data();
        float new_data = old_data - lr * grad;

        //std::cout << "Parameter update:\n";
        //std::cout << "  grad: " << grad << "\n";
        //std::cout << "  lr * grad: " << (lr * grad) << "\n";
        //std::cout << "  old value: " << old_data << "\n";
        //std::cout << "  new value: " << new_data << "\n";
        //std::cout << "-------------------\n";

        param->set_data(new_data);
        param->set_grad(0.0);
        param->clear_prev();
    }
}