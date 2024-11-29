#include "../include/net.h"

// Virtual method

void Module::zero_grad() {
    for (auto& param : parameters()) {
        param->set_grad(0.0);
    }
}

// Neuron

Neuron::Neuron(int nin, bool nonlin) : nonlin(nonlin),
                                       bias(std::make_shared<Node>(0.0)) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Xavier initialization
    float limit = sqrt(6.0f / (nin + 1));
    std::uniform_real_distribution<float> dist(-limit, limit);

    weights.reserve(nin);

    for (int i = 0; i < nin; i++) {
        weights.emplace_back(std::make_shared<Node>(dist(gen)));
    }
}

NodePtr Neuron::operator()(NodePtrVec& x) {
    NodePtr out = std::make_shared<Node>(0.0f);

    for (size_t i = 0; i < x.size(); i++) {
        out = *out + *(*x[i] * *weights[i]);
    }
    out = *out + *bias;
    if (nonlin) {
        out = out->relu();
    }
    // std::cout << "Neuron output: " << out->get_data() << "\n";

    return out;
}

NodePtrVec Neuron::parameters() {
    NodePtrVec params;
    params.reserve(weights.size() + 1);

    for (const auto& weight : weights) {
        params.emplace_back(weight);
    }
    params.emplace_back(bias);
    return params;
}

std::string Neuron::display_params() {
    std::stringstream ss;
    ss << "Neuron(";
    for (const auto& w : weights) {
        ss << w->get_data() << ", ";
    }
    ss << "b = " << bias->get_data() << ")\n";
    return ss.str();
}

// Layer

Layer::Layer(int nin, int nouts) : n_params((nin + 1) * nouts) {
    neurons.reserve(nouts + 1);

    for (int i = 0; i < nouts; i++) {
        Neuron n{nin};
        neurons.emplace_back(n);
    }
}

NodePtrVec Layer::operator()(NodePtrVec& x) {
    NodePtrVec out;
    out.reserve(neurons.size() + 1);

    for (auto& neuron : neurons) {
        out.emplace_back(neuron(x));  // Use neuron's operator() to compute output
    }

    return out;
}

NodePtrVec Layer::parameters() {
    NodePtrVec params;
    params.reserve(n_params + 1);

    for (auto& neuron : neurons) {
        for (auto& param : neuron.parameters()) {
            params.emplace_back(param);
        }
    }

    return params;
}

std::string Layer::display_params() {
    std::stringstream ss;
    ss << "Layer(\n";
    for (auto& neuron : neurons) {
        ss << "  " << neuron.display_params();
    }
    // ss << "]";
    return ss.str();
}

