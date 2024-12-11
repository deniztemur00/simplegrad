#include "../include/net.h"

// Virtual method

void Module::zero_grad() {
    for (auto& param : parameters()) {
        param->set_grad(0.0);
        param->clear_prev();
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

    return out;
}

NodePtrVec& Neuron::parameters() {
    static NodePtrVec params;
    params.clear();
    params.reserve(weights.size() + 1);

    for (auto& w : weights) {
        params.push_back(w);
    }
    params.push_back(bias);

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

void Neuron::clear_weights() {
    float bias_value = bias->get_data();
    std::vector<float> weight_values;
    weight_values.reserve(weights.size());
    for (const auto& w : weights) {
        weight_values.push_back(w->get_data());
    }

    bias.reset();
    for (auto& w : weights) {
        w.reset();
    }
    weights.clear();
    std::vector<NodePtr>().swap(weights);

    weights.reserve(weight_values.size());
    bias = std::make_shared<Node>(bias_value);
    for (float w_val : weight_values) {
        weights.push_back(std::make_shared<Node>(w_val));
    }
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
        out.emplace_back(neuron(x)); 
    }

    return out;
}

NodePtrVec& Layer::parameters() {
    static NodePtrVec params;
    params.clear();
    params.reserve(n_params + 1);

    for (auto& neuron : neurons) {
        auto neuron_params = neuron.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
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

void Layer::clear_neurons() {
    for (auto& neuron : neurons) {
        neuron.clear_weights();
    }
}
