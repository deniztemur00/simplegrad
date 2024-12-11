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

NodePtr& Neuron::operator()(NodePtrVec& x) {
    static NodePtr activation;
    
    // Initialize sum with bias
    NodePtr sum = std::make_shared<Node>(*bias);
    
    // Compute weighted sum
    for (size_t i = 0; i < weights.size(); i++) {
        sum = *sum + *(*weights[i] * *x[i]);
    }

    if (nonlin) {
        activation = sum->relu(); 
    }

    return activation;
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

NodePtrVec& Layer::operator()(NodePtrVec& x) {
    static NodePtrVec out;
    out.clear();
    out.reserve(neurons.size());

    for (auto& neuron : neurons) {
        auto neuron_out = neuron(x);
        out.emplace_back(neuron_out);
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
