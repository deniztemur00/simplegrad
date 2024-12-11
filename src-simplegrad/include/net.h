#pragma once
#include <pybind11/numpy.h>  // MLP numpy support

#include <iostream>
#include <random>
#include <vector>

#include "node.h"

using NodePtrVec = std::vector<NodePtr>;

class Module {
   public:
    void zero_grad();
    virtual NodePtrVec& parameters() = 0;
};

class Neuron : public Module {
    private:
    NodePtrVec weights;
    NodePtr bias;
    bool nonlin;

   public:
    Neuron(int nin, bool nonlin = true);
    auto begin() { return weights.begin(); }
    auto end() { return weights.end(); }
    size_t size() const { return weights.size(); }
    NodePtr& operator()(NodePtrVec& x);
    NodePtrVec& parameters() override;
    std::string display_params();
    void clear_weights();
};

class Layer : public Module {
   private:
    std::vector<Neuron> neurons;
    int n_params;

   public:
    auto begin() { return neurons.begin(); }
    auto end() { return neurons.end(); }
    size_t input_size(size_t idx) const { return neurons[idx].size(); }
    size_t size() const { return neurons.size(); }
    Layer(int nin, int nouts);
    NodePtrVec& operator()(NodePtrVec& x);
    NodePtrVec& parameters() override;
    std::string display_params();
    void clear_neurons();
};
