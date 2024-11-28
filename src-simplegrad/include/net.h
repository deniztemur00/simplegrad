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
    virtual NodePtrVec parameters() = 0;
};

class Neuron : public Module {
   private:
    NodePtrVec weights;
    NodePtr bias;
    bool nonlin;

   public:
    Neuron(int nin, bool nonlin = true);
    NodePtr operator()(NodePtrVec& x);
    NodePtrVec parameters() override;
    std::string display_params();
};

class Layer : public Module {
   private:
    std::vector<Neuron> neurons;
    int n_params;

   public:
    Layer(int nin, int nouts);
    NodePtrVec operator()(NodePtrVec& x);
    NodePtrVec parameters() override;
    std::string display_params();
};

class MLP : public Module {
   private:
    std::vector<Layer> layers;
    int n_params;

   public:
    MLP(int nin, std::vector<int> nouts);
    NodePtrVec operator()(NodePtrVec& x);
    NodePtrVec operator()(const pybind11::array_t<float>& x);  // numpy support
    NodePtrVec parameters() override;
    std::string display_params();
};