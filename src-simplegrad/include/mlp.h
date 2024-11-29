#pragma once
#include "../include/net.h"

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
    void step(float lr);
};