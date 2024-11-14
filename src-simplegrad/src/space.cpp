// Space.cpp
#include "../include/space.h"

class Space::Impl {
   private:
    float data;
    float grad;
    std::string op;
    void (*backward)();
    std::tuple<Space*, Space*> prev;

   public:
    Impl(float d, std::tuple<> c, const std::string& o)
        : data(d), grad(0.0), op(o), backward(nullptr) {
        std::cout<<"Inside Impl constructor\n";
        prev = std::make_tuple(nullptr, nullptr);
    }
};

Space::Space(float data, std::tuple<> children, std::string op = "")
    : pimpl(std::make_unique<Impl>(data, children, op)) {}

Space::~Space() = default;

Space::Space(Space&&) noexcept = default;
Space& Space::operator=(Space&&) noexcept = default;
