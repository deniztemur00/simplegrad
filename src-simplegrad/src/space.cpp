// Space.cpp
#include "../include/space.h"

class Space::Impl {
   public:
    float data;
    float grad;
    std::string op;
    std::function<void()> backward;
    std::tuple<Space*, Space*> prev;
    Impl(float d)
        : data(d), grad(0.0), op(""), backward(nullptr) {
        prev = std::make_tuple(nullptr, nullptr);
    }
    ~Impl() {
    }

    float get_data() {
        return data;
    }
    float get_grad() {
        return grad;
    }

    const std::string& get_op() {
        return op;
    }

    void build_topo(Space* v, std::vector<Space*>& topo, std::set<Space*>& visited) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            auto [prev1, prev2] = v->pimpl->prev;
            if (prev1) build_topo(prev1, topo, visited);
            if (prev2) build_topo(prev2, topo, visited);
            topo.push_back(v);
        }
    }
};

// Constructors
Space::Space(float data)
    : pimpl(std::make_unique<Impl>(data)) {}

Space::~Space() = default;

Space::Space(Space&&) noexcept = default;
Space& Space::operator=(Space&&) noexcept = default;

// Accessors
const float Space::data() {
    return pimpl->get_data();
}
const float Space::grad() {
    return pimpl->get_grad();
}
const std::string& Space::op() {
    return pimpl->get_op();
}

// __repr__ method
std::string Space::print() const {
    std::stringstream ss;
    ss << "Space(data=" << pimpl->get_data()
       << ", grad=" << pimpl->get_grad();
    if (!pimpl->get_op().empty()) {
        ss << ", op='" << pimpl->get_op() << "'";
    }
    ss << ")";
    return ss.str();
}

// Operators
Space Space::operator+(const Space& other) const {
    Space out(this->pimpl->get_data() + other.pimpl->get_data());
    out.pimpl->op = "+";
    out.pimpl->prev = std::make_tuple(
        const_cast<Space*>(this),
        const_cast<Space*>(&other));

    out.pimpl->backward = [&out]() {
        auto [a, b] = out.pimpl->prev;
        a->pimpl->grad += out.pimpl->grad;
        b->pimpl->grad += out.pimpl->grad;
    };

    return out;
}

Space Space::operator+(float other) const {
    return *this + Space(other);
}

Space Space::operator*(const Space& other) const {
    Space out(this->pimpl->get_data() * other.pimpl->get_data());
    out.pimpl->op = "*";
    out.pimpl->prev = std::make_tuple(
        const_cast<Space*>(this),
        const_cast<Space*>(&other));

    out.pimpl->backward = [&out]() {
        auto [a, b] = out.pimpl->prev;
        a->pimpl->grad += b->pimpl->get_data() * out.pimpl->grad;
        b->pimpl->grad += a->pimpl->get_data() * out.pimpl->grad;
    };

    return out;
}

Space Space::operator*(float other) const {
    return *this * Space(other);
}

Space Space::pow(float other) const {
    Space out(std::pow(this->pimpl->get_data(), other));
    out.pimpl->op = "^";

    Space* base = const_cast<Space*>(this);
    out.pimpl->prev = std::make_tuple(base, nullptr);

    out.pimpl->backward = [&base, other, &out]() {
        // derivative of x^n is n * x^(n-1)
        base->pimpl->grad += other *
                             std::pow(base->pimpl->get_data(), other - 1) *
                             out.pimpl->grad;
    };

    return out;
}

Space Space::operator-() const {
    Space out(-this->pimpl->get_data());
    out.pimpl->op = "-";
    out.pimpl->prev = std::make_tuple(
        const_cast<Space*>(this),
        nullptr);

    out.pimpl->backward = [&out]() {
        auto [a, b] = out.pimpl->prev;
        a->pimpl->grad -= out.pimpl->grad;
    };

    return out;
}

Space Space::neg() const {
    return -(*this);
}

Space Space::sub(const Space& other) const {
    return *this + (-other);
}

Space Space::truediv(const Space& other) const {
    return *this * other.pow(-1);
}

Space Space::radd(const Space& other) const {
    return other + *this;
}

Space Space::rsub(const Space& other) const {
    return other + (-*this);
}

Space Space::rmul(const Space& other) const {
    return other * *this;
}

Space Space::rtruediv(const Space& other) const {
    return other * this->pow(-1);
}

// Functions
Space Space::relu() {
    Space out(std::max(0.0f, this->pimpl->get_data()));
    out.pimpl->op = "ReLU";
    out.pimpl->prev = std::make_tuple(
        const_cast<Space*>(this),
        nullptr);

    out.pimpl->backward = [&out]() {
        auto [a, b] = out.pimpl->prev;
        a->pimpl->grad += (out.pimpl->data > 0) * out.pimpl->grad;
    };

    return out;
}

void Space::backward() {
    // Build topological ordering
    std::vector<Space*> topo;
    std::set<Space*> visited;
    pimpl->build_topo(this, topo, visited);

    // Initialize gradient of root to 1.0
    this->pimpl->grad = 1.0;

    // Backpropagate in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->pimpl->backward();
    }
}