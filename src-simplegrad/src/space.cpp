// Space.cpp
#include "../include/space.h"

class Space::Impl {
   public:
    float data;
    float grad;
    std::string op;
    std::function<void()> backward;
    std::vector<Space> prev;

    Impl(float d, std::vector<Space> children = {}, std::string op = "")
        : data(d), grad(0.0), op(op), backward(nullptr), prev(std::move(children)) {}
    ~Impl() {
    }

    void build_topo(Space* v, std::vector<Space*>& topo, std::set<Space*>& visited) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (auto& child : v->pimpl->prev) {
                build_topo(&child, topo, visited);
            }
            topo.push_back(v);
        }
    }
};

// Constructors
Space::Space(float data, std::vector<Space> children = {}, std::string op = "")
    : pimpl(std::make_unique<Impl>(data, children, op)) {}


Space::~Space() = default;

// Accessors
float Space::data() const {
    return pimpl->data;
}
float Space::grad() const {
    return pimpl->grad;
}
const std::string& Space::op() const {
    return pimpl->op;
}

// __repr__ method
std::string Space::print() const {
    std::stringstream ss;
    ss << "Space(data=" << pimpl->data
       << ", grad=" << pimpl->grad;
    if (!pimpl->op.empty()) {
        ss << ", op='" << pimpl->op << "'";
    }
    ss << ")";
    return ss.str();
}

// Operators
Space Space::operator+(const Space& other) const {
    Space out(this->pimpl->data + other.pimpl->data,
              {*this, other},
              "+");

    auto* out_raw = out.pimpl.get();

    out.pimpl->backward = [out_raw] {
        auto self = out_raw->prev[0];
        auto other = out_raw->prev[1];

        self.pimpl->grad += out_raw->grad;
        other.pimpl->grad += out_raw->grad;

        std::cout << "Updated grads: first=" << self.pimpl->grad
                  << " second=" << other.pimpl->grad << std::endl;
    };

    return out;
}

Space Space::operator+(float other) const {
    return *this + Space(other);
}

Space Space::operator*(const Space& other) const {
    Space out(this->pimpl->data * other.pimpl->data,
              {*this, other},
              "*");

    // Capture values instead of references
    auto* out_raw = out.pimpl.get();
    out.pimpl->backward = [out_raw] {
        auto first = out_raw->prev[0];
        auto second = out_raw->prev[1];

        first.pimpl->grad += second.pimpl->data * out_raw->grad;
        second.pimpl->grad += first.pimpl->data * out_raw->grad;

        std::cout << "Updated grads: first=" << first.pimpl->grad
                  << " second=" << second.pimpl->grad << std::endl;
    };

    return out;
}

Space Space::operator*(float other) const {
    return *this * Space(other);
}

Space Space::pow(float other) const {
    Space out(std::pow(this->pimpl->data, other),
              {
                  *this,
              },
              "^");

    auto* out_raw = out.pimpl.get();
    out.pimpl->backward = [out_raw, other]() {
        // derivative of x^n is n * x^(n-1)
        auto base = out_raw->prev[0];
        base.pimpl->grad += other * std::pow(base.pimpl->data, other - 1) * out_raw->grad;

        std::cout << "Updated grad: " << base.pimpl->grad << std::endl;
    };

    return out;
}

Space Space::operator-() const {
    Space out(-this->pimpl->data,
              {
                  *this,
              },
              "-");

    auto* out_raw = out.pimpl.get();
    out.pimpl->backward = [out_raw]() {
        auto base = out_raw->prev[0];
        base.pimpl->grad -= out_raw->grad;
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
    other.pimpl->data += 1e-9;  // Avoid division by zero
    return *this * other.pow(-1);
}

Space Space::radd(const Space& other) const {
    return other + *this;
}

// Space Space::radd(float other) const {
//     return  Space(other) + *this;
// }

Space Space::rsub(const Space& other) const {
    return other + (-*this);
}

Space Space::rmul(const Space& other) const {
    return other * *this;
}

Space Space::rtruediv(const Space& other) const {
    other.pimpl->data += 1e-9;
    return other * this->pow(-1);
}

// Functions
Space Space::relu() {
    Space out(std::max(0.0f, this->pimpl->data),
              {
                  *this,
              },
              "ReLU");

    auto* out_raw = out.pimpl.get();

    out.pimpl->backward = [out_raw] {
        // Get first (and only) previous node
        auto& self = out_raw->prev[0];
        // Same logic as Python version
        self.pimpl->grad += (out_raw->data > 0) * out_raw->grad;
    };

    return out;
}

void Space::backward() {
    if (!pimpl) {
        std::cerr << "Null pimpl in backward()" << std::endl;
        return;
    }

    // Build topological ordering
    std::vector<Space*> topo;
    std::set<Space*> visited;
    pimpl->build_topo(this, topo, visited);

    // Initialize gradient of root to 1.0
    pimpl->grad = 1.0;

    // Backpropagate in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->pimpl->backward) {
            (*it)->pimpl->backward();
        }
    }
}
