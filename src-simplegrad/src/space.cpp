// Space.cpp
#include "../include/space.h"

class Space::Impl {
   public:
    float data;
    float grad;
    std::string op;
    std::function<void()> backward;
    std::set<Space*> prev;
    Impl(float d)
        : data(d), grad(0.0), op(""), backward(nullptr) {}
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
            // Iterate through all previous nodes in prev set
            for (Space* child : v->pimpl->prev) {
                build_topo(child, topo, visited);
            }
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
Space::Space(const Space& other) : pimpl(new Impl(*other.pimpl)) {}

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

    Space* a = const_cast<Space*>(this);
    Space* b = const_cast<Space*>(&other);
    out.pimpl->op = "+";
    out.pimpl->prev.insert(a);
    out.pimpl->prev.insert(b);

    // Store raw pointer to implementation
    auto* out_raw = out.pimpl.get();

    out.pimpl->backward = [a, b, out_raw] {
        std::cout << "Add backward: out.grad = " << out_raw->grad << std::endl;
        // Gradients flow equally to both inputs
        a->pimpl->grad += out_raw->grad;
        b->pimpl->grad += out_raw->grad;
        std::cout << "Updated grads: a=" << a->pimpl->grad << " b=" << b->pimpl->grad << std::endl;
    };

    return out;
}

Space Space::operator+(float other) const {
    return *this + Space(other);
}

Space Space::operator*(const Space& other) const {
    Space out(this->pimpl->get_data() * other.pimpl->get_data());

    // Store raw pointers
    Space* a = const_cast<Space*>(this);
    Space* b = const_cast<Space*>(&other);

    out.pimpl->prev.insert(a);
    out.pimpl->prev.insert(b);

    // Capture values instead of references
    auto* out_raw = out.pimpl.get();
    out.pimpl->backward = [a, b, out_raw] {  // Capture by value
        std::cout << "Mul backward: out.grad = " << out_raw->grad << std::endl;
        // Match Python's gradient computation exactly
        a->pimpl->grad += b->pimpl->get_data() * out_raw->grad;  // self.grad += other.data * out.grad
        b->pimpl->grad += a->pimpl->get_data() * out_raw->grad;  // other.grad += self.data * out.grad
        std::cout << "Updated grads: a=" << a->pimpl->grad << " b=" << b->pimpl->grad << std::endl;
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
    out.pimpl->prev.insert(base);

    auto* out_raw = out.pimpl.get();
    out.pimpl->backward = [base, other, out_raw]() {
        // derivative of x^n is n * x^(n-1)
        base->pimpl->grad += other *
                             std::pow(base->pimpl->get_data(), other - 1) *
                             out_raw->grad;
    };

    return out;
}

Space Space::operator-() const {
    Space out(-this->pimpl->get_data());
    out.pimpl->op = "-";
    Space* base = const_cast<Space*>(this);
    out.pimpl->prev.insert(base);

    auto* out_raw = out.pimpl.get();
    out.pimpl->backward = [base, out_raw]() {
        base->pimpl->grad -= out_raw->grad;
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
    Space out(std::max(0.0f, this->pimpl->get_data()));
    out.pimpl->op = "ReLU";
    Space* base = const_cast<Space*>(this);
    out.pimpl->prev.insert(base);

    out.pimpl->backward = [base, &out]() {
        base->pimpl->grad += (out.pimpl->data > 0) * out.pimpl->grad;
    };

    return out;
}

void Space::backward() {
    std::vector<Space*> topo;
    std::set<Space*> visited;

    if (!pimpl) {
        std::cout << "Null pimpl in backward()" << std::endl;
        return;
    }

    pimpl->build_topo(this, topo, visited);

    // Initialize gradient of root to 1.0
    pimpl->grad = 1.0;

    std::cout << "Topo size: " << topo.size() << std::endl;

    // Iterate through nodes in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Space* node = *it;
        if (!node || !node->pimpl) {
            std::cout << "Null node or pimpl encountered" << std::endl;
            continue;
        }

        std::cout << "Processing node: " << node->print()
                  << " at " << node
                  << " grad: " << node->pimpl->grad << std::endl;

        if (node->pimpl->backward) {
            try {
                node->pimpl->backward();
            } catch (const std::exception& e) {
                std::cout << "Exception in backward: " << e.what() << std::endl;
            }
        }
        std::cout << "After backward: " << node->print() << std::endl;
    }
}
