#include "../include/node.h"


// Constructor
Node::Node(float data, std::unordered_set<NodePtr> prev, const std::string& op)
    : data(data), _prev(std::move(prev)), _op(op) {
    this->_backward = [this] {
        for (const auto& child : this->_prev) {
            child->_backward();
        }
    };
}

Node::~Node() = default;

float Node::get_data() const {
    return data;
}

float Node::get_grad() const {
    return grad;
}

const std::string& Node::get_op() const {
    return _op;
}

const std::unordered_set<NodePtr>& Node::get_prev() const {
    return _prev;
}

void Node::set_data(float data) {
    this->data = data;
}

void Node::set_grad(float grad) const {
    this->grad = grad;
}

std::string Node::print() const {
    std::stringstream ss;
    ss << "Node(data=" << data
       << ", grad=" << grad
       << ", op='" << _op << "')";
    return ss.str();
}

// Operators

NodePtr Node::operator+(const Node& other) const {
    auto result = std::make_shared<Node>(
        this->data + other.data,
        std::unordered_set<NodePtr>{
            std::const_pointer_cast<Node>(shared_from_this()),
            std::const_pointer_cast<Node>(other.shared_from_this())},
        "+");

    result->_backward = [this, &other, result]() {
        grad += result->grad;
        other.grad += result->grad;
    };
    return result;
}

NodePtr Node::radd(const Node& other) const {
    return other + *this;
}

NodePtr Node::operator+(float other) const {
    return *this + *std::make_shared<Node>(other);
}

NodePtr operator+(float lhs, const Node& rhs) {
    return rhs + lhs;
}

NodePtr Node::operator*(const Node& other) const {
    auto result = std::make_shared<Node>(
        this->data * other.data,
        std::unordered_set<NodePtr>{
            std::const_pointer_cast<Node>(shared_from_this()),
            std::const_pointer_cast<Node>(other.shared_from_this())},
        "*");

    result->_backward = [this, &other, result]() {
        grad += other.data * result->grad;
        other.grad += this->data * result->grad;
    };
    return result;
}

NodePtr Node::rmul(const Node& other) const {
    return *std::make_shared<Node>(other) * *this;
}

NodePtr Node::operator*(float other) const {
    return *this * *std::make_shared<Node>(other);
}

NodePtr operator*(float lhs, const Node& rhs) {
    return rhs * lhs;
}

NodePtr Node::operator-() const {
    auto result = std::make_shared<Node>(
        -this->data,
        std::unordered_set<NodePtr>{
            std::const_pointer_cast<Node>(shared_from_this())},
        "neg");

    result->_backward = [this, result]() {
        grad -= result->grad;
    };
    return result;
}

NodePtr Node::sub(const Node& other) const {
    return *this + *(-other);
}

NodePtr Node::rsub(const Node& other) const {
    return other + *(-*this);
}

NodePtr Node::pow(const Node& other) const {
    auto result = std::make_shared<Node>(
        std::pow(this->data, other.data),
        std::unordered_set<NodePtr>{
            std::const_pointer_cast<Node>(shared_from_this()),
            std::const_pointer_cast<Node>(other.shared_from_this())},
        "pow");

    result->_backward = [this, &other, result]() {
        grad += other.data * std::pow(this->data, other.data - 1) * result->grad;
        other.grad += std::pow(this->data, other.data) * std::log(this->data) * result->grad;
    };

    return result;
}

NodePtr Node::pow(float exponent) const {
    auto result = std::make_shared<Node>(
        std::pow(this->data + EPSILON, exponent),
        std::unordered_set<NodePtr>{
            std::const_pointer_cast<Node>(shared_from_this())},
        "pow");

    result->_backward = [this, result, exponent]() {
        grad += exponent * std::pow(this->data + EPSILON, exponent - 1) * result->grad;
    };

    return result;
}

NodePtr Node::rdiv(const Node& other) const {
    return *this * *other.pow(-1.0);
}

NodePtr Node::rtruediv(const Node& other) const {
    return other * *this->pow(-1.0);
}

NodePtr Node::relu() const {
    // std::cout<<"data inside relu func: "<<data<<"\n";
    auto result = std::make_shared<Node>(data > 0 ? data : 0.0,
                                         std::unordered_set<NodePtr>{
                                             std::const_pointer_cast<Node>(shared_from_this())},
                                         "ReLU");

    result->_backward = [this, result]() {
        if (result->data > 0) {
            grad += result->grad;
        }
    };

    return result;
}

/**
 * Computes gradients through backpropagation.
 * Traverses the computation graph in reverse topological order,
 * accumulating gradients for each node.
 */
void Node::backward() {
    {
        std::vector<NodePtr> topo;
        topo.reserve(10);  // Reserve some space to avoid reallocations
        std::unordered_set<NodePtr> visited;

        std::function<void(const NodePtr&)> build_topo = [&](const NodePtr& v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);

                for (const auto& child : v->_prev) {
                    build_topo(child);
                }
                topo.emplace_back(v);
            }
        };

        build_topo(shared_from_this());

        grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const auto& v = *it;
            v->_backward();
            // std::cout << "After Backward: " << v->print() << std::endl; // success.
        }
    }
}