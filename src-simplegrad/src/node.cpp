#include "../include/node.h"

// Constructor
Node::Node(float data, std::unordered_set<std::shared_ptr<Node>> prev, const std::string& op)
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

const std::unordered_set<std::shared_ptr<Node>>& Node::get_prev() const {
    return _prev;
}

std::string Node::print() const {
    std::stringstream ss;
    ss << "Node(data=" << data
       << ", grad=" << grad
       << ", op='" << _op << "')";
    return ss.str();
}
std::shared_ptr<Node> Node::operator+(const Node& other) const {
    auto result = std::make_shared<Node>(
        this->data + other.data,
        std::unordered_set<std::shared_ptr<Node>>{
            std::const_pointer_cast<Node>(shared_from_this()),
            std::const_pointer_cast<Node>(other.shared_from_this())},
        "+");

    result->_backward = [this, &other, result]() {
        grad += result->grad;
        other.grad += result->grad;
    };
    return result;
}

std::shared_ptr<Node> Node::operator+(float other) const {
    return *this + Node(other);
}

std::shared_ptr<Node> operator+(float lhs, const Node& rhs) {
    return rhs + lhs;
}

/*
Node Node::operator*(const Node& other) const {
    auto shared_this = std::make_shared<Node>(*this);
    auto shared_other = std::make_shared<Node>(other);
    auto out = std::make_shared<Node>(data * other.data,
                                      std::vector<std::shared_ptr<Node>>{shared_this, shared_other}, "*");

    out->set_backward([shared_this, shared_other, out]() {
        shared_this->grad += shared_other->data * out->grad;
        shared_other->grad += shared_this->data * out->grad;
        //std::cout << "Backward: " << shared_this->grad << " " << shared_other->grad << std::endl;
    });

    return *out;
}

Node Node::operator*(float other) const {
    return *this * Node(other);
}

Node Node::pow(float exponent) const {
    auto shared_this = std::make_shared<Node>(*this);
    auto out = std::make_shared<Node>(std::pow(shared_this->data, exponent),
                                      std::vector<std::shared_ptr<Node>>{shared_this}, "pow");

    out->set_backward([shared_this, exponent, out]() {
        shared_this->grad += (exponent * std::pow(shared_this->data, exponent - 1)) * out->grad;
    });

    return *out;
}

Node Node::operator-() const {
    auto shared_this = std::make_shared<Node>(*this);
    auto out = std::make_shared<Node>(-shared_this->data, std::vector<std::shared_ptr<Node>>{shared_this}, "-");

    out->set_backward([shared_this, out]() {
        shared_this->grad += -out->grad;
    });

    return *out;
}

Node Node::sub(const Node& other) const {
    return *this + (-other);
}

Node Node::neg() const {
    return -(*this);
}

Node Node::radd(const Node& other) {
    return *this + other;
}

Node Node::rsub(const Node& other) {
    return other + (-*this);
}

Node Node::rmul(const Node& other) {
    return other * *this;
}

Node Node::rdiv(const Node& other) {
    return other * this->pow(-1);
}

Node Node::rtruediv(const Node& other) {
    return *this * other.pow(-1);
}

Node Node::relu() const {
    auto shared_this = std::make_shared<Node>(*this);
    auto out = std::make_shared<Node>(std::max(0.0, shared_this->data), std::vector<std::shared_ptr<Node>>{shared_this}, "ReLU");

    out->set_backward([shared_this, out]() {
        if (out->data > 0) {
            shared_this->grad += out->grad;
        }
    });

    return *out;
}
*/

void Node::backward() {
    std::vector<std::shared_ptr<Node>> topo;
    topo.reserve(10); // Reserve some space to avoid reallocations
    std::unordered_set<std::shared_ptr<Node>> visited;

    std::function<void(const std::shared_ptr<Node>&)> build_topo = [&](const std::shared_ptr<Node>& v) {
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
        std::cout << "After Backward: " << v->print() << std::endl; // success.
    }
}