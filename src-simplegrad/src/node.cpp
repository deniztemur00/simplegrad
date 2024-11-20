#include "../include/node.h"

Node::Node(double data)
    : data(data), grad(0.0), _prev({}), _backward([] {}), _op("") {}

Node::Node(double data, std::vector<std::shared_ptr<Node>> _prev, std::string _op)
    : data(data), grad(0.0), _prev({}), _backward([] {}), _op(_op) {
    for (auto& p : _prev) {
        _prev.push_back(p);
    }
}

void Node::set_backward(std::function<void()> backward_func) {
    _backward = backward_func;
}

std::string Node::print() const {
    std::stringstream ss;
    ss << "Space(data=" << data
       << ", grad=" << grad;
    if (_op.empty()) {
        ss << ", op='" << _op << "'";
    }
    ss << ")";
    return ss.str();
}

Node Node::operator+(const Node& other) const {
    auto shared_this = std::make_shared<Node>(*this);
    auto shared_other = std::make_shared<Node>(other);
    auto out = std::make_shared<Node>(data + other.data, {shared_this, shared_other}, "+");
    out->set_backward([shared_this, shared_other, out]() {
        shared_this->grad += out->grad;
        shared_other->grad += out->grad;
    });

    return *out;
}

Node Node::operator+(double other) const {
    return *this + Node(other);
}

Node Node::operator*(const Node& other) const {
    auto shared_this = std::make_shared<Node>(*this);
    auto shared_other = std::make_shared<Node>(other);
    auto out = std::make_shared<Node>(data * other.data, {shared_this, shared_other}, "*");

    out->set_backward([shared_this, shared_other, out]() {
        shared_this->grad += shared_other->data * out->grad;
        shared_other->grad += shared_this->data * out->grad;
    });

    return *out;
}

Node Node::operator*(double other) const {
    return *this * Node(other);
}

Node Node::pow(double exponent) const {
    auto shared_this = std::make_shared<Node>(*this);
    auto out = std::make_shared<Node>(std::pow(shared_this->data, exponent), {shared_this}, "pow");

    out->set_backward([shared_this, exponent, out]() {
        shared_this->grad += (exponent * std::pow(shared_this->data, exponent - 1)) * out->grad;
    });

    return *out;
}
Node Node::operator-() const {
    auto shared_this = std::make_shared<Node>(*this);
    auto out = std::make_shared<Node>(-shared_this->data, {shared_this}, "-");

    out->set_backward([shared_this, out]() {
        shared_this->grad += -out->grad;
    });

    return *out;
}

Node Node::operator-(const Node& other) const {
    return *this + (-other);
}

Node Node::operator/(const Node& other) const {
    return *this * other.pow(-1);
}

// Reverse addition
Node radd(double other, const Node& self) {
    return self + other;
}

// Reverse subtraction
Node rsub(double other, const Node& self) {
    return other + (-self);
}

// Reverse multiplication
Node rmul(double other, const Node& self) {
    return self * other;
}

// Reverse division
Node rdiv(double other, const Node& self) {
    return other * self.pow(-1);
}



Node Node::relu() const {
    auto shared_this = std::make_shared<Node>(*this);
    auto out = std::make_shared<Node>(std::max(0.0, shared_this->data), {shared_this}, "ReLU");

    out->set_backward([shared_this, out]() {
        if (out->data > 0) {
            shared_this->grad += out->grad;
        }
    });

    return *out;
}

void Node::backward() {
    // Topological sort
    std::vector<std::shared_ptr<Node>> topo;
    std::set<std::shared_ptr<Node>> visited;

    std::function<void(std::shared_ptr<Node>)> build_topo;
    build_topo = [&](std::shared_ptr<Node> v) {
        if (visited.find() == visited.end()) {
            visited.insert(v);
            for (auto& w : v->_prev) {
                if (auto sp = w.lock()) {
                    build_topo(sp);
                }
            }
            topo.push_back(v);
        }
    };

    build_topo(std::make_shared<Node>(*this));

    // Backpropagate gradients
    this->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}