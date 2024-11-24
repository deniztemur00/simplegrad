#include "../include/node.h"

double Node::get_data() const {
    return data;
}

double Node::get_grad() const {
    return grad;
}

std::string Node::get_op() const {
    return _op;
}

void Node::set_backward(std::function<void(double)> backward_func) {
    _backward = backward_func;
}

std::string Node::print() const {
    std::stringstream ss;
    ss << "Node(data=" << data
       << ", grad=" << grad
       << ", op='" << _op << "')";
    return ss.str();
}

Node Node::operator+(const Node& other) const {
    auto self_ptr = std::make_shared<Node>(*this);
    auto other_ptr = std::make_shared<Node>(other);
    std::vector<std::shared_ptr<Node>> children = {self_ptr, other_ptr};
    Node out(this->data + other.data, children, "+");

    out._backward = [self_ptr, other_ptr](double out_grad) {
        self_ptr->grad += out_grad;
        other_ptr->grad += out_grad;
        std::cout << "Backward: " << self_ptr->grad << " " << other_ptr->grad << std::endl;
    };

    return out;
}

Node Node::operator+(double other) const {
    return *this + Node(other);
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

Node Node::operator*(double other) const {
    return *this * Node(other);
}

Node Node::pow(double exponent) const {
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
    std::vector<Node*> topo;
    std::set<Node*> visited;

    std::function<void(Node*)> build_topo = [&](Node* v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const auto& child : v->_prev) {
                build_topo(child.get());
            }
            topo.push_back(v);
        }
    };

    build_topo(this);
    this->grad = 1.0;
    std::cout << "This node: " << this->print() << std::endl;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        std::cout << "Processing backward on: " << (*it)->print() << std::endl;
        (*it)->_backward((*it)->grad);
        std::cout << "After backward: " << (*it)->print() << std::endl;
    }
}