#include "../include/node.h"

Node::Node(double data, std::vector<Node*> _prev, std::string _op)
    : data(data), grad(0.0), _backward([] {}), _op(_op) {
    for (auto& p : _prev) {
        graph.emplace_back(p);
        std::cout << "Prev: " << p->data << std::endl;
    }
    std::cout << "Constructed Node for Data: " << data << std::endl;
}

double Node::get_data() const {
    return data;
}

double Node::get_grad() const {
    return grad;
}

std::string Node::get_op() const {
    return _op;
}

void Node::set_backward(std::function<void()> backward_func) {
    _backward = backward_func;
}

std::string Node::print() const {
    std::stringstream ss;
    ss << "Node(data=" << data
       << ", grad=" << grad
       << ", op='" << _op << "')";
    return ss.str();
}

Node Node::operator+(Node& other) const {
    Node out = Node(data + other.data, std::vector<Node*>{this, &other}, "+");
    
}

Node Node::operator+(double other) const {
    return *this + Node(other);
}

Node Node::operator*(const Node& other) const {
    auto shared_this = std::make_shared<Node>(*this);
    auto shared_other = std::make_shared<Node>(other);
    auto out = std::make_shared<Node>(data * other.data,
                                      std::vector<std::shared_ptr<Node>>{shared_this, shared_other}, "*");

    out->set_backward([shared_this, shared_other, out]() {
        shared_this->grad += shared_other->data * out->grad;
        shared_other->grad += shared_this->data * out->grad;
        std::cout << "Backward: " << shared_this->grad << " " << shared_other->grad << std::endl;
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

void Node::backward() {
    this->grad = 1.0;
    for (auto it = graph.rbegin(); it != graph.rend(); ++it) {
        (*it)->_backward();
    }
}

/*
void Node::backward() {
    std::cout << "\n=== Starting backward() ===\n";

    // Topological sort
    std::vector<std::shared_ptr<Node>> topo;
    std::set<std::shared_ptr<Node>> visited;

    std::function<void(std::shared_ptr<Node>)> build_topo;

    build_topo = [&](std::shared_ptr<Node> v) {
        std::cout << "Processing node with value: " << v->data << std::endl;

        if (visited.find(v) == visited.end()) {
            std::cout << "  Node not visited before, adding to visited set\n";
            visited.insert(v);

            std::cout << "  Checking " << v->_prev.size() << " previous nodes\n";
            for (auto& w : v->_prev) {
                if (auto sp = w.lock()) {
                    std::cout << "    Following edge to node with value: " << sp->data << std::endl;
                    build_topo(sp);
                } else {
                    std::cout << "    Warning: Weak pointer expired\n";
                }
            }
            std::cout << "  Adding node " << v->data << " to topological sort\n";
            topo.push_back(v);
        } else {
            std::cout << "  Node already visited, skipping\n";
        }
    };

    std::cout << "Starting topological sort from root node\n";
    build_topo(std::make_shared<Node>(*this));

    std::cout << "\nTopological sort complete. Nodes in order:\n";
    for (const auto& node : topo) {
        std::cout << node->data << " ";
    }
    std::cout << "\n\nStarting backward pass\n";

    this->grad = 1.0;
    std::cout << "Set root gradient to 1.0\n";

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        std::cout << "Processing node with value: " << (*it)->data << ", gradient: " << (*it)->grad << std::endl;
        if ((*it)->_backward) {
            std::cout << "  Calling _backward()\n";
            (*it)->_backward();
        } else {
            std::cout << "  No backward function defined\n";
        }
        std::cout << "  Node gradient after backward: " << (*it)->grad << std::endl;
    }

    std::cout << "=== Finished backward() ===\n\n";
}
*/