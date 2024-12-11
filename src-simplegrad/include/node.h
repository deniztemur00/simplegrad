#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

using NodePtr = std::shared_ptr<class Node>;
static double constexpr EPSILON = 1e-9;  // for division
class Node : public std::enable_shared_from_this<Node> {
   private:
    float data;
    mutable float grad = 0.0;
    std::string _op;
    std::vector<NodePtr> _prev;
    std::function<void()> _backward;

   public:
    // Constructors
    Node(float data, std::vector<NodePtr> prev = {}, const std::string& op = "");
    ~Node();
    // Getters
    float get_data() const;
    float get_grad() const;
    const std::string& get_op() const;
    const std::vector<NodePtr>& get_prev() const;

    // Setters
    void set_data(float data);
    void set_grad(float grad) const;

    // Overloaded operators
    NodePtr operator+(const Node& other) const;
    NodePtr operator+(float other) const;
    NodePtr radd(const Node& other) const;
    friend NodePtr operator+(float lhs, const Node& rhs);

    NodePtr operator*(const Node& other) const;
    NodePtr operator*(float other) const;
    NodePtr rmul(const Node& other) const;
    friend NodePtr operator*(float lhs, const Node& rhs);

    NodePtr operator-() const;
    NodePtr sub(const Node& other) const;
    NodePtr rsub(const Node& other) const;

    NodePtr pow(float exponent) const;
    NodePtr pow(const Node& other) const;

    NodePtr rdiv(const Node& other) const;
    NodePtr rtruediv(const Node& other) const;

    // Other methods
    NodePtr relu() const;
    void backward();
    std::string print() const;
    void clear_prev();
};