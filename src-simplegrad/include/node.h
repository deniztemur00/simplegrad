#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>

using NodePtr = std::shared_ptr<class Node>;
#define EPSILON 1e-9
class Node : public std::enable_shared_from_this<Node> {
   private:
    float data;
    mutable float grad = 0.0;
    std::unordered_set<NodePtr> _prev;
    std::function<void()> _backward;
    std::string _op;

   public:
    // Constructors
    Node(float data, std::unordered_set<NodePtr> prev = {}, const std::string& op = "");
    ~Node();
    // Getters
    float get_data() const;
    float get_grad() const;
    const std::string& get_op() const;
    const std::unordered_set<NodePtr>& get_prev() const;

    // Overloaded operators
    NodePtr operator+(const Node& other) const;
    NodePtr operator+(float other) const;
    NodePtr radd(const Node& other) const;
    friend NodePtr operator+(float lhs, const Node& rhs);

    NodePtr operator*(const Node& other) const;
    NodePtr operator*(float other) const;
    NodePtr rmul(const Node& other) const;
    NodePtr operator-() const;
    NodePtr sub(const Node& other) const;
    NodePtr rsub(const Node& other) const;
    NodePtr pow(float exponent) const;
    NodePtr pow(const Node& other) const;
    NodePtr rdiv(const Node& other) const;
    NodePtr rtruediv(const Node& other) const;
    NodePtr relu() const;
    //Node rdiv(const Node& other);
    //Node operator*(const Node& other) const;
    //Node operator*(float other) const;
    //Node operator-(const Node& other) const;
    //Node sub(const Node& other) const;
    //Node pow(float exponent) const;
    //Node rtruediv(const Node& other);
    //Node rsub(const Node& self);
    //Node radd(const Node& self);


    //Node operator/(const Node& other) const;
    //Node rmul(const Node& self);
    //Node relu() const;

    void backward();

    std::string print() const;
};