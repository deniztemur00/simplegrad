#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>


class Node : public std::enable_shared_from_this<Node> {
   private:
    float data;
    mutable float grad = 0.0;
    std::unordered_set<std::shared_ptr<Node>> _prev;
    std::function<void()> _backward;
    std::string _op;

   public:
    // Constructors
    Node(float data, std::unordered_set<std::shared_ptr<Node>> prev = {}, const std::string& op = "");
    ~Node();
    // Getters
    float get_data() const;
    float get_grad() const;
    const std::string& get_op() const;
    const std::unordered_set<std::shared_ptr<Node>>& get_prev() const;

    // Overloaded operators
    std::shared_ptr<Node> operator+(const Node& other) const;
    std::shared_ptr<Node> operator+(float other) const;

    // Free functions
    friend std::shared_ptr<Node> operator+(float lhs, const Node& rhs);

    Node operator-() const;
    Node operator-(const Node& other) const;
    Node operator/(const Node& other) const;
    Node operator*(const Node& other) const;
    Node operator*(float other) const;
    Node radd(const Node& self);
    Node rsub(const Node& self);
    Node rmul(const Node& self);
    Node rdiv(const Node& other);
    Node rtruediv(const Node& other);
    Node pow(float exponent) const;
    Node neg() const;
    Node sub(const Node& other) const;
    Node relu() const;

    void backward();

    std::string print() const;
};