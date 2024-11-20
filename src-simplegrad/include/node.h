#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <set>
#include <iostream>
#include <sstream>

class Node {
private:
    double data;
    double grad;
    std::vector<std::weak_ptr<Node>> _prev;
    std::function<void()> _backward;
    std::string _op;

public:
    Node(double data);

    Node(double data, std::vector<std::shared_ptr<Node>> _prev, std::string _op);

    // Overloaded operators
    Node operator+(const Node& other) const;
    Node operator+(double other) const;
    Node operator-() const;
    Node operator-(const Node& other) const;
    Node operator/(const Node& other) const;
    Node operator*(const Node& other) const;
    Node operator*(double other) const;
    Node radd(double other, const Node& self);
    Node rsub(double other, const Node& self);
    Node rmul(double other, const Node& self);
    Node pow(double exponent) const;
    Node neg() const;
    Node sub(const Node& other) const;
    Node relu() const;

    void backward();

    void set_backward(std::function<void()> backward_func);

    std::string print() const;
};