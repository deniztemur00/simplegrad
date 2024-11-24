#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class Node {
   private:
    double data;
    double grad;
    std::vector<std::shared_ptr<Node>> _prev;
    std::function<void()> _backward;
    std::string _op;

   public:
    // Constructors
    Node(double data, const std::vector<std::shared_ptr<Node>>& prev = {}, const std::string& op = "")
        : data(data), grad(0.0), _prev(prev), _op(op), _backward([]() {}) {
    }

    // Getters
    double get_data() const;
    double get_grad() const;
    std::string get_op() const;

    // Overloaded operators
    Node operator+(const Node& other) const;
    Node operator+(double other) const;
    Node operator-() const;
    Node operator-(const Node& other) const;
    Node operator/(const Node& other) const;
    Node operator*(const Node& other) const;
    Node operator*(double other) const;
    Node radd(const Node& self);
    Node rsub(const Node& self);
    Node rmul(const Node& self);
    Node rdiv(const Node& other);
    Node rtruediv(const Node& other);
    Node pow(double exponent) const;
    Node neg() const;
    Node sub(const Node& other) const;
    Node relu() const;

    void backward();

    void set_backward(std::function<void()> backward_func);

    std::string print() const;
};