// Space.h
#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <sstream>
#include <cmath>
#include <set>

class Space {
   public:
    Space(float data);
    ~Space();

    // Accessors
    const float data();
    const float grad();
    const std::string& op();
    std::string print() const;

    // Operators
    Space operator+(const Space& other) const;
    Space operator+(float other) const;
    Space operator*(const Space& other) const;
    Space operator*(float other) const;
    Space operator-() const;
    Space pow(float other) const;
    Space neg() const;
    Space sub(const Space& other) const;
    Space truediv(const Space& other) const;
    Space radd(const Space& other) const;
    //Space radd(float other) const;
    Space rsub(const Space& other) const;
    Space rmul(const Space& other) const;
    Space rtruediv(const Space& other) const;

    // Functions
    Space relu();
    void backward();

    // Prevent copying, allow moving
    Space(const Space& other);
    //Space& operator=(const Space&) = default;
    Space(Space&&) noexcept;
    Space& operator=(Space&&) noexcept;

   private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};
