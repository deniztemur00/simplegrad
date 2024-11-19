// Space.h
#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <set>


class Space {
   public:
    // Constructors
    Space(float data, std::vector<Space> children, std::string op);
    ~Space();

    // Accessors
    float data() const;
    float grad() const;
    const std::string& op() const;
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
    // Space radd(float other) const;
    Space rsub(const Space& other) const;
    Space rmul(const Space& other) const;
    Space rtruediv(const Space& other) const;

    // Functions
    Space relu();
    void backward();




    Space(const Space& other); // Declare copy constructor
    Space& operator=(const Space&);
    Space(Space&& other) noexcept = default;
    Space& operator=(Space&& other) noexcept = default;

    //Space(const Space& other);
    //Space& operator=(const Space&) = delete;
    //Space(Space&&) noexcept;
    //Space& operator=(Space&&) noexcept;

   private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};
