// Space.h
#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <tuple>

class Space {
public:
    Space(float data, std::tuple<> children, std::string op);
    ~Space();
    
    // Prevent copying, allow moving
    Space(const Space&) = delete;
    Space& operator=(const Space&) = delete;
    Space(Space&&) noexcept;
    Space& operator=(Space&&) noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};


