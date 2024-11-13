cmake_minimum_required(VERSION 3.12)
project(simplegrad)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

# Find pybind11 package
find_package(pybind11 REQUIRED)

# Set the source directory
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src-simplegrad)

# Include pybind11 directories
include_directories(${pybind11_INCLUDE_DIRS})

# Add the pybind11 module
pybind11_add_module(simplegrad ${SRC_DIR}/simplegrad.cpp)

set_target_properties(simplegrad PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/py-simplegrad/module/
)