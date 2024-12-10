cmake_minimum_required(VERSION 3.12)
project(simplegrad)

set(CMAKE_CXX_STANDARD 17)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()


if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})


find_package(pybind11)

if(NOT pybind11_FOUND)
    include(FetchContent)
    message(STATUS "pybind11 not found. Downloading...")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

if(NOT TARGET pybind11::module)
    message(FATAL_ERROR "pybind11 installation failed or incomplete.")
endif()

message(STATUS "Found pybind11 v${pybind11_VERSION}")

file(GLOB SOURCES "src-simplegrad/src/*.cpp")
file(GLOB HEADERS "src-simplegrad/include/*.h")



set(SRC_DIR ${CMAKE_SOURCE_DIR}/src-simplegrad)


include_directories(${pybind11_INCLUDE_DIRS})


pybind11_add_module(simplegrad ${SRC_DIR}/simplegrad.cpp ${SOURCES} ${HEADERS})

set_target_properties(simplegrad PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/py-simplegrad/simplegrad-dev/
)