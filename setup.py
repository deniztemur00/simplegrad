from setuptools import setup, Extension
import sys
import os

# Define package metadata and extensions
cpp_sources = [
    os.path.join("src-simplegrad", "simplegrad.cpp"),
    os.path.join("src-simplegrad", "src", "node.cpp"),
    os.path.join("src-simplegrad", "src", "mlp.cpp"),
    os.path.join("src-simplegrad", "src", "net.cpp"),
]


def get_compiler_flags():
    if sys.platform.startswith("win"):
        return [
            "/std:c++17",  # C++17 standard
            "/O2",  # Optimize for speed
            "/GL",  # Whole program optimization
            "/MD",  # Multi-threaded DLL runtime
            "/EHsc",  # Exception handling
            "/DNDEBUG",  # Disable debug
        ]
    else:
        flags = ["-std=c++17", "-fvisibility=hidden", "-O3", "-DNDEBUG"]
        if sys.platform.startswith("linux"):
            flags.extend(
                ["-march=native", "-ffast-math", "-flto", "-funroll-loops", "-fPIC"]
            )
        return flags


def get_ext_modules():
    try:
        import pybind11

        include_dirs = [
            pybind11.get_include(),
            os.path.join(os.path.dirname(__file__), "src-simplegrad", "include"),
        ]

        ext = Extension(
            "simplegrad._simplegrad",
            cpp_sources,
            include_dirs=include_dirs,
            language="c++",
            extra_compile_args=get_compiler_flags(),
            extra_link_args=["/DLL"] if sys.platform.startswith("win") else ["-flto"],
        )
        return [ext]
    except ImportError:
        print("pybind11 is not installed.")
        return []


setup(
    name="simplegrad",
    version="0.0.54",
    description="Automatic differentiation library for basic arithmetic operations",
    author="Deniz",
    url="https://github.com/deniztemur00/simplegrad.git",
    long_description=open("MANIFEST.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"simplegrad": "py-simplegrad/simplegrad-dev"},
    packages=["simplegrad"],
    requires=["pybind11"],
    ext_modules=get_ext_modules(),
    package_data={
        "simplegrad": ["*.pyi", "*.so","*.pyd","*.dll","*.lib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    license="MIT",
    python_requires=">=3.6",
)

