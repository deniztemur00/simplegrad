import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

cpp_sources = [
    "src-simplegrad/simplegrad.cpp",
    "src-simplegrad/src/node.cpp",
    "src-simplegrad/src/mlp.cpp",
    "src-simplegrad/src/net.cpp",
]

# No -march=native / -ffast-math: wheels must run on any CPU and keep
# IEEE float semantics. Pybind11Extension already adds /O2 or -O2-level
# defaults, -fvisibility=hidden, and the C++ standard flag.
extra_compile_args = [] if sys.platform == "win32" else ["-O3", "-funroll-loops"]

ext_modules = [
    Pybind11Extension(
        "simplegrad._simplegrad",
        sources=cpp_sources,
        include_dirs=["src-simplegrad/include"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    package_dir={"simplegrad": "py-simplegrad/simplegrad-dev"},
    packages=["simplegrad"],
    package_data={"simplegrad": ["simplegrad.pyi", "py.typed"]},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
