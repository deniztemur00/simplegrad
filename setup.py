from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "simplegrad",
        sources=[
            "src-simplegrad/simplegrad.cpp",
            "src-simplegrad/src/mlp.cpp",
            "src-simplegrad/src/net.cpp",
            "src-simplegrad/src/node.cpp",
        ],
        include_dirs=[
            "src-simplegrad/include",
        ],
    ),
]

setup(
    name="simplegrad",
    version="0.0.42",
    description="Automatic differentiation library for basic arithmetic operations",
    author="Deniz",
    url="https://github.com/deniztemur00/simplegrad.git",
    long_description=open("MANIFEST.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "simplegrad": ["*.pyi", "py.typed"],
    },
    data_files=[
        (
            "simplegrad",
            [
                "py-simplegrad/simplegrad/simplegrad.pyi",
                "py-simplegrad/simplegrad/py.typed",
            ],
        )
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    options={"bdist_wheel": {"universal": True}},
)
