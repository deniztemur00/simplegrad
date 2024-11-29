from setuptools import setup, find_packages

setup(
    name="simplegrad",
    version="0.1.2",
    description="Automatic differentiation library for basic arithmetic operations",
    author="Deniz",
    packages=find_packages(),
    package_data={
        "simplegrad": ["*.so", "*.pyi"],
    },
    include_package_data=True,
)