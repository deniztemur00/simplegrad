from setuptools import setup, find_packages

setup(
    name="simplegrad",
    version="0.1.1",
    description="A simple autograd library",
    author="Deniz",
    packages=find_packages(),
    package_data={
        "simplegrad": ["*.so"],
    },
    include_package_data=True,
)