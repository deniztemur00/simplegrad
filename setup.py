from setuptools import setup, find_packages

setup(
    name="simplegrad",
    version="0.0.1",
    description="A simple autograd library",
    author="Deniz",
    packages=find_packages(),
    package_data={
        "py-simplegrad": ["*.so"],
    },
    include_package_data=True,
)