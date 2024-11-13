from setuptools import setup, find_packages

setup(
    name="simplegrad",
    packages=find_packages(),
    package_data={
        "simplegrad": ["./build/*.so", "*.pyd"],
    },
)