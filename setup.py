from setuptools import setup


setup(
    name="simplegrad",
    version="0.0.45",
    description="Automatic differentiation library for basic arithmetic operations",
    author="Deniz",
    url="https://github.com/deniztemur00/simplegrad.git",
    long_description=open("MANIFEST.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"simplegrad": "py-simplegrad/simplegrad-dev"},
    packages=["simplegrad"],
    package_data={
        "simplegrad": ["*.pyi", "py.typed", "*.so", "__init__.py"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    license="MIT",
    author_email="deniztemur00@gmail.com",
    python_requires=">=3.6",
    options={"bdist_wheel": {"universal": True}},
)
