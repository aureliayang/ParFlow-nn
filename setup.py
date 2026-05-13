#!/usr/bin/env python
from setuptools import setup


setup(
    name="parflow-nn",
    version="0.1.0",
    description="Hybrid Physics-ML framework for ParFlow pressure prediction",
    packages=[],
    python_requires=">=3.10,<3.11",
    install_requires=[
        "pftools",
    ],
    zip_safe=False,
)
