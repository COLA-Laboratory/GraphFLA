import re
from pathlib import Path

from setuptools import setup, find_packages

long_description = """
graphfla: A Python package for Graph-based Fitness Landscape Analysis.
========================================================
graphfla provides tools for generating, constructing, analyzing and 
manipulating fitness landscapes commonly encountered in evolutionary biology 
and black-box optimization. It includes a variety of features chacterizing
different aspects of fitness landscape topography, such as ruggedness,
navigability, neutrality, and epistasis.
"""

init_py = (Path(__file__).parent / "graphfla" / "__init__.py").read_text()
version = re.search(r'^__version__ = "([^"]+)"', init_py, re.M).group(1)

setup(
    name="graphfla",
    version=version,
    author="Mingyu Huang",
    author_email="m.huang.gla@outlook.com",
    description="A Python package for Graph-based Fitness Landscape Analysis.",
    long_description=long_description,
    long_description_content_type="text/plain",
    license="MIT",
    url="https://github.com/COLA-Laboratory/GraphFLA/tree/main",
    packages=find_packages(include=["graphfla", "graphfla.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.9",
    install_requires=[
        "joblib>=1.0.0",
        "numpy>=1.19",
        "pandas>=1.1",
        "python-igraph>=0.9",
        "scikit-learn>=0.24",
        "scipy>=1.6.0",
        "rich>=13.0",
    ],
)
