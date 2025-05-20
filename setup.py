long_description = """
graphfla: A Python package for Graph-based Fitness Landscape Analysis.
========================================================
graphfla provides tools for generating, constructing, analyzing and 
manipulating fitness landscapes commonly encountered in evolutionary biology 
and black-box optimization. It includes a variety of features chacterizing
different aspects of fitness landscape topography, such as ruggedness,
navigability, neutrality, and epistasis.
"""

from setuptools import setup, find_packages

setup(
    name="graphfla",
    version="0.1.0",
    author="Mingyu Huang",
    author_email="m.huang.gla@outlook.com",
    description="A Python package for Graph-based Fitness Landscape Analysis.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/COLA-Laboratory/GraphFLA/tree/main",
    packages=find_packages(include=["graphfla", "graphfla.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
    install_requires=[
        "joblib>=1.0.0",
        "numpy>=1.19",
        "pandas>=1.1",
        "python-igraph>=0.9",
        "scikit-learn>=0.24",
        "scipy>=1.6.0",
        "tqdm>=4.40",
    ],
)
