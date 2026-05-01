"""
CrysCo: Hybrid Graph-Transformer Model for Materials Property Prediction

A PyTorch implementation of a hybrid graph-transformer neural network
that combines Edge Graph Attention Networks (EGAT) and Transformer architectures
for predicting inorganic material properties.
"""

from setuptools import setup, find_packages

setup(
    name="crysco",
    version="0.1.0",
    description="Hybrid Graph-Transformer Model for Materials Property Prediction",
    long_description=__doc__,
    author="Mohammad Madani",
    author_email="mohammad73madani73@gmail.com",
    url="https://github.com/user/CrysCo",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torch-geometric>=2.1.0",
        "torch-scatter>=2.0.9",
        "torch-sparse>=0.6.15",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "ase>=3.22.0",
        "pymatgen>=2022.7.0",
        "matminer>=0.7.8",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    keywords="materials science, machine learning, graph neural networks, transformers, pytorch",
)