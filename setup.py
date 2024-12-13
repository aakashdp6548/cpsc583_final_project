from setuptools import setup, find_packages

setup(
    name="cell-interaction-gnn",
    version="0.1.0",
    description="GNN-based model for predicting cell-type interactions using gene expression data",
    author="Aakash Patel",
    author_email="aakash.patel.ap2853@yale.edu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "argparse>=1.4.0",
    ],
)