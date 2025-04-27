from setuptools import setup, find_packages

setup(
    name="selector_mlp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchmetrics>=0.11.0",
        "pytorch-lightning>=1.9.0",
        "scikit-learn>=1.0.2",
        "numpy>=1.20.0",
        "pandas>=1.4.0",
        "anndata>=0.8.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        "scanpy>=1.9.0",
        "seaborn>=0.12.0",
    ],
    author="Alexander Epstein",
    author_email="alexander.epstein96@gmail.com",
    description="A PyTorch-based feature selection and classification framework for high-dimensional biological data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aepstein96/gene-selector-mlp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
) 