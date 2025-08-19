from setuptools import setup, find_packages

setup(
    name="matrixbuffer",
    version="0.2.0",
    description="A Python package for high-performance GPU/CPU buffer rendering with support for tables, text, and graphics.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Alec Candidato",
    author_email="aleccandidato@gmail.com",
    url="https://github.com/0202alcc/matrixbuffer",
    packages=find_packages(),  # <- just find all packages under current dir
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "pygame>=2.0.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="MIT",  # SPDX license expression
)
