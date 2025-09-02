# setup.py
from setuptools import setup, find_packages

setup(
    name="pymatgraph",
    version="0.3.0",
    author="Alec Candidato",
    description="A multiprocess-safe tensor buffer with text and table rendering to RGB buffers.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/0202alcc/pymatgraph",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "numpy",
        "Pillow",
        "pygame",
        "kivy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        "matrixbuffer": ["fonts/*"],
    },
)
