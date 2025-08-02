from setuptools import setup, find_packages

setup(
    name='matrixbuffer',
    version='0.1.0',
    author='Your Name',
    author_email='aleccandidato@gmail.com',
    description='A multiprocess-safe buffer for PyTorch tensors with rendering capabilities using Pygame.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/matrixbuffer',  # Replace with your project's URL
    packages=find_packages(),
    install_requires=[
        'pygame',
        'torch',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)