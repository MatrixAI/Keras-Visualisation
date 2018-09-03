#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='keras-visualisaton',
    version='0.0.1',
    author='CMCDragonkai',
    author_email='roger.qiu@polyhack.io',
    long_description=long_description,
    packages=find_packages(),
    scripts=[],
    install_requires=['numpy', 'Keras'])
