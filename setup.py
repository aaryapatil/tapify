#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tapify",
    version="0.1",
    author="Aarya Patil",
    author_email="patil@astro.utoronto.ca",
    description="A python package for multitaper periodograms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaryapatil/tapify",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    ],
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy"],
    extras_require={
        "extras": ["astropy", "nfft"],
        "tests": ["pytest"]}
)
