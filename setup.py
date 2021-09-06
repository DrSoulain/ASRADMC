#!/usr/bin/env python

from setuptools import setup

setup(
    name="asradmc",
    version="0.1",
    packages=['asradmc'],
    install_requires=["matplotlib", "munch", "numpy", "pandas",
                      "astropy", "scipy", "termcolor", "tqdm",
                      "uncertainties", "astropy", "pathlib",
                      "h5py", "ffmpeg", "seaborn"],
)
