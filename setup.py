"""
This file contains the setup for the `radiologynet-toolset` package. The
`radiologynet-toolset` package is in the local directory `radiologynet`.
To install the `radiologynet-toolset` package, simply run:
    `pip install .`
from within the directory where this setup.py is located.
"""

from setuptools import setup, find_packages

setup(
    name='radiologynet-toolset',
    packages=find_packages(exclude=['_testing']),
    version='0.0.1',
    description='RadiologyNet: Machine Learning for Knowledge Transfer',
    author='@RITEH: mnapravnik, fhrzic, rbazdaric, istajduh',
    license='MIT',
    install_requires=[
        # any requirements are to be listed here.
        'numpy>=1.16.3',
        'scipy>=1.4.1',
        'pydicom>=2.1.1',
        'pandas>=0.25.1',
        'dask>=2022.2.1',
        'distributed>=2022.2.1',
        'scikit-learn>=1.0.2',
        'SQLAlchemy>=1.4.32',
        'Pillow>=9.0.1',
        'statsmodels>=0.13.2'
        # torch install is specified in README
        # ...
    ]
)
