import sys

from setuptools import setup

if sys.version_info.major != 3:
    print("This module is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(
    name='ui_adapt',
    author='Dagasfi',
    version='0.0.2',
    install_requires=[
        'pandas==1.0.5',
        'scipy==1.2.2',
        'gym==0.10.5',
        'numpy==1.13.3',
        'matplotlib',
        'scikit-learn==0.24.2'
    ]
)