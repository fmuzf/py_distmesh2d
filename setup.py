from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='distmesh2d',
    version='0.0.3',
    url="https://github.com/fmuzf/py_distmesh2d",
    description='A Python re-implementation of distmesh2d by Persson and Strang',
    long_description=read('README.md'),
    packages=['distmesh2d'],
    install_requires=[
        "distribute",
        "numpy >= 1.7.0",
        "scipy >= 0.10"
    ]
)
