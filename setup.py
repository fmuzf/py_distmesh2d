from setuptools import setup, find_packages

setup(
    name='distmesh2d',
    version='0.0.2',
    url="https://github.com/fmuzf/py_distmesh2d",
    description='A Python re-implementation of distmesh2d by Persson and Strang',
    packages=find_packages(),
    install_requires=[
        "distribute",
        "numpy >= 1.7.0",
        "scipy >= 0.10"
    ]
)
