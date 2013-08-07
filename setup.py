from setuptools import setup, find_packages

setup(
    name='distmesh2d',
    version='0.0.1',
    url="https://github.com/fmuzf/py_distmesh2d",
    packages=find_packages(),
    license='MIT',
    install_requires=[
        "distribute",
        "numpy >= 1.7.0",
        "scipy >= 0.10"
    ]
)
