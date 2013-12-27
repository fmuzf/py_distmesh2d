py_fem_distmesh2d
=================

A self-contained finite element example using Python and scipy and matplotlib.

The codes `py_distmesh2d.py` and `mesh_examples.py` are from [py_distmesh2d](https://github.com/ckhroulev/py_distmesh2d) by Constantine Khroulev.  They are a Python re-implementation of `distmesh2d` in *P.-O. Persson, G. Strang, A Simple Mesh Generator in MATLAB. SIAM Review, Volume 46 (2), pp. 329-345, June 2004* (http://persson.berkeley.edu/distmesh/).

Several tools in `meshtools.py` and also `poisson.py` are Python re-implementations of codes from the [Math 692 one-page Matlab FEM code challenge](http://www.dms.uaf.edu/~bueler/challenge.htm), which dates to Fall 2004.  These tools include `fixmesh`, `edgelist`, and `bdyrefine`.

To demo:

    $ python fem_examples.py

It gives this solution plot:

![disc solution](https://github.com/bueler/py_fem_distmesh2d/raw/master/ex_disc_soln.png)
