import numpy as np
import matplotlib.pyplot as plt

from py_distmesh2d import *
from poisson import poisson
from meshtools import plot_mesh, plot_mesh_indexed

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, tri

bbox = [[-1, 1], [-1, 1]]

def f_ex(pts):
    return 4.0

def fd_disc(pts):
    return dcircle(pts, 0.0, 0.0, 1.0)

def ex_disc(h0):
    print "  meshing ..."
    p1, t1 = distmesh2d(fd_disc, huniform, h0, bbox, [])
    pts, mytri = fixmesh(p1,t1)
    print mytri
    edges, tedges = edgelist(pts,mytri)
    print edges
    print tedges
    fig1 = plt.figure()
    plot_mesh_indexed(pts, mytri, h0)
    uh, inside = poisson(f_ex,fd_disc,h0,pts,mytri,announce=True)
    print "  plotting ..."
    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')
    ax.plot_trisurf(pts[:,0], pts[:,1], uh, cmap=cm.jet, linewidth=0.2)
    uexact = 1.0 - pts[:,0]**2.0 - pts[:,1]**2.0   # exact:  u(x,y) = 1 - x^2 - y^2
    err = max(abs(uh-uexact))
    print "max error = %f" % err

def fd_ell(pts):
    return ddiff(drectangle(pts, -1, 1, -1, 1), drectangle(pts, -2, 0, -2, 0))

def ex_ell(h0):
    pfix = [[1,1], [1, -1], [0, -1], [0, 0], [-1, 0], [-1, 1]]
    plotmethod = 2
    print "  meshing ..."
    fig1 = plt.figure()
    p1, t1 = distmesh2d(fd_ell, huniform, h0, bbox, pfix)
    pts, mytri = fixmesh(p1,t1)
    plot_mesh(pts, mytri)
    uh, inside = poisson(f_ex,fd_ell,h0,pts,mytri,announce=True)
    print "  plotting ..."
    fig2 = plt.figure()
    if plotmethod == 1:
        plot_mesh(pts, tri, uh)
    elif plotmethod == 2:
        ax = fig2.gca(projection='3d')
        ax.scatter(pts[:,0], pts[:,1], uh, c=uh, cmap=cm.jet)
    else:
        # seems unreliable
        TRI = tri.Triangulation(pts[:,0], pts[:,1], triangles=mytri)
        ax.plot_trisurf(pts[:,0], pts[:,1], uh, TRI.triangles, cmap=cm.jet, linewidth=0.2)

if __name__ == '__main__':
    ex_disc(0.6)
    plt.show()
