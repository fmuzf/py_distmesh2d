from pylab import triplot, tripcolor, axis, axes, text
from py_distmesh2d import *
import numpy as np

def plot_mesh(pts, tri, *args):
    if len(args) > 0:
        tripcolor(pts[:,0], pts[:,1], tri, args[0], edgecolor='black', cmap="Blues")
    else:
        triplot(pts[:,0], pts[:,1], tri, "k-", lw=2)
    axis('tight')
    axes().set_aspect('equal')

def plot_mesh_indexed(pts, tri, h0, *args):
    if len(args) > 0:
        tripcolor(pts[:,0], pts[:,1], tri, args[0], edgecolor='black', cmap="Blues")
    else:
        triplot(pts[:,0], pts[:,1], tri, "k-", lw=2)
    for i in range(np.shape(tri)[0]):   # label triangle index in green
        x = np.sum(pts[tri[i,:],0]) / 3.0
        y = np.sum(pts[tri[i,:],1]) / 3.0
        text(x,y,'%d' % i, color='g')
    for j in range(np.shape(pts)[0]):   # label point index in red
        text(pts[j,0]+h0/10.0,pts[j,1],'%d' % j, color='r')
    axis('tight')
    axes().set_aspect('equal')

