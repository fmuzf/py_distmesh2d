from pylab import figure, triplot, tripcolor, axis, axes, show
from py_distmesh2d import *
from examples import plot_mesh
import numpy as np
from numpy.linalg import det, inv, solve

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, tri
import matplotlib.pyplot as plt

bbox = [[-1, 1], [-1, 1]]

def f_ex(pts):
    return 4.0

def fd_disc(pts):
    return dcircle(pts, 0.0, 0.0, 1.0)

def ex_disc():
    h0 = 0.15
    print "  meshing ..."
    pts, mytri = distmesh2d(fd_disc, huniform, h0, bbox, [])
    fig1 = plt.figure()
    plot_mesh(pts, mytri)
    uh, inside = poisson(f_ex,fd_disc,h0,pts,mytri)
    print "  plotting ..."
    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')
    ax.plot_trisurf(pts[:,0], pts[:,1], uh, cmap=cm.jet, linewidth=0.2)

def fd_ell(pts):
    return ddiff(drectangle(pts, -1, 1, -1, 1), drectangle(pts, -2, 0, -2, 0))

def ex_ell():
    pfix = [[1,1], [1, -1], [0, -1], [0, 0], [-1, 0], [-1, 1]]
    h0 = 0.15
    plotmethod = 2
    print "  meshing ..."
    fig1 = plt.figure()
    p1, t1 = distmesh2d(fd_ell, huniform, h0, bbox, pfix)
    pts, mytri = fixmesh(p1,t1)
    plot_mesh(pts, mytri)
    uh, inside = poisson(f_ex,fd_ell,h0,pts,mytri)
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

def poisson(f_rhs,f_dist,h0,pts,tri):
    """Solve Poisson's equation on a domain D by the FE method:
       - Laplacian u = f on D,  u=0 on boundary of D
    using triangulation described by pts, tri, a mesh size h0, and a signed 
    distance function fd (see Persson & Strang's distmesh2d.m).  Returns
    an approximate solution  uh  defined at all vertices.  Returns
       uh = solution value
       inside = (if nonnegative, gives index of interior point)  
    """

    #function [uh,in]=poissonv2(f,fd,h0,p,t,varargin);
    #Example:
    #>> f=inline('4','p'); fd=inline('sqrt(sum(p.^2,2))-1','p');
    #>> [p,t]=distmesh2d(fd,@huniform,0.5,[-1,-1;1,1],[]);
    #>> [uh,in]=poissonv2(f,fd,0.5,p,t);
    #>> u=1-sum(p.^2,2); err=max(abs(uh-u))

    print "  assembling ..."
    geps = 0.001 * h0
    ii = (f_dist(pts) < -geps)      # boolean array for interior nodes
    Npts = np.shape(pts)[0]     # = number of nodes
    N = ii.sum()                # = number of *interior* nodes
    inside = np.zeros(Npts,dtype=np.int32) # index only the interior nodes
    count = 0
    for j in range(Npts):
        if ii[j]:
          inside[j] = count
          count = count + 1
        else:
          inside[j] = -1

    # eval f_rhs once for each node
    ff = np.zeros(Npts)
    for j in range(Npts):
        ff[j] = f_rhs(pts[j])
    
    #% loop over triangles to set up stiffness matrix A and load vector b
    # NOTE: not using sparse matrices at all
    A = np.zeros((N,N))
    b = np.zeros(N)
    for n in range(np.shape(tri)[0]):
        j = tri[n,0]
        k = tri[n,1]
        l = tri[n,2]
        vj = inside[j]
        vk = inside[k]
        vl = inside[l]
        
        # Jacobian of triangle
        #J=[p(k,1)-p(j,1), p(l,1)-p(j,1); p(k,2)-p(j,2), p(l,2)-p(j,2)];
        Jac = np.array([[ pts[k,0] - pts[j,0], pts[l,0] - pts[j,0] ],
                        [ pts[k,1] - pts[j,1], pts[l,1] - pts[j,1] ]])
        #ar=abs(det(J))/2;  C=ar/12;  Q=inv(J'*J);  fT=[ff(j) ff(k) ff(l)];
        ar = abs(det(Jac))/2.0
        C = ar/12.0
        Q = inv(np.dot(Jac.transpose(),Jac))
        fT = np.array([ff[j], ff[k], ff[l]])
        
        if ii[j]:
            A[vj,vj] += ar * np.sum(Q)
            b[vj]    += C * np.dot(fT, np.array([2,1,1]))
        if ii[k]:
            A[vk,vk] += ar * Q[0,0]
            b[vk]    += C * np.dot(fT, np.array([1,2,1]))
        if ii[l]:
            A[vl,vl] += ar * Q[1,1]
            b[vl]    += C * np.dot(fT, np.array([1,1,2]))
        if ii[j] & ii[k]:
            A[vj,vk] -= ar * np.sum(Q[:,0])
            A[vk,vj] = A[vj,vk]
        if ii[j] & ii[l]:
            A[vj,vl] -= ar * np.sum(Q[:,1])
            A[vl,vj] = A[vj,vl]
        if ii[k] & ii[l]:
            A[vk,vl] += ar * Q[0,1]
            A[vl,vk] = A[vk,vl]

    print "  solving ..."
    uh = np.zeros(Npts)
    uh[ii] = solve(A,b)
    return uh, inside

if __name__ == '__main__':
    ex_disc()
    show()
