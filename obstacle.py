import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, tri

from py_distmesh2d import *
from meshtools import plotmesh, fixmesh, bdyrefine
from poisson import poisson

bbox = [[-1, 1], [-1, 1]]

def fd_disc(pts):
    return dcircle(pts, 0.0, 0.0, 1.0)

def f_ex(pts):
    return 0.0

def psi_ex(pts):
    return -np.sum((3.0*pts)**4.0,axis=1)+1.0

def obscircle(h0):
    print "  meshing and fixing mesh ..."
    p1, t1 = distmesh2d(fd_disc, huniform, h0, bbox, [])
    p1a, t1a = fixmesh(p1,t1)
    print "     ... original mesh has %d nodes" % np.shape(p1a)[0]

    print "  refining mesh once ..."
    p2, t2, e, ind = bdyrefine(p1a,t1a,fd_disc,h0)
    print "     ... first refined mesh has %d nodes" % np.shape(p2)[0]
    print "  refining mesh again ..."
    pts, mytri, e, ind = bdyrefine(p2,t2,fd_disc,h0)
    print "     ... second refined mesh has %d nodes" % np.shape(pts)[0]

    print "  solving ..."
    tol = 1.0e-6
    uh, ii, ierr = obstacle(psi_ex, f_ex, tol, fd_disc, h0, pts, mytri, \
                            announce=True)
    print "          ... %d iterations total to reach iteration tolerance %.2e" \
        % (len(ierr), tol)

    # FIXME: show coincidence set at black dots?
    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')
    ax.plot_trisurf(pts[:,0], pts[:,1], uh, cmap=cm.jet, linewidth=0.2)
    psi = np.maximum(psi_ex(pts),-1.0*np.ones(np.shape(pts)[0]))
    ax.plot_trisurf(pts[:,0], pts[:,1], psi, cmap=cm.Blues, linewidth=0.1)
    ax.set_xlim3d(-1.0,1.0)
    ax.set_ylim3d(-1.0,1.0)
    ax.set_zlim3d(-1.0,2.0)

    fig4 = plt.figure()
    plt.semilogy(np.array(range(len(ierr)))+1.0,ierr,'o-')
    plt.xlabel('j = iteration')
    plt.ylabel('max diff successive iterations')
    plt.grid(True)


def obstacle(psi,f_rhs,tol,f_dist,h0,pts,tri,*args,**kwargs):
    """Solve the obstacle problem
        u >= psi  in D,
        - Laplacian u = f  on  {(x,y) : u(x,y) > psi(x,y)},
        u = 0  on boundary of D,
    using triangulation described by  pts,tri.  Assumes psi <= 0 on boundary
    of D.  Returns:
        uh   = solution,
        ii   = indices of interior points, and
        ierr = error at each iteration.
    Example:  See obscircle().
    """
    announce = kwargs.get('announce',False)
    if announce:
        print "  obstacle: asking poisson() for linear system"
    # use poisson to get unconstrained stiffness, load
    uhpoisson, inside, AA, bb = poisson(f_ex,fd_disc,h0,pts,tri,announce=True,getsys=True)
    omega = 1.75     # found by trial and error
    maxiter = 300
    Npts = np.shape(pts)[0]            # = number of nodes
    geps = 0.001 * h0
    ii = (f_dist(pts, *args) < -geps)  # boolean array for interior nodes
    N = ii.sum()                       # = number of interior nodes
    UU = np.triu(AA,1)
    LL = np.tril(AA,-1)
    dd = np.diag(AA).copy()
    if any(dd == 0.0):
      print 'ERROR: stiffness matrix has zero on diagonal'
      return None
    # first guess is max(uhpoisson,psi)
    ps = np.maximum(psi(pts[ii]),np.zeros(N))  # FIXME: does not work well if f < 0?
    uold = np.maximum(uhpoisson[ii],ps)
    unew = uold.copy()
    omcomp = 1.0 - omega
    ierr = np.array([])
    # iterate: constrained point over-relaxation
    for l in range(maxiter+1):
        Ux = np.dot(UU,uold)
        for j in range(N):  # iterate over interior vertices
            # Gauss-Seidel idea:
            if j == 0:
                utmp = (bb[j] - Ux[j]) / dd[j]
            else:
                utmp = (bb[j] - np.dot(LL[j,:j],unew[:j]) - Ux[j]) / dd[j]
            # over-relax and project up to psi if needed
            unew[j] = np.maximum(omcomp * uold[j] + omega * utmp, ps[j])
        er = max(abs(unew-uold))
        ierr = np.append(ierr,er)
        uold = unew.copy()
        if er < tol:
            break
        if l == maxiter:
            print 'WARNING: max number of iterations reached'

    uh = uhpoisson.copy()
    uh[ii] = unew
    return uh, ii, ierr

if __name__ == '__main__':
    obscircle(0.2)
    plt.show()
