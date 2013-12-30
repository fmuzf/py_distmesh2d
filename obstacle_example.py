import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, tri

from py_distmesh2d import *
from meshtools import plotmesh, fixmesh, bdyrefine
from obstacle import obstacle

def u_exact_sphere(pts):
    '''Exact solution for radially-symmetric membrane over spherical obstacle
    can be found.  Here the sphere has radius 1/2 so it has equation
        z = sqrt(0.25 - r^2)
    in polar (cylindrical) coordinates.  The free boundary is at r=a such that:
        u(r=1) = 0
        u(r=a) = sqrt(0.25 - a^2)
        u_r(r=a) = -a / sqrt(0.25 - a^2)
    On the other hand a solution u(r) which is radially-symmetric and which
    solves laplace's equation solves
        u_rr - (1/r) u_r = 0
    so
        u(r) = b log(r) + c.
    We find c=0 and that
        b = - a^2 / sqrt(0.25 - a^2)
    and that
        sqrt(0.25 - a^2) = b log(a)
        - a / sqrt(0.25 - a^2) = b / a.
    This gives a single nonlinear equation to solve for a, namely
        a^2 (log(a) - 1) + 0.25 = 0.
    This method solves that equation using Newton's method, with a = 0.35
    as initial guess, and then computes u(x,y) at the given points.'''
    def G(a):
        return a*a*(np.log(a)-1) + 0.25
    def dG(a):
        return 2.0*a*(np.log(a)-1) + a
    a = 0.35
    for j in range(5):
        a = a - G(a) / dG(a)
    b = - a**2 / np.sqrt(0.25 - a**2)
    print '  for exact solution: a = %.14f, b = %.14f' % (a,b)
    r = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    return b * np.log(r)

bbox = [[-1, 1], [-1, 1]]

def fd_disc(pts):
    return dcircle(pts, 0.0, 0.0, 1.0)

def f_ex(pts):
    return 0.0

def psi_fourth(pts):
    return -np.sum((3.0*pts)**4.0,axis=1)+1.0

def psi_sphere(pts):
    return np.sqrt(np.maximum(0.0,0.25 - pts[:,0]**2 - pts[:,1]**2))

def obscircle(h0,psifcn):
    print "  meshing and fixing mesh ..."
    p1, t1 = distmesh2d(fd_disc, huniform, h0, bbox, [])
    p1a, t1a = fixmesh(p1,t1)
    print "     ... original mesh has %d nodes" % np.shape(p1a)[0]

    print "  refining mesh once ..."
    p2, t2, e, ind = bdyrefine(p1a,t1a,fd_disc,h0)
    print "     ... first refined mesh has %d nodes" % np.shape(p2)[0]
    print "  refining mesh again ..."
    p3, t3, e, ind = bdyrefine(p2,t2,fd_disc,h0)
    print "     ... second refined mesh has %d nodes" % np.shape(p3)[0]
    print "  refining mesh again ..."
    pts, mytri, e, ind = bdyrefine(p3,t3,fd_disc,h0)
    print "     ... third refined mesh has %d nodes" % np.shape(pts)[0]

    print "  solving ..."
    tol = 1.0e-6
    uh, ii, ierr = obstacle(psifcn, f_ex, tol, fd_disc, h0, pts, mytri, \
                            announce=True)
    print "  obstacle: %d iterations total to reach iteration tolerance %.2e" \
        % (len(ierr), tol)

    psi = np.maximum(psifcn(pts),-1.0*np.ones(np.shape(pts)[0]))
    fig1 = plt.figure()
    plotmesh(pts, mytri)
    coin = (uh==psi)   # coincidence set
    plt.plot(pts[coin,0],pts[coin,1],'ro',markersize=8.0)

    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')
    #ax.plot_trisurf(pts[:,0], pts[:,1], psi, cmap=cm.Blues, linewidth=0.1)
    ax.plot_trisurf(pts[:,0], pts[:,1], uh, cmap=cm.jet, linewidth=0.2)
    ax.set_xlim3d(-1.0,1.0)
    ax.set_ylim3d(-1.0,1.0)
    ax.set_zlim3d(-0.25,1.25)

    fig4 = plt.figure()
    plt.semilogy(np.array(range(len(ierr)))+1.0,ierr,'o-')
    plt.xlabel('j = iteration')
    plt.ylabel('max diff successive iterations')
    plt.grid(True)

    if psifcn == psi_sphere:
        print 'for sphere obstacle, exact solution is known:'
        print '  (away from the obstacle:  u(r) = b log(r) )'
        apart = (uh > psi)
        print '  max norm error = %.3e' % max(abs(uh[apart] - u_exact_sphere(pts[apart])))

if __name__ == '__main__':
    #obscircle(0.2,psi_fourth)
    obscircle(0.2,psi_sphere)  # smallest that works:  0=0.75*0.125
    plt.show()
