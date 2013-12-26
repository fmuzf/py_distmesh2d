from py_distmesh2d import *
import numpy as np
from numpy.linalg import det, inv, solve


def poisson(f_rhs,f_dist,h0,pts,tri,*args,**kwargs):
    """Solve Poisson's equation on a domain D by the FE method:
         - Laplacian u = f_rhs
    on D and
         u = 0
    on boundary of D.   The right-hand side is  f = f_rhs(pts,*args).
    We use a triangulation described by points pts, triangles tri,
    a mesh scale h0, and a signed distance function f_dist(pts,*args);
    see py_distmesh2d.py.  Returns
       uh     = approximate solution value at pts
       inside = index of interior point (or -1 if not interior)
    See fem_examples.py for examples.
    """
    announce = kwargs.get('announce',False)
    geps = 0.001 * h0
    ii = (f_dist(pts, *args) < -geps)      # boolean array for interior nodes
    Npts = np.shape(pts)[0]     # = number of nodes
    N = ii.sum()                # = number of *interior* nodes
    if announce:
        print "  poisson: assembling on mesh with  %d  nodes and  %d  interior nodes ..." \
            % (Npts,N)
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
        ff[j] = f_rhs(pts[j], *args)
    # loop over triangles to set up stiffness matrix A and load vector b
    # NOTE: not using sparse matrices at all
    A = np.zeros((N,N))
    b = np.zeros(N)
    for n in range(np.shape(tri)[0]):        # loop over triangles
        # indices, coordinates, and Jacobian of triangle
        j, k, l = tri[n,:]
        vj = inside[j]
        vk = inside[k]
        vl = inside[l]
        Jac = np.array([[ pts[k,0] - pts[j,0], pts[l,0] - pts[j,0] ],
                        [ pts[k,1] - pts[j,1], pts[l,1] - pts[j,1] ]])
        ar = abs(det(Jac))/2.0
        C = ar/12.0
        Q = inv(np.dot(Jac.transpose(),Jac))
        fT = np.array([ff[j], ff[k], ff[l]])
        # add triangle's contribution to linear system  A x = b
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
    if announce:
        print "  poisson: solving linear system  A uh = b  with  N = %d  unknowns ..." % N
    uh = np.zeros(Npts)
    uh[ii] = solve(A,b)
    return uh, inside

