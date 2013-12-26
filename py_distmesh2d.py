#!/usr/bin/env python
import numpy as np
from numpy import sqrt, sum, vstack

__all__ = ["distmesh2d", "fixmesh", "longedgelist", "edgelist",
           "dcircle", "drectangle",
           "ddiff", "dintersect", "dunion",
           "huniform"]

try:
    from scipy.spatial import Delaunay
    def delaunay(pts):
        return Delaunay(pts).vertices
except:
    import matplotlib.delaunay as md
    def delaunay(pts):
        _, _, tri, _ = md.delaunay(pts[:,0], pts[:,1])
        return tri

def fixmesh(pts, tri):
    # find doubles
    doubles = []
    N = pts.shape[0]
    for i in xrange(N):
        for j in xrange(i+1,N):
            if np.linalg.norm(pts[i] - pts[j]) == 0:
                doubles.append(j)

    # remove doubles
    while len(doubles) > 0:
        j = doubles.pop()

        # remove a double
        pts = np.vstack([pts[0:j], pts[j+1:]])

        # update all triangles that reference points after the one removed
        for k in xrange(tri.shape[0]):
            for l in xrange(3):
                if tri[k, l] > j:
                    tri[k, l] -= 1

    # check (and fix) node order in triangles
    for k in xrange(tri.shape[0]):
        a = pts[tri[k, 0]]
        b = pts[tri[k, 1]]
        c = pts[tri[k, 2]]

        if np.cross(b - a, c - a) > 0:
            tri[k, 2], tri[k, 1] = tri[k, 1], tri[k, 2]

    return pts, tri

def longedgelist(pts, tri):
    """From a triangulation (pts,tri) create a 'long' list of edges, with
    redundancies.

    in:  pts = list of points (nx2)
         tri = list indices into pts defining a triangulation (Nx3)
    out: longe = a sorted list of edges with three entries per triangle.
    
    For each edge in each triangle, longe has a row of indices
         [ p0, p1, f, e ]
    where p0 < p1 = indices of the endpoints,
          f       = index into tri that this edge came from, and
          e       = {0,1,2} for which of the three edges it is.
    The shape of longe is  (m,4)  where m = 3N if N is the number of triangles.

    Sorting: longe is sorted in increasing order by p0, and then in increasing
    order by p1 if p0 are equal.

    Example:  If  tri[42,:] = [ 3, 9, 7 ]  then we will put these
    three entries in the long edge list:
        ...
        [ 3 7 42 2 ]
        [ 3 9 42 0 ]
        ...
        [ 7 9 42 1 ]
        ...
    """
    N = np.shape(tri)[0]
    longe = np.zeros((N*3,4),dtype=np.int32)
    for i in range(N):
        for j in range(3):
            M = tri[i,j]
            m = tri[i,np.mod(j+1,3)]
            if M < m:
                M, m = m, M
            eindex = 3*i + j
            longe[eindex] = [m, M, i, j] 
    # sort the long edge list so that p0 increases
    longe = longe[longe[:,0].argsort(),]
    # for equal p0, sort so that p1 increases
    for i in range(np.shape(pts)[0]):
        tmp = longe[longe[:,0]==i,:]
        if len(tmp)>0:   # in some cases p0 may not be a first point value
            longe[longe[:,0]==i] = tmp[tmp[:,1].argsort(),]
    return longe

def edgelist(pts, tri):
    """From a triangulation (pts,tri) extract a list of unique edges.  Also get
    a list of the triangles by their edges.
    in:  pts = list of points (nx2)
         tri = list indices into pts defining a triangulation (Nx3)
    out: e   = for each edge a list of indices into p defining the
               endpoints of the edge (mx2)
         te  = for each triangle in the triangulation a list of indices
               into e of the associated edges (Nx3)
    """
    longe = longedgelist(pts,tri)
    tmp = longe[:,[0,1]]
    d = ( np.sum(abs(np.diff(tmp,axis=0)),axis=1) != 0 ) # Tag unique elements
    d = np.concatenate(([True],d))
    e = longe[d][:,[0,1]]      # extract unique elements

    te = np.zeros(np.shape(tri),dtype=np.int32)
    ecum = np.cumsum(d) - 1
    for k in range(np.shape(longe)[0]):
       te[longe[k,2],longe[k,3]] = ecum[k]
    return e, te

def distmesh2d(fd, fh, h0, bbox, pfix, *args):
    """A re-implementation of the MATLAB distmesh2d function by Persson and Strang.

    See P.-O. Persson, G. Strang, A Simple Mesh Generator in MATLAB.
    SIAM Review, Volume 46 (2), pp. 329-345, June 2004

    and http://persson.berkeley.edu/distmesh/

    Parameters:
    ==========

    fd: a signed distance function, negative inside the domain
    fh: a triangle size function
    bbox: bounding box, [[x_min, x_max], [y_min, y_max]]
    pfix: fixed points, [[x1, y1], [x2, y2], ...]

    Extra arguments are passed to fd and fh.

    Returns
    =======

    p: list of points
    t: list of triangles (list of triples of indices in p)
    """
    # parameters
    dptol = 0.001; ttol = 0.1; Fscale = 1.2; deltat = 0.2;
    geps = 0.001 * h0; deps = sqrt(np.finfo(float).eps) * h0

    # create the initial point distribution:
    x, y = np.meshgrid(np.arange(bbox[0][0], bbox[0][1], h0),
                       np.arange(bbox[1][0], bbox[1][1], h0 * sqrt(3) / 2))

    x[1::2,:] += h0 / 2

    p = np.array((x.flatten(), y.flatten())).T

    # discard exterior points
    p = p[fd(p, *args) < geps]
    r0 = 1.0 / fh(p, *args)**2
    selection = np.random.rand(p.shape[0], 1) < r0 / r0.max()
    p = p[selection[:,0]]

    # add fixed points:
    if len(pfix) > 0:
        p = np.vstack((pfix, p))

    pold = np.zeros_like(p); pold[:] = np.inf
    Ftot = np.zeros_like(p)

    def triangulate(pts):
        """
        Compute the Delaunay triangulation and remove trianges with
        centroids outside the domain.
        """
        tri = np.sort(delaunay(pts), axis=1)
        pmid = sum(pts[tri], 1) / 3
        return tri[fd(pmid, *args) < -geps]

    while True:
        # check if it is time to re-compute the triangulation
        if sqrt(sum((p - pold)**2, 1)).max() > ttol:
            pold[:] = p[:]
            t = triangulate(p)
            # find unique edges of trianges
            bars = t[:, [[0,1], [1,2], [0,2]]].reshape((-1, 2))
            bars = np.unique(bars.view("i,i")).view("i").reshape((-1,2))

        barvec = p[bars[:,0]] - p[bars[:,1]]
        L = sqrt(sum(barvec**2, 1)).reshape((-1,1))
        hbars = fh((p[bars[:,0]] + p[bars[:,1]]) / 2.0, *args).reshape((-1,1))
        L0 = hbars * Fscale * sqrt(sum(L**2) / sum(hbars**2))

        # Compute forces for each bar:
        F = np.maximum(L0 - L, 0)
        Fvec = F * (barvec / L)

        # Sum to get total forces for each point:
        Ftot[:] = 0
        for j in xrange(bars.shape[0]):
            Ftot[bars[j]] += [Fvec[j], -Fvec[j]]

        # zero out forces at fixed points:
        Ftot[0:len(pfix), :] = 0.0

        # update point locations:
        p += deltat * Ftot

        # find points that ended up outside the domain and project them onto the boundary:
        d = fd(p, *args); ix = d > 0
        dgradx = (fd(vstack((p[ix,0] + deps, p[ix,1])).T, *args)        - d[ix]) / deps
        dgrady = (fd(vstack((p[ix,0],        p[ix,1] + deps)).T, *args) - d[ix]) / deps
        p[ix] -= vstack((d[ix] * dgradx, d[ix] * dgrady)).T

        # the stopping criterion:
        if (sqrt(sum((deltat * Ftot[d < -geps])**2, 1)) / h0).max() < dptol:
            break

    return p, triangulate(p)

def dcircle(pts, xc, yc, r):
    "Distance function for the circle centered at (xc, yc)."
    return sqrt((pts[:,0] - xc)**2 + (pts[:,1] - yc)**2) - r

def drectangle(pts, x1, x2, y1, y2):
    "Distance function for the rectangle (x1, x2) * (y1, y2)."
    return -np.minimum(np.minimum(np.minimum(-y1+pts[:,1], y2-pts[:,1]),
                                  -x1+pts[:,0]), x2-pts[:,0])

def ddiff(d1, d2):
    "Distance function for the difference of two sets."
    return np.maximum(d1, -d2)

def dintersect(d1, d2):
    "Distance function for the intersection of two sets."
    return np.maximum(d1, d2)

def dunion(d1, d2):
    "Distance function for the union of two sets."
    return np.minimum(d1, d2)

def huniform(pts, *args):
    "Triangle size function giving a near-uniform mesh."
    return np.ones((pts.shape[0], 1))

