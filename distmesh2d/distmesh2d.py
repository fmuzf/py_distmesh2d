#!/usr/bin/env python
import numpy as _np

__all__ = ["distmesh2d", "fixmesh"]

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
            if _np.linalg.norm(pts[i] - pts[j]) == 0:
                doubles.append(j)

    # remove doubles
    while len(doubles) > 0:
        j = doubles.pop()

        # remove a double
        pts = _np.vstack([pts[0:j], pts[j+1:]])

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

        if _np.cross(b - a, c - a) > 0:
            tri[k, 2], tri[k, 1] = tri[k, 1], tri[k, 2]

    return pts, tri

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
    geps = 0.001 * h0; deps = _np.sqrt(_np.finfo(float).eps) * h0

    # create the initial point distribution:
    x, y = _np.meshgrid(_np.arange(bbox[0][0], bbox[0][1], h0),
                       _np.arange(bbox[1][0], bbox[1][1], h0 * _np.sqrt(3) / 2))

    x[1::2,:] += h0 / 2

    p = _np.array((x.flatten(), y.flatten())).T

    # discard exterior points
    p = p[fd(p, *args) < geps]
    r0 = 1.0 / fh(p, *args)**2
    selection = _np.random.rand(p.shape[0], 1) < r0 / r0.max()
    p = p[selection[:,0]]

    # add fixed points:
    if len(pfix) > 0:
        p = _np.vstack((pfix, p))

    pold = _np.zeros_like(p); pold[:] = _np.inf
    Ftot = _np.zeros_like(p)

    def triangulate(pts):
        """
        Compute the Delaunay triangulation and remove trianges with
        centroids outside the domain.
        """
        tri = _np.sort(delaunay(pts), axis=1)
        pmid = _np.sum(pts[tri], 1) / 3
        return tri[fd(pmid, *args) < -geps]

    while True:
        # check if it is time to re-compute the triangulation
        if _np.sqrt(_np.sum((p - pold)**2, 1)).max() > ttol:
            pold[:] = p[:]
            t = triangulate(p)
            # find unique edges of trianges
            bars = t[:, [[0,1], [1,2], [0,2]]].reshape((-1, 2))
            bars = _np.unique(bars.view("i,i")).view("i").reshape((-1,2))

        barvec = p[bars[:,0]] - p[bars[:,1]]
        L = _np.sqrt(_np.sum(barvec**2, 1)).reshape((-1,1))
        hbars = fh((p[bars[:,0]] + p[bars[:,1]]) / 2.0, *args).reshape((-1,1))
        L0 = hbars * Fscale * _np.sqrt(_np.sum(L**2) / _np.sum(hbars**2))

        # Compute forces for each bar:
        F = _np.maximum(L0 - L, 0)
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
        dgradx = (fd(_np.vstack((p[ix,0] + deps, p[ix,1])).T, *args)        - d[ix]) / deps
        dgrady = (fd(_np.vstack((p[ix,0],        p[ix,1] + deps)).T, *args) - d[ix]) / deps
        p[ix] -= _np.vstack((d[ix] * dgradx, d[ix] * dgrady)).T

        # the stopping criterion:
        if (_np.sqrt(_np.sum((deltat * Ftot[d < -geps])**2, 1)) / h0).max() < dptol:
            break

    return p, triangulate(p)
