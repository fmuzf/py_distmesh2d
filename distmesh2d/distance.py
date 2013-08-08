import numpy as _np

__all__ = ["dcircle", "drectangle", "ddiff", "dintersect", "dunion", "huniform"]


def dcircle(pts, xc, yc, r):
    "Distance function for the circle centered at (xc, yc)."
    return _np.sqrt((pts[:,0] - xc)**2 + (pts[:,1] - yc)**2) - r


def drectangle(pts, x1, x2, y1, y2):
    "Distance function for the rectangle (x1, x2) * (y1, y2)."
    return -_np.minimum(_np.minimum(_np.minimum(-y1+pts[:,1], y2-pts[:,1]),
                                  -x1+pts[:,0]), x2-pts[:,0])

def ddiff(d1, d2):
    "Distance function for the difference of two sets."
    return _np.maximum(d1, -d2)


def dintersect(d1, d2):
    "Distance function for the intersection of two sets."
    return _np.maximum(d1, d2)


def dunion(d1, d2):
    "Distance function for the union of two sets."
    return _np.minimum(d1, d2)


def huniform(pts, *args):
    "Triangle size function giving a near-uniform mesh."
    return _np.ones((pts.shape[0], 1))
