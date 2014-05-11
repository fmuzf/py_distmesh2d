from pylab import figure, show
import numpy as np
from distmesh2d import *
from distmesh2d.meshtools import plotmesh

def example1(pts):
    return dcircle(pts, 0, 0, 1)

def example2(pts):
    return ddiff(dcircle(pts, 0, 0, 0.7), dcircle(pts, 0, 0, 0.3))

def example3(pts):
    return ddiff(drectangle(pts, -1, 1, -1, 1), dcircle(pts, 0, 0, 0.4))

def example3_h(pts):
    return np.minimum(4*np.sqrt(sum(pts**2, 1)) - 1, 2)

def example3_online(pts):
    return ddiff(drectangle(pts, -1, 1, -1, 1), dcircle(pts, 0, 0, 0.5))

def example3_online_h(pts):
    return 0.05 + 0.3 * dcircle(pts, 0, 0, 0.5)

def annulus_h(pts):
    return 0.04 + 0.15 * dcircle(pts, 0, 0, 0.3)

def star(pts):
    return dunion(dintersect(dcircle(pts, np.sqrt(3), 0, 2), dcircle(pts, -np.sqrt(3), 0, 2)),
                  dintersect(dcircle(pts, 0, np.sqrt(3), 2), dcircle(pts, 0, -np.sqrt(3), 2)))

def circle_h(pts):
    return 0.1 - example1(pts)

bbox = [[-1, 1], [-1, 1]]
square = [[-1,-1], [-1,1], [1,-1], [1,1]]

# example 1a
def example_1a():
    figure()
    pts, tri = distmesh2d(example1, huniform, 0.4, bbox, [])
    plotmesh(pts, tri)
    show()

# example 1b
def example_1b():
    figure()
    pts, tri = distmesh2d(example1, huniform, 0.2, bbox, [])
    plotmesh(pts, tri)
    show()

# example 1c
def example_1c():
    figure()
    pts, tri = distmesh2d(example1, huniform, 0.1, bbox, [])
    plotmesh(pts, tri)
    show()

# example 2
def example_2():
    figure()
    pts, tri = distmesh2d(example2, huniform, 0.1, bbox, [])
    plotmesh(pts, tri)
    show()

# example 3a
def example_3a():
    figure()
    pts, tri = distmesh2d(example3, huniform, 0.15, bbox, square)
    plotmesh(pts, tri, example3(pts))
    show()

# example 3b
def example_3b():
    figure()
    pts, tri = distmesh2d(example3, example3_h, 0.035, bbox, square)
    plotmesh(pts, tri)
    show()

# example (current online version)
def example_3_online():
    figure()
    pts, tri = distmesh2d(example3_online, example3_online_h, 0.02, bbox, square)
    plotmesh(pts, tri)
    show()

# annulus, non-uniform
def annulus():
    figure()
    pts, tri = distmesh2d(example2, annulus_h, 0.04, bbox, square)
    plotmesh(pts, tri)
    show()

# a "star" built using circles
def star_mesh():
    figure()
    # fake the corners:
    pfix = [[0.25, 0.25], [-0.25, 0.25], [-0.25, -0.25], [0.25, -0.25]]
    pts, tri = distmesh2d(star, huniform, 0.1, bbox, pfix)
    plotmesh(pts, tri)
    show()

# a circle, finer mesh near the boundary
def circle_nonuniform():
    figure()
    # fake the corners:
    pts, tri = distmesh2d(example1, circle_h, 0.1, bbox, [])
    plotmesh(pts, tri)
    show()

def ell():
    """L-shaped domain from 'Finite Elements and Fast Iterative Solvers'
    by Elman, Silvester, and Wathen."""

    pfix = [[1,1], [1, -1], [0, -1], [0, 0], [-1, 0], [-1, 1]]

    def d(pts):
        return ddiff(drectangle(pts, -1, 1, -1, 1), drectangle(pts, -2, 0, -2, 0))

    figure()
    pts, tri = distmesh2d(d, huniform, 0.25, bbox, pfix)
    plotmesh(pts, tri)
    show()

if __name__ == '__main__':
    example_3_online()
