import numpy as np
from poisson import poisson

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
        print "  obstacle: asking poisson() for linear system and unconstrained soln ..."
    # use poisson to get unconstrained stiffness, load
    uhpoisson, inside, AA, bb = poisson(f_rhs,f_dist,h0,pts,tri,announce=True,getsys=True)
    omega = 1.75     # found by trial and error
    maxiter = 500
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
    # construct solution by filling interior values and boundary values
    uh = uhpoisson.copy()
    uh[ii] = unew
    return uh, ii, ierr

