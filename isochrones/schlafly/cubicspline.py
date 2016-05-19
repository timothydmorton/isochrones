# Based on Numerical Recipes
import numpy
from scipy.linalg import solve_banded
import pdb

def splint(spl, x):
    npts = len(spl.x)
    lo = numpy.searchsorted(spl.x, x)-1
    lo = numpy.clip(lo, 0, npts-2)
    hi = lo + 1
    dx = spl.x[hi] - spl.x[lo]
    a = (spl.x[hi] - x)/dx
    b = (x-spl.x[lo])/dx
    y = (a*spl.y[lo]+b*spl.y[hi]+
         ((a**3-a)*spl.y2[lo]+(b**3-b)*spl.y2[hi])*dx**2./6.)
    return y

class CubicSpline:
    def __init__(self, x, y, yp=None):
        npts = len(x)
        mat = numpy.zeros((3, npts))
        # enforce continuity of 1st derivatives
        mat[1,1:-1] = (x[2:  ]-x[0:-2])/3.
        mat[2,0:-2] = (x[1:-1]-x[0:-2])/6.
        mat[0,2:  ] = (x[2:  ]-x[1:-1])/6.
        bb = numpy.zeros(npts)
        bb[1:-1] = ((y[2:  ]-y[1:-1])/(x[2:  ]-x[1:-1]) -
                    (y[1:-1]-y[0:-2])/(x[1:-1]-x[0:-2]))
        if yp is None: # natural cubic spline
            mat[1,0] = 1.
            mat[1,-1] = 1.
            bb[0] = 0.
            bb[-1] = 0.
        elif yp == '3d=0':
            mat[1, 0] = -1./(x[1]-x[0])
            mat[0, 1] =  1./(x[1]-x[0])
            mat[1,-1] =  1./(x[-2]-x[-1])
            mat[2,-2] = -1./(x[-2]-x[-1])
            bb[ 0] = 0.
            bb[-1] = 0.
        else:
            mat[1, 0] = -1./3.*(x[1]-x[0])
            mat[0, 1] = -1./6.*(x[1]-x[0])
            mat[2,-2] =  1./6.*(x[-1]-x[-2])
            mat[1,-1] =  1./3.*(x[-1]-x[-2])
            bb[ 0] = yp[0]-1.*(y[ 1]-y[ 0])/(x[ 1]-x[ 0])
            bb[-1] = yp[1]-1.*(y[-1]-y[-2])/(x[-1]-x[-2])
        y2 = solve_banded((1,1), mat, bb)
        self.x, self.y, self.y2 = (x, y, y2)
    def __call__(self, x):
        return splint(self, x)
    
    
    
    
