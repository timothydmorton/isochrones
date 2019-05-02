import numba as nb
import numpy as np


@nb.jit(nopython=True)
def eep_fn(x, p5, p4, p3, p2, p1, p0, A, x0, tau, order=5):
    """Polynomial + exponential to approximate eep(age) for given track
    """
    if order < 5:
        p5 = 0
        if order < 4:
            p4 = 0
            if order < 3:
                p3 = 0
                if order < 2:
                    p2 = 0

    return p5*x**5 + p4*x**4 + p3*x**3 + p2*x**2 + p1*x + p0 + A*np.exp((x - x0)/tau)


@nb.jit(nopython=True)
def eep_jac(x, p5, p4, p3, p2, p1, p0, A, x0, tau, order=5):
    """Jacobian of eep_fn
    """
    if order < 5:
        p5 = 0
        if order < 4:
            p4 = 0
            if order < 3:
                p3 = 0
                if order < 2:
                    p2 = 0

    n = len(x)

    result = np.empty((n, 9), dtype=nb.float64)
    for i in range(n):
        xi = x[i]
        result[i, 0] = xi*xi*xi*xi*xi
        result[i, 1] = xi*xi*xi*xi
        result[i, 2] = xi*xi*xi
        result[i, 3] = xi*xi
        result[i, 4] = xi
        result[i, 5] = 0.
        result[i, 6] = np.exp((xi - x0)/tau)
        result[i, 7] = -1./tau * A * np.exp((xi - x0) / tau)
        result[i, 8] = -1./tau**2 * (xi - x0) * A * np.exp((xi - x0) / tau)
    return result

def eep_fn_p0(ages, eeps, order=5):
    """seems to work well
    """
    m = eeps < 300
    p1, p0 = np.polyfit(ages[m], eeps[m], 1)
    return [0, 0, 0, 0, p1, p0, 1, ages.max()-0.3, 0.05]

def fit_section_poly(age, eep, a, b, order=3):
    m = (a < eep) & (eep < b)
    if m.sum() < order + 1:
        raise ValueError
    return np.polyfit(age[m], eep[m], order)
