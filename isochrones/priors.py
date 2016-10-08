from __future__ import print_function, division
import numpy as np

def age_prior(age, bounds=(9,10.15)):
    """
    Uniform true age prior; where 'age' is actually log(age)
    """
    minage, maxage = bounds
    if age < minage or age > maxage:
        return 0
    return np.log(10) * 10**age / (10**maxage - 10**minage)

def distance_prior(distance, bounds=(0,3000)):
    """
    Distance prior ~ d^2
    """
    min_distance, max_distance = bounds
    if distance <= min_distance or distance > max_distance:
        return 0
    return 3/max_distance**3 * distance**2

def AV_prior(AV, bounds=(0,1.)):
    if AV < bounds[0] or AV > bounds[1]:
        return 0
    return 1./bounds[1]

def q_prior(q, m=1, gamma=0.3, bounds=(0.1,1)):
    """Default prior on mass ratio q ~ q^gamma
    """
    qmin, qmax = bounds
    try:
        if q < qmin or q > qmax:
            return 0
    except ValueError:
        pass

    C = 1/(1/(gamma+1)*(1 - qmin**(gamma+1)))
    result = C*q**gamma
    if np.size(result) > 1:
        result[(q < qmin) | (q > qmax)] = 0
    return result

def salpeter_prior(m,alpha=-2.35, bounds=(0.1,10)):
    minmass, maxmass = bounds
    C = (1+alpha)/(maxmass**(1+alpha)-minmass**(1+alpha))
    try:
        if m < minmass or m > maxmass:
            return 0
        else:   
            return C*m**(alpha)
    except ValueError:
        result = C*m**(alpha)
        result[(m < minmass) | (m > maxmass)] = 0
        return result

def feh_prior(feh, bounds=None, halo_fraction=0.001):
    """feh PDF based on local SDSS distribution
    
    From Jo Bovy:
    https://github.com/jobovy/apogee/blob/master/apogee/util/__init__.py#L3
    2D gaussian fit based on Casagrande (2011)
    """
    disk_norm = 2.5066282746310007 # integral of the below from -np.inf to np.inf
    disk_fehdist = 1./disk_norm * (0.8/0.15*np.exp(-0.5*(feh-0.016)**2./0.15**2.)\
                   +0.2/0.22*np.exp(-0.5*(feh+0.15)**2./0.22**2.))

    halo_mu, halo_sig = -1.5, 0.4
    halo_fehdist = 1./np.sqrt(2*np.pi*halo_sig**2) * np.exp(-0.5*(feh-halo_mu)**2/halo_sig**2)

    return halo_fraction * halo_fehdist + (1 - halo_fraction)*disk_fehdist
