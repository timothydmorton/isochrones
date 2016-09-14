from __future__ import print_function, division
import numpy as np

def addmags(*mags):
    """
    mags is either list of magnitudes or list of (mag, err) pairs
    """
    tot = 0
    uncs = []
    for mag in mags:
        try:
            tot += 10**(-0.4*mag)
        except:
            m, dm = mag
            f = 10**(-0.4*m)
            tot += f
            unc = f * (1 - 10**(-0.4*dm))
            uncs.append(unc)
    
    totmag = -2.5*np.log10(tot)
    if len(uncs) > 0:
        f_unc = np.sqrt(np.array([u**2 for u in uncs]).sum())
        return totmag, -2.5*np.log10(1 - f_unc/tot)
    else:
        return totmag 
    
    
def distance(pos0, pos1):
    """distance between two positions defined by (separation, PA)
    """
    r0, pa0 = pos0
    #logging.debug('r0={}, pa0={} (from {})'.format(r0, pa0, self))
    ra0 = r0*np.sin(pa0*np.pi/180)
    dec0 = r0*np.cos(pa0*np.pi/180)
    
    r1, pa1 = pos1
    #logging.debug('r1={}, pa1={} (from {})'.format(r0, pa0, other))
    ra1 = r1*np.sin(pa1*np.pi/180)
    dec1 = r1*np.cos(pa1*np.pi/180)

    dra = (ra1 - ra0)
    ddec = (dec1 - dec0)
    return np.sqrt(dra**2 + ddec**2)