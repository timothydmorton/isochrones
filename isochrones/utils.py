from __future__ import print_function, division

import requests
import os
import logging

import numpy as np
import numba as nb
from math import log10


def band_pairs(bands):
    return [(bands[i], bands[-1]) for i in range(len(bands)-1)]

def download_file(url, path=None, clobber=False):
    """
    thanks to: https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py

    path : str
        local path to download to.
    """
    if path is None:
        local_filename = os.path.join(directory, url.split('/')[-1])
    else:
        local_filename = path

    if os.path.exists(local_filename) and not clobber:
        logging.info('{} exists; not downloading.'.format(local_filename))
        return local_filename

    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename


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

@nb.jit(nopython=True)
def fast_addmags(mags):
    """
    mags is list of magnitudes
    """
    tot = 0
    for mag in mags:
        tot += 10**(-0.4 * mag)
    return -2.5*log10(tot)


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


@nb.jit(nopython=True)
def trapz(y, x):
    n = len(y)
    tot = 0

    for i in range(n-1):
        i2 = i + 1
        tot = tot + 0.5 * (y[i] + y[i2]) * (x[i2] - x[i])

    return tot


@nb.jit(nopython=True)
def polyval(p, x):
    N = len(p)
    result = 0
    for i in range(N):
        result += p[i] * x**(N - 1 - i)
    return result
