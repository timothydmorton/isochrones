import os, os.path, re

from ..config import on_rtd

if not on_rtd:
    DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

    # Wavelength dependence of extinction from Schlafly+ (2016)
    #  http://e.schlaf.ly/apored/extcurve.html
    from .schlafly.extcurve_s16 import extcurve

    extcurve_0 = extcurve(0.)

    #Read data defining effective central wavelengths of filters
    FILTERFILE = os.path.join(DATADIR,'filters.txt')
    LAMBDA_EFF = {}
    for line in open(FILTERFILE,'r'):
        if re.match('#', line):
            continue
        line = line.split()
        LAMBDA_EFF[line[0]] = float(line[1])


    #Read data defining extinction in different bands (relative to A_V)
    EXTINCTIONFILE = '{}/extinction.txt'.format(DATADIR)
    EXTINCTION = dict()
    EXTINCTION5 = dict()
    for line in open(EXTINCTIONFILE,'r'):
        line = line.split()
        EXTINCTION[line[0]] = float(line[1])
        EXTINCTION5[line[0]] = float(line[2])

    EXTINCTION['kep'] = 0.85946
    EXTINCTION['V'] = 1.0
    EXTINCTION['Ks'] = EXTINCTION['K']
    EXTINCTION['Kepler'] = EXTINCTION['kep']

    from astropy.coordinates import SkyCoord
    from six.moves import urllib
    import re

def get_AV_infinity(ra,dec,frame='icrs'):
    """
    Gets the A_V exctinction at infinity for a given line of sight.

    Queries the NED database.

    :param ra,dec:
        Desired coordinates, in degrees.
    :param frame: (optional)
        Frame of input coordinates (e.g., ``'icrs', 'galactic'``)
    """
    coords = SkyCoord(ra,dec,unit='deg',frame=frame).transform_to('icrs')

    rah,ram,ras = coords.ra.hms
    decd,decm,decs = coords.dec.dms
    if decd > 0:
        decsign = '%2B'
    else:
        decsign = '%2D'
    url = 'http://ned.ipac.caltech.edu/cgi-bin/nph-calc?in_csys=Equatorial&in_equinox=J2000.0&obs_epoch=2010&lon='+'%i' % rah + \
        '%3A'+'%i' % ram + '%3A' + '%05.2f' % ras + '&lat=%s' % decsign + '%i' % abs(decd) + '%3A' + '%i' % abs(decm) + '%3A' + '%05.2f' % abs(decs) + \
        '&pa=0.0&out_csys=Equatorial&out_equinox=J2000.0'

    AV = None
    for line in urllib.request.urlopen(url).readlines():
        m = re.search(b'^Landolt V \(0.54\)\s+(\d+\.\d+)', line)
        if m:
            AV = (float(m.group(1)))
            break

    if AV is None:
        raise RuntimeError('AV query fails!  URL is {}'.format(url))

    return AV
