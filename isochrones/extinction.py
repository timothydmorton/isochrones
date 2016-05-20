import os, os.path, re

DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

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

