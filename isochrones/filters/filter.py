from __future__ import print_function, division

import numpy as np
from ..extinction import extcurve

class Filter(object):
    lam_col = 0
    throughput_col = 1
    def __init__(self, filename, x=0):
        self.filename = filename
        self.x = x

        self._data = None
        self._extcurve = None
        self._extinction = None

    @property
    def data(self):
        if self._data is None:
            self._data = np.loadtxt(self.filename, usecols=(self.lam_col, 
                                                            self.throughput_col))
        return self._data
    

    @property
    def lam(self):
        return self.data[:, self.lam_col]
    
    @property
    def throughput(self):
        return self.data[:, self.throughput_col]

    @property
    def extcurve(self):
        if self._extcurve is None:
            self._extcurve = extcurve(self.x)
        return self._extcurve

    @property
    def extinction(self):
        if self._extinction is None:
            l, t, e = self.lam, self.throughput, self.extcurve(self.lam)
            self._extinction = np.trapz(t*e, l) / np.trapz(t, l)
        return self._extinction
