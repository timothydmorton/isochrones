from .schlafly.extcurve_s16 import extcurve

from pyextinction import ExtinctionLaw as _ExtinctionLaw
from pyextinction import Fitzpatrick99 as _Fitzpatrick99
from pyextinction import Gordon03_SMCBar as _Gordon03_SMCBar
from pyextinction import Cardelli as _Cardelli
from pyextinction import Calzetti as _Calzetti

class ExtinctionLaw(_ExtinctionLaw):
    """Customization of ExtinctionLaw class from pyextinction

    Makes the Rv (or Rv-like) parameter a class attribute,
    and fixes AV to 1., because we always want a scale factor.
    """
    param_name = 'Rv'
    param_default = None

    def __init__(self, parameter=None):
        if parameter is None:
            parameter = self.param_default
        self.parameter = parameter

    @property
    def Rv(self):
        return self.parameter
    
    def function(self, lamb, **kwargs):
        kwargs = {'Av':1., self.param_name:self.parameter}
        return super(ExtinctionLaw, self).function(lamb, **kwargs)

class Fitzpatrick99(ExtinctionLaw, _Fitzpatrick99):
    name = 'Fitzpatrick'
    param_default = 3.1

class Gordon03_SMCBar(ExtinctionLaw, _Gordon03_SMCBar):
    name = 'Gordon'
    param_default = 2.74

class Cardelli(ExtinctionLaw, _Cardelli):
    name = 'Cardelli'
    param_default = 3.1

class Calzetti(ExtinctionLaw, _Calzetti):
    name = 'Calzetti'
    param_default = 4.05

class Schlafly(ExtinctionLaw):
    name = 'Schlafly'
    param_name = 'x'
    param_default = 0.

    def __init__(self, **kwargs):
        super(Schlafly, self).__init__(**kwargs)
        self.extcurve = extcurve(self.parameter)

    @property
    def x(self):
        return self.parameter

    def function(self, lamb, **kwargs):
        """lamb in Angstrom; returns A_lambda
        """
        return self.extcurve(lamb)

def get_extinction_curve(name, **kwargs):
    if name.lower().startswith('schl'):
        return Schlafly(**kwargs)
    elif name.lower().startswith('fitz'):
        return Fitzpatrick99(**kwargs)
    elif name.lower().startswith('gord'):
        return Gordon03_SMCBar(**kwargs)
    elif name.lower().startswith('card'):
        return Cardelli(**kwargs)
    elif name.lower().startswith('calz'):
        return Calzetti(**kwargs)
    else:
        raise ValueError('{} not a recognized extinction curve!'.format(name))