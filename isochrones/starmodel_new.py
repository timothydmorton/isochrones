from __future__ import print_function, division

from .observation import ObservationTree, Observation, Source

class StarModel(object):
    """

    :param ic: 
        :class:`Isochrone` object used to model star.

    :param obs: (optional)
        :class:`ObservationTree` object containing photometry information.
        If not provided, then one will be constructed from the provided
        keyword arguments (which must include at least one photometric
            bandpass).  This should only happen in the simplest case
        of a single star system---if multiple stars are detected
        in any of the observations being used, an :class:`ObservationTree`
        should be passed.

    :param N:
        Number of model stars to assign to each "leaf node" of the 
        :class:`ObservationTree`.    

    :param maxAV: (optional)
        Maximum allowed extinction (i.e. the extinction @ infinity in direction of star).  Default is 1.

    :param max_distance: (optional)
        Maximum allowed distance (pc).  Default is 3000.

    :param **kwargs:
            Keyword arguments must be properties of given isochrone, e.g., logg,
            feh, Teff, and/or magnitudes.  The values represent measurements of
            the star, and must be in (value,error) format. All such keyword
            arguments will be held in ``self.properties``.  ``parallax`` is
            also a valid property, and should be provided in miliarcseconds.        
    """
    def __init__(self, ic, obs=None, N=1, index=0,
                 maxAV=1., max_distance=3000.,
                 min_logg=None, name='', **kwargs):

        self.maxAV = maxAV
        self.max_distance = max_distance
        self.min_logg = None
        self.name = name
        self._ic = ic

        # If obs is not provided, build it
        if obs is None:
            self._build_obs(**kwargs)
        else:
            self.obs = obs

        self.obs.define_models(ic, N=N, index=index)
        self._add_properties(**kwargs)

        self._priors = {'mass':salpeter_prior,
                        'feh':local_fehdist,
                        'q':q_prior,
                        'age':age_prior,
                        'distance':distance_prior,
                        'AV':AV_prior}
        self._bounds = {'mass':None,
                        'feh':None,
                        'age':None,
                        'q':(0.1,1.0),
                        'distance':(0,self.max_distance),
                        'AV':(0,self.maxAV)}

    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic

    def bounds(self, prop):
        if self._bounds[prop] is not None:
            return self._bounds[prop]
        elif prop=='mass':
            self._bounds['mass'] = (self.ic.minmass,
                                    self.ic.maxmass)
        elif prop=='feh':
            self._bounds['feh'] = (self.ic.minfeh,
                                   self.ic.maxfeh)
        elif prop=='age':
            self._bounds['age'] = (self.ic.minage,
                                   self.ic.maxage)
        else:
            raise ValueError('Unknown property {}'.format(prop))

    def _build_obs(self, **kwargs):
        """
        Builds ObservationTree out of keyword arguments

        Ignores anything that is not a photometric bandpass.
        This should not be used if there are multiple stars observed.

        Creates self.obs
        """
        tree = ObservationTree()
        for k,v in kwargs.items():
            if k in self.ic.bands:
                if len(v) != 2:
                    logging.warning('Property {}={} ignored.'.format(k,v))
                    continue
                o = Observation('',k,99) #bogus resolution=99
                o.add_source(Source(v[0],v[1]))
                tree.add_observation(o)
        self.obs = tree

    def _add_properties(self, **kwargs):
        """
        Adds non-photometry properties to ObservationTree
        """
        for k,v in kwargs.items():
            if k=='parallax':
                self.obs.add_parallax(v)
            elif k not in self.ic.bands:
                par = {k:v}
                self.obs.add_spectroscopy(**par)

    def lnlike(self, *args, **kwargs):
        return self.obs.lnlike(*args, **kwargs)

    def lnprior(self, p):
        N = self.obs.Nstars
        i = 0
        lnp = 0
        for s in self.obs.systems:
            age, feh, dist, AV = p[i+N[s]:i+N[s]+4]
            for prop, val in zip(['age','feh','distance','AV'],
                                 [age, feh, dist, AV])
                lnp += np.log(self.prior(prop, val, 
                                  bounds=self.bounds(prop)))
                if not np.isfinite(lnp):
                    return -np.inf

            # Note: this is just assuming proper order.
            #  Is this OK?  Should keep eye out for bugs here.

            masses = p[i:i+N[s]]

            # Mass prior for primary
            lnp += np.log(self.prior('mass', masses[0],
                                bounds=self.bounds('mass')))
            # Priors for mass ratios
            for j in range(N[s]-1):
                q = masses[j+1]/masses[0]
                lnp += np.log(self.prior('q', q,
                            bounds=self.bounds('q')))

            i += N[s] + 4

        return lnp

    def prior(self, prop, val, **kwargs):
        return self._priors[prop](val, **kwargs)
    
    

def age_prior(age, bounds):
    """
    Uniform true age prior; where 'age' is actually log(age)
    """
    minage, maxage = bounds
    if age < minage or age > maxage:
        return 0
    return age * (2/(maxage**2-minage**2))

def distance_prior(distance, bounds):
    """
    Distance prior ~ d^2
    """
    min_distance, max_distance = bounds
    if distance <= min_distance or distance > max_distance:
        return 0
    return np.log(3/max_distance**3 * distance**2)

def AV_prior(AV, bounds):
    if AV < bounds[0] or AV > bounds[1]:
        return 0
    return 1./bounds[1]

def q_prior(q, m=1, gamma=0.3, bounds=(0.1,1)):
    """Default prior on mass ratio q ~ q^gamma
    """
    qmin, qmax = bounds
    if q < qmin or q > qmax:
        return 0
    C = 1/(1/(gamma+1)*(1 - qmin**(gamma+1)))
    return C*q**gamma

def salpeter_prior(m,alpha=-2.35, bounds=(0.1,10)):
    minmass, maxmass = bounds
    C = (1+alpha)/(maxmass**(1+alpha)-minmass**(1+alpha))
    if m < minmass or m > maxmass:
        return 0
    else:
        return C*m**(alpha)

def local_fehdist(feh, bounds=None):
    """feh PDF based on local SDSS distribution
    
    From Jo Bovy:
    https://github.com/jobovy/apogee/blob/master/apogee/util/__init__.py#L3
    2D gaussian fit based on Casagrande (2011)
    """
    fehdist= 0.8/0.15*np.exp(-0.5*(feh-0.016)**2./0.15**2.)\
        +0.2/0.22*np.exp(-0.5*(feh+0.15)**2./0.22**2.)

    return fehdist
