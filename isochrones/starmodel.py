from __future__ import print_function, division

import os, os.path, sys, re, glob
import itertools
from copy import deepcopy
import logging
import json

from .config import on_rtd

if not on_rtd:
    import numpy as np
    import pandas as pd

    import numpy.random as rand
    from scipy.stats import gaussian_kde
    import scipy
    import emcee
    import corner

    try:
        import pymultinest
    except ImportError:
        logging.warning('PyMultiNest not imported.  MultiNest fits will not work.')

    import configobj
    from astropy.coordinates import SkyCoord

    try:
        basestring
    except NameError:
        basestring = str

from .utils import addmags
from .observation import ObservationTree, Observation, Source
from .priors import age_prior, distance_prior, AV_prior, q_prior
from .priors import salpeter_prior, feh_prior
from .isochrone import get_ichrone, Isochrone

def _parse_config_value(v):
    try:
        val = float(v)
    except:
        try:
            val = [float(x) for x in v]
        except:
            val = v
    #print('{} becomes {}, type={}'.format(v,val,type(val)))
    return val



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
        should be passed.  If `obs` is a string, then it is assumed
        to be a filename of an obs summary DataFrame.

    :param N:
        Number of model stars to assign to each "leaf node" of the
        :class:`ObservationTree`.  If you want to model a binary star,
        provide ``N=2``.

    :param **kwargs:
        Keyword arguments must be properties of given isochrone, e.g., logg,
        feh, Teff, and/or magnitudes.  The values represent measurements of
        the star, and must be in (value,error) format. All such keyword
        arguments will be held in ``self.properties``.  ``parallax`` is
        also a valid property, and should be provided in miliarcseconds,
        as is ``density`` [g/cc], and ``nu_max`` and ``delta_nu``
        (asteroseismic parameters in uHz.)
    """

    # These are allowable parameters that are not photometric bands
    _not_a_band = ('RA','dec','ra','Dec','maxAV','parallax',
                  'logg','Teff','feh','density', 'separation',
                 'PA','resolution','relative','N','index', 'id')

    _default_name = 'single'

    def __init__(self, ic, obs=None, N=1, index=0,
                 name='', use_emcee=False,
                 RA=None, dec=None, coords=None,
                 **kwargs):

        self.name = name if name else self._default_name

        if coords is None:
            if RA is not None and dec is not None:
                try:
                    coords = SkyCoord(RA, dec)
                except:
                    coords = SkyCoord(float(RA), float(dec), unit='deg')
        self.coords = coords
        self._ic = ic

        self.use_emcee = use_emcee

        # If obs is not provided, build it
        if obs is None:
            self._build_obs(**kwargs)
            self.obs.define_models(ic, N=N, index=index)
            self._add_properties(**kwargs)
        elif isinstance(obs, basestring):
            df = pd.read_csv(obs)
            obs = ObservationTree.from_df(df)
            obs.define_models(ic, N=N, index=index)
            self.obs = obs
            self._add_properties(**kwargs)
        else:
            self.obs = obs
            if len(self.obs.get_model_nodes())==0:
                self.obs.define_models(ic, N=N, index=index)
                self._add_properties(**kwargs)

        self._priors = {'mass':salpeter_prior,
                        'feh':feh_prior,
                        'q':q_prior,
                        'age':age_prior,
                        'distance':distance_prior,
                        'AV':AV_prior}

        self._bounds = {'mass':None,
                        'feh':None,
                        'age':None,
                        'q':q_prior.bounds,
                        'distance':distance_prior.bounds,
                        'AV':AV_prior.bounds}

        if 'maxAV' in kwargs:
            self.set_bounds(AV=(0, kwargs['maxAV']))

        if 'max_distance' in kwargs:
            self.set_bounds(distance=(0, kwargs['max_distance']))

        self._directory = '.'
        self._samples = None

    @property
    def directory(self):
        return self._directory

    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic

    @classmethod
    def _parse_band(cls, kw):
        """Returns photometric band from inifile keyword
        """
        m = re.search('([a-zA-Z0-9]+)(_\d+)?', kw)
        if m:
            if m.group(1) in cls._not_a_band:
                return None
            else:
                return m.group(1)

    @classmethod
    def get_bands(cls, inifile):

        bands = []
        c = configobj.ConfigObj(inifile)
        for kw,v in c.items():
            if type(v) is configobj.Section:
                for kw in v:
                    b = cls._parse_band(kw)
                    if b is not None:
                        bands.append(b)
            else:
                b = cls._parse_band(kw)
                if b is not None:
                    bands.append(b)

        return list(set(bands))


    @classmethod
    def from_ini(cls, ic, folder='.', ini_file='star.ini', **kwargs):
        """
        Initialize a StarModel from a .ini file

        The "classic" format (version <= 0.9) should still work for a single star,
        where all properties are just listed in the file; e.g.,

            J = 10, 0.05
            H = 9.5, 0.05
            K = 9.0, 0.05
            Teff = 5000, 150

        If there are multiple stars observed, you can either define them in
        the ini file, or use the `obsfile` keyword, pointing to a file with
        the summarized photometric observations.  In this case, spectroscopic/parallax
        info should still be included in the .ini file; e.g.,

            obsfile = obs.csv
            Teff = 5000, 150

        The obsfile should be a comma-separated table with the following columns:
        `[name, band, resolution, mag, e_mag, separation, pa, relative]`.

          * `name` is the name of instrument
          * `band` is the photometric bandpass
          * `resolution` is the approximate spatial resolution of instrument
          * `mag`, `e_mag` describe magnitude of source (absolute or relative)
          * `separation`, `pa` describe position of source
          * `relative`: single-bit flag; if 1 then magnitudes taken with this
            instrument are assumed to be relative rather than absolute.

        If an obsfile is not provided, you can also define all the same information
        in the ini file, following these rules:

          * Every instrument/survey gets its own [section].  Sections are only
        created for different photometric observations.

          * if photometry relates to *all* stars in aperture,
            there is no extra info in the section, just the photometry.  In this case, it is
            also assumed that the photometry is absolute. (`relative=False`)

          * If 'resolution' is an attribute under a particular survey section (and
        'relative' is not explicitly stated), then the survey is assumed to have relative
        photometry, and to be listing
        information about companion stars.  In this case, there must be "separation"
        and "PA" included for each companion.  If there is more than one companion star,
        they must be identifed by tag, e.g., separation_1, PA_1, Ks_1, J_1, etc.  The
        tag can be anything alphanumeric, but it must be consistent within a particular
        section (instrument).  If there
        is no tag, there is assumed to be only one companion detected.

          * If there are no sections, then bands will be interpreted at face value
        and will all be assumed to apply to all stars modeled.

          * Default is to model each star in the highest-resolution observation as a
        single star, at the same distance/age/feh/AV.


        The `N` and `index`
        parameters may also be provided, to specify the relations between the
        model stars.  If these are not provided, then `N` will default to `1`
        (one model star per star observed in highest-resolution observation)
        and `index` will default to all `0` (all stars physically associated).

        """

        if not os.path.isabs(ini_file):
            ini_file = os.path.join(folder,ini_file)

        bands = cls.get_bands(ini_file)

        if not isinstance(ic, Isochrone):
            ic = get_ichrone(ic, bands)

        logging.debug('Initializing StarModel from {}'.format(ini_file))

        c = configobj.ConfigObj(ini_file)

        RA = c.get('RA')
        dec = c.get('dec')
        maxAV = c.get('maxAV')

        if len(c.sections) == 0:
            for k in c:
                kwargs[k] = _parse_config_value(c[k])
            obs = None
        else:

            columns = ['name', 'band', 'resolution', 'relative', 'separation', 'pa', 'mag', 'e_mag']
            df = pd.DataFrame(columns=columns)
            i = 0
            for k in c:
                if type(c[k]) != configobj.Section:
                    kwargs[k] = _parse_config_value(c[k])
                else:
                    instrument = k

                    # Set values of 'resolution' and 'relative'
                    if 'resolution' in c[k]:
                        resolution = float(c[k]['resolution'])
                        relative = True
                    else:
                        resolution = 4.0 #default
                        relative = False

                    # Overwrite value of 'relative' if it is explicitly set
                    if 'relative' in c[k]:
                        relative = c[k]['relative']=='True'


                    # Check if there are multiple stars (defined by whether
                    # any separations are listed).
                    # While we're at it, keep track of tags if they exist,
                    #  and pull out the names of the bands.
                    multiple = False
                    tags = []
                    bands = []
                    for label in c[k]:
                        m = re.search('separation(_\w+)?', label)
                        if m:
                            multiple = True
                            if m.group(1) is not None:
                                if m.group(1) not in tags:
                                    tags.append(m.group(1))
                        elif re.search('PA', label) or re.search('id', label) or \
                                label in ['resolution', 'relative']:
                            continue
                        else:
                            # At this point, this should be a photometric band
                            m = re.search('([a-zA-Z0-9]+)(_\w+)?', label)
                            b = m.group(1)
                            if b not in bands:
                                bands.append(b)

                    # If a blank tags needs to be created, do so
                    if len(bands) > 0 and (len(tags)==0 or bands[0] in c[k]):
                        tags.append('')

                    # For each band and each star, create a row
                    for b in bands:
                        for tag in tags:
                            if '{}{}'.format(b, tag) not in c[k]:
                                continue
                            row = {}
                            row['name'] = instrument
                            row['band'] = b
                            row['resolution'] = resolution
                            row['relative'] = relative
                            if 'separation{}'.format(tag) in c[k]:
                                row['separation'] = c[k]['separation{}'.format(tag)]
                                row['pa'] = c[k]['PA{}'.format(tag)]
                            else:
                                row['separation'] = 0.
                                row['pa'] = 0.
                            mag, e_mag = c[k]['{}{}'.format(b,tag)]
                            row['mag'] = float(mag)
                            row['e_mag'] = float(e_mag)
                            if not np.isnan(row['mag']) and not np.isnan(row['e_mag']):
                                df = df.append(pd.DataFrame(row, index=[i]))
                                i += 1

                        # put the reference star in w/ mag=0
                        if relative:
                            row = {}
                            row['name'] = instrument
                            row['band'] = b
                            row['resolution'] = resolution
                            row['relative'] = relative
                            row['separation'] = 0.
                            row['pa'] = 0.
                            row['mag'] = 0.
                            row['e_mag'] = 0.01
                            df = df.append(pd.DataFrame(row, index=[i]))
                            i += 1

            obs = ObservationTree.from_df(df)

        if 'obsfile' in c:
            obs = c['obsfile']

        logging.debug('Obs is {}'.format(obs))

        if 'name' not in kwargs:
            kwargs['name'] = os.path.basename(folder)
        new = StarModel(ic, obs=obs, **kwargs)
        new._directory = os.path.abspath(folder)

        return new

    def print_ascii(self):
        """Prints an ascii representation of the observation tree structure.
        """
        return self.obs.print_ascii()

    def bounds(self, prop):
        if self._bounds[prop] is not None:
            return self._bounds[prop]
        elif prop=='mass':
            lo, hi = (self.ic.minmass, self.ic.maxmass)
            self._bounds['mass'] = (lo, hi)
            self._priors['mass'].bounds = (lo, hi)
        elif prop=='feh':
            lo, hi = (self.ic.minfeh, self.ic.maxfeh)
            self._bounds['feh'] = (lo, hi)
            self._priors['feh'].bounds = (lo, hi)
        elif prop=='age':
            lo, hi = (self.ic.minage, self.ic.maxage)
            self._bounds['age'] = (lo, hi)
            self._priors['age'].bounds = (lo, hi)
            self._bounds['age'] = (self.ic.minage,
                                   self.ic.maxage)
        else:
            raise ValueError('Unknown property {}'.format(prop))
        return self._bounds[prop]

    def set_bounds(self, **kwargs):
        for k,v in kwargs.items():
            if len(v) != 2:
                raise ValueError('Must provide (min, max)')
            self._bounds[k] = v
            self._priors[k].bounds = v

    def _build_obs(self, **kwargs):
        """
        Builds ObservationTree out of keyword arguments

        Ignores anything that is not a photometric bandpass.
        This should not be used if there are multiple stars observed.

        Creates self.obs
        """
        logging.debug('Building ObservationTree...')
        tree = ObservationTree()
        for k,v in kwargs.items():
            if k in self.ic.bands:
                if np.size(v) != 2:
                    logging.warning('{}={} ignored.'.format(k,v))
                    # continue
                    v = [v, np.nan]
                o = Observation('', k, 99) #bogus resolution=99
                s = Source(v[0], v[1])
                o.add_source(s)
                logging.debug('Adding {} ({})'.format(s,o))
                tree.add_observation(o)

        self.obs = tree

    def _add_properties(self, **kwargs):
        """
        Adds non-photometry properties to ObservationTree
        """
        for k,v in kwargs.items():
            if k=='parallax':
                self.obs.add_parallax(v)
            elif k in ['Teff', 'logg', 'feh', 'density']:
                par = {k:v}
                self.obs.add_spectroscopy(**par)
            elif re.search('_', k):
                m = re.search('^(\w+)_(\w+)$', k)
                prop = m.group(1)
                tag = m.group(2)
                self.obs.add_spectroscopy(**{prop:v, 'label':'0_{}'.format(tag)})


    @property
    def param_description(self):
        return self.obs.param_description

    @property
    def param_names(self):
        return self.param_description

    @property
    def mags(self):
        return {n.band : n.value[0] for n in self.obs.get_obs_nodes()}

    def lnpost(self, p, **kwargs):
        lnpr = self.lnprior(p)
        if not np.isfinite(lnpr):
            return lnpr
        return lnpr + self.lnlike(p, **kwargs)

    def lnlike(self, p, **kwargs):
        lnl = self.obs.lnlike(p, **kwargs)
        return lnl

    def lnprior(self, p):
        N = self.obs.Nstars
        i = 0
        lnp = 0
        for s in self.obs.systems:
            age, feh, dist, AV = p[i+N[s]:i+N[s]+4]
            for prop, val in zip(['age','feh','distance','AV'],
                                 [age, feh, dist, AV]):
                lo,hi = self.bounds(prop)
                if val < lo or val > hi:
                    return -np.inf
                lnp += np.log(self.prior(prop, val))
                if not np.isfinite(lnp):
                    logging.debug('lnp=-inf for {}={} (system {})'.format(prop,val,s))
                    return -np.inf

            # Note: this is just assuming proper order.
            #  Is this OK?  Should keep eye out for bugs here.

            masses = p[i:i+N[s]]

            # Mass prior for primary
            lnp += np.log(self.prior('mass', masses[0]))
            if not np.isfinite(lnp):
                logging.debug('lnp=-inf for mass={} (system {})'.format(masses[0],s))

            # Priors for mass ratios
            for j in range(N[s]-1):
                q = masses[j+1]/masses[0]
                qmin, qmax = self.bounds('q')

                ## The following would enforce MA > MB > MC, but seems to make things very slow:
                #if j+1 > 1:
                #    qmax = masses[j] / masses[0]

                lnp += np.log(self.prior('q', q))
                if not np.isfinite(lnp):
                    logging.debug('lnp=-inf for q={} (system {})'.format(q,s))
                    return -np.inf

            i += N[s] + 4

        return lnp

    def prior(self, prop, val, **kwargs):
        return self._priors[prop](val, **kwargs)


    @property
    def n_params(self):
        tot = 0
        for _,n in self.obs.Nstars.items():
            tot += 4+n
        return tot

    def mnest_prior(self, cube, ndim, nparams):
        i = 0
        for _,n in self.obs.Nstars.items():
            minmass, maxmass = self.bounds('mass')
            for j in range(n):
                cube[i+j] = (maxmass - minmass)*cube[i+j] + minmass

            for j, par in enumerate(['age','feh','distance','AV']):
                lo, hi = self.bounds(par)
                cube[i+n+j] = (hi - lo)*cube[i+n+j] + lo
            i += 4 + n

    def mnest_loglike(self, cube, ndim, nparams):
        """loglikelihood function for multinest
        """
        return self.lnpost(cube)

    @property
    def labelstring(self):
        return '--'.join(['-'.join([n.label for n in l.children]) for l in self.obs.get_obs_leaves()])

    def fit(self, **kwargs):
        if self.use_emcee:
            return self.fit_mcmc(**kwargs)
        else:
            return self.fit_multinest(**kwargs)

    @property
    def mnest_basename(self):
        """Full path to basename
        """
        if not hasattr(self, '_mnest_basename'):
            s = self.labelstring
            if s=='0_0':
                s = 'single'
            elif s=='0_0-0_1':
                s = 'binary'
            elif s=='0_0-0_1-0_2':
                s = 'triple'

            s = '{}-{}'.format(self.ic.name, s)
            self._mnest_basename = os.path.join('chains', s+'-')

        if os.path.isabs(self._mnest_basename):
            return self._mnest_basename
        else:
            return os.path.join(self.directory, self._mnest_basename)

    @mnest_basename.setter
    def mnest_basename(self, basename):
        if os.path.isabs(basename):
            self._mnest_basename = basename
        else:
            self._mnest_basename = os.path.join('chains', basename)

    def lnpost_polychord(self, theta):
        phi = [0.0] #nDerived
        return self.lnpost(theta), phi

    def fit_polychord(self, basename, verbose=False, **kwargs):
        from .config import POLYCHORD
        sys.path.append(POLYCHORD)
        import PyPolyChord.PyPolyChord as PolyChord

        return PolyChord.run_nested_sampling(self.lnpost_polychord,
                        self.n_params, 0, file_root=basename, **kwargs)


    def fit_multinest(self, n_live_points=1000, basename=None,
                      verbose=True, refit=False, overwrite=False,
                      test=False,
                      **kwargs):
        """
        Fits model using MultiNest, via pymultinest.

        :param n_live_points:
            Number of live points to use for MultiNest fit.

        :param basename:
            Where the MulitNest-generated files will live.
            By default this will be in a folder named `chains`
            in the current working directory.  Calling this
            will define a `_mnest_basename` attribute for
            this object.

        :param verbose:
            Whether you want MultiNest to talk to you.

        :param refit, overwrite:
            Set either of these to true if you want to
            delete the MultiNest files associated with the
            given basename and start over.

        :param **kwargs:
            Additional keyword arguments will be passed to
            :func:`pymultinest.run`.

        """

        if basename is not None: #Should this even be allowed?
            self.mnest_basename = basename

        basename = self.mnest_basename
        if verbose:
            logging.info('MultiNest basename: {}'.format(basename))

        folder = os.path.abspath(os.path.dirname(basename))
        if not os.path.exists(folder):
            os.makedirs(folder)


        #If previous fit exists, see if it's using the same
        # observed properties
        prop_nomatch = False
        propfile = '{}properties.json'.format(basename)

        """
        if os.path.exists(propfile):
            with open(propfile) as f:
                props = json.load(f)
            if set(props.keys()) != set(self.properties.keys()):
                prop_nomatch = True
            else:
                for k,v in props.items():
                    if np.size(v)==2:
                        if not self.properties[k][0] == v[0] and \
                                self.properties[k][1] == v[1]:
                            props_nomatch = True
                    else:
                        if not self.properties[k] == v:
                            props_nomatch = True

        if prop_nomatch and not overwrite:
            raise ValueError('Properties not same as saved chains ' +
                            '(basename {}*). '.format(basename) +
                            'Use overwrite=True to fit.')
        """

        if refit or overwrite:
            files = glob.glob('{}*'.format(basename))
            [os.remove(f) for f in files]

        short_basename = self._mnest_basename

        mnest_kwargs = dict(n_live_points=n_live_points, outputfiles_basename=short_basename,
                        verbose=verbose)

        for k,v in kwargs.items():
            mnest_kwargs[k] = v

        if test:
            print('pymultinest.run() with the following kwargs: {}'.format(mnest_kwargs))
        else:
            wd = os.getcwd()
            os.chdir(os.path.join(folder, '..'))
            pymultinest.run(self.mnest_loglike, self.mnest_prior, self.n_params,
                            **mnest_kwargs)
            os.chdir(wd)
            #with open(propfile, 'w') as f:
            #    json.dump(self.properties, f, indent=2)

            self._make_samples()

    @property
    def mnest_analyzer(self):
        """
        PyMultiNest Analyzer object associated with fit.

        See PyMultiNest documentation for more.
        """
        return pymultinest.Analyzer(self.n_params, self.mnest_basename)

    @property
    def evidence(self):
        """
        Log(evidence) from multinest fit
        """
        s = self.mnest_analyzer.get_stats()
        return (s['global evidence'],s['global evidence error'])


    def maxlike(self, p0, **kwargs):
        """ Finds (local) optimum in parameter space.
        """
        def fn(p):
            return -self.lnpost(p)

        if 'method' not in kwargs:
            kwargs['method'] = 'Nelder-Mead'

        p0 = [0.8, 9.5, 0.0, 200, 0.2]
        fit = scipy.optimize.minimize(fn, p0, **kwargs)
        return fit

    def sample_from_prior(self, n):
        return self.emcee_p0(n)

    def emcee_p0(self, nwalkers):

        def sample_row(nstars, n=nwalkers):
            p = []
            m0 = self._priors['mass'].sample(n)
            age0 = self._priors['age'].sample(n)
            feh0 = self._priors['feh'].sample(n)
            d0 = self._priors['distance'].sample(n)
            AV0 = self._priors['AV'].sample(n)

            for i in range(nstars):
                p += [m0 * 0.95**i]
            p += [age0, feh0, d0, AV0]
            return p

        p0 = []
        for _,n in self.obs.Nstars.items():
            p0 += sample_row(n)

        p0 = np.array(p0).T

        nbad = 1

        while True:
            ibad = []
            for i, p in enumerate(p0):
                if not np.isfinite(self.lnpost(p)):
                    ibad.append(i)

            nbad = len(ibad)
            if nbad == 0:
                break

            pnew = []
            for _, n in self.obs.Nstars.items():
                pnew += sample_row(n, n=nbad)

            pnew = np.array(pnew).T

            p0[ibad, :] = pnew

        return p0

    def fit_mcmc(self,nwalkers=300,nburn=200,niter=100,
                 p0=None,initial_burn=None,
                 ninitial=50, loglike_kwargs=None,
                 **kwargs):
        """Fits stellar model using MCMC.

        :param nwalkers: (optional)
            Number of walkers to pass to :class:`emcee.EnsembleSampler`.
            Default is 200.

        :param nburn: (optional)
            Number of iterations for "burn-in."  Default is 100.

        :param niter: (optional)
            Number of for-keeps iterations for MCMC chain.
            Default is 200.

        :param p0: (optional)
            Initial parameters for emcee.  If not provided, then chains
            will behave according to whether inital_burn is set.

        :param initial_burn: (optional)
            If `True`, then initialize walkers first with a random initialization,
            then cull the walkers, keeping only those with > 15% acceptance
            rate, then reinitialize sampling.  If `False`, then just do
            normal burn-in.  Default is `None`, which will be set to `True` if
            fitting for distance (i.e., if there are apparent magnitudes as
            properties of the model), and `False` if not.

        :param ninitial: (optional)
            Number of iterations to test walkers for acceptance rate before
            re-initializing.

        :param loglike_args:
            Any arguments to pass to :func:`StarModel.loglike`, such
            as what priors to use.

        :param **kwargs:
            Additional keyword arguments passed to :class:`emcee.EnsembleSampler`
            constructor.

        :return:
            :class:`emcee.EnsembleSampler` object.

        """

        #clear any saved _samples
        if self._samples is not None:
            self._samples = None

        npars = self.n_params

        if p0 is None:
            p0 = self.emcee_p0(nwalkers)
            if initial_burn:
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost,
                                                **kwargs)
                #ninitial = 300 #should this be parameter?
                pos, prob, state = sampler.run_mcmc(p0, ninitial)

                # Choose walker with highest final lnprob to seed new one
                i,j = np.unravel_index(sampler.lnprobability.argmax(),
                                        sampler.shape)
                p0_best = sampler.chain[i,j,:]
                print("After initial burn, p0={}".format(p0_best))
                p0 = p0_best * (1 + rand.normal(size=p0.shape)*0.001)
                print(p0)
        else:
            p0 = np.array(p0)
            p0 = rand.normal(size=(nwalkers,npars))*0.01 + p0.T[None,:]

        sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0=state)

        self._sampler = sampler
        return sampler

    @property
    def sampler(self):
        """
        Sampler object from MCMC run.
        """
        if hasattr(self,'_sampler'):
            return self._sampler
        else:
            raise AttributeError('MCMC must be run to access sampler')

    def _make_samples(self):

        if not self.use_emcee:
            filename = '{}post_equal_weights.dat'.format(self.mnest_basename)
            try:
                chain = np.loadtxt(filename)
                try:
                    lnprob = chain[:,-1]
                    chain = chain[:,:-1]
                except IndexError:
                    lnprob = np.array([chain[-1]])
                    chain = np.array([chain[:-1]])
            except:
                logging.error('Error loading chains from {}'.format(filename))
                raise
        else:
            #select out only walkers with > 0.15 acceptance fraction
            ok = self.sampler.acceptance_fraction > 0.15

            chain = self.sampler.chain[ok,:,:]
            chain = chain.reshape((chain.shape[0]*chain.shape[1],
                                        chain.shape[2]))

            lnprob = self.sampler.lnprobability[ok, :].ravel()

        df = pd.DataFrame()

        i=0
        for s,n in self.obs.Nstars.items():
            age = chain[:,i+n]
            feh = chain[:,i+n+1]
            distance = chain[:,i+n+2]
            AV = chain[:,i+n+3]
            for j in range(n):
                mass = chain[:,i+j]
                d = self.ic(mass, age, feh,
                             distance=distance, AV=AV)
                for c in d.columns:
                    df[c+'_{}_{}'.format(s,j)] = d[c]
            df['age_{}'.format(s)] = age
            df['feh_{}'.format(s)] = feh
            df['distance_{}'.format(s)] = distance
            df['AV_{}'.format(s)] = AV

            i += 4 + n

        for b in self.ic.bands:
            tot = np.inf
            for s,n in self.obs.Nstars.items():
                for j in range(n):
                    tot = addmags(tot,df[b + '_mag_{}_{}'.format(s,j)])
            df[b + '_mag'] = tot

        df['lnprob'] = lnprob

        self._samples = df.copy()

    @property
    def samples(self):
        """Dataframe with samples drawn from isochrone according to posterior

        Columns include both the sampling parameters from the MCMC
        fit (mass, age, Fe/H, [distance, A_V]), and also evaluation
        of the :class:`Isochrone` at each of these sample points---this
        is how chains of physical/observable parameters get produced.

        """
        if not hasattr(self,'sampler') and self._samples is None:
            raise AttributeError('Must run MCMC (or load from file) '+
                                 'before accessing samples')

        if self._samples is not None:
            df = self._samples
        else:
            self._make_samples()
            df = self._samples

        return df

    def random_samples(self, n):
        """
        Returns a random sampling of given size from the existing samples.

        :param n:
            Number of samples

        :return:
            :class:`pandas.DataFrame` of length ``n`` with random samples.
        """
        samples = self.samples
        inds = rand.randint(len(samples),size=int(n))

        newsamples = samples.iloc[inds]
        newsamples.reset_index(inplace=True)
        return newsamples

    def triangle(self, *args, **kwargs):
        return self.corner(*args, **kwargs)

    def corner(self, params, query=None, **kwargs):
        df = self.samples
        if query is not None:
            df = df.query(query)

        priors = []
        for p in params:
            if re.match('mass', p):
                priors.append(lambda x: self.prior('mass', x, bounds=self.bounds('mass')))
            elif re.match('age', p):
                priors.append(lambda x: self.prior('age', x, bounds=self.bounds('age')))
            elif re.match('feh', p):
                priors.append(lambda x: self.prior('feh', x, bounds=self.bounds('feh')))
            elif re.match('distance', p):
                priors.append(lambda x: self.prior('distance', x, bounds=self.bounds('distance')))
            elif re.match('AV', p):
                priors.append(lambda x: self.prior('AV', x, bounds=self.bounds('AV')))
            else:
                priors.append(None)

        try:
            fig = corner.corner(df[params], labels=params, priors=priors, **kwargs)
        except:
            logging.warning("Use Tim's version of corner to plot priors.")
            fig = corner.corner(df[params], labels=params, **kwargs)
        fig.suptitle(self.name, fontsize=22)
        return fig

    def triangle_physical(self, *args, **kwargs):
        return self.corner_physical(*args, **kwargs)

    def corner_plots(self, basename, **kwargs):
        fig1, fig2 = self.corner_physical(**kwargs), self.corner_observed(**kwargs)
        fig1.savefig(basename + '_physical.png')
        fig2.savefig(basename + '_observed.png')
        return fig1, fig2

    def triangle_plots(self, *args, **kwargs):
        return self.corner_plots(*args, **kwargs)

    def corner_physical(self, props=['mass','radius','feh','age','distance','AV'], **kwargs):
        collective_props = ['feh','age','distance','AV']
        indiv_props = [p for p in props if p not in collective_props]
        sys_props = [p for p in props if p in collective_props]

        props = ['{}_{}'.format(p,l) for p in indiv_props for l in self.obs.leaf_labels]
        props += ['{}_{}'.format(p,s) for p in sys_props for s in self.obs.systems]

        if 'range' not in kwargs:
            rng = [0.995 for p in props]

        return self.corner(props, range=rng, **kwargs)

    def mag_plot(self, *args, **kwargs):
        pass

    def corner_observed(self, **kwargs):
        """Makes corner plot for each observed node magnitude
        """
        tot_mags = []
        names = []
        truths = []
        rng = []
        for n in self.obs.get_obs_nodes():
            labels = [l.label for l in n.get_model_nodes()]
            band = n.band
            mags = [self.samples['{}_mag_{}'.format(band, l)] for l in labels]
            tot_mag = addmags(*mags)

            if n.relative:
                name = '{} $\Delta${}'.format(n.instrument, n.band)
                ref = n.reference
                if ref is None:
                    continue
                ref_labels = [l.label for l in ref.get_model_nodes()]
                ref_mags = [self.samples['{}_mag_{}'.format(band, l)] for l in ref_labels]
                tot_ref_mag = addmags(*ref_mags)
                tot_mags.append(tot_mag - tot_ref_mag)
                truths.append(n.value[0] - ref.value[0])
            else:
                name = '{} {}'.format(n.instrument, n.band)
                tot_mags.append(tot_mag)
                truths.append(n.value[0])

            names.append(name)
            rng.append((min(truths[-1], np.percentile(tot_mags[-1],0.5)),
                        max(truths[-1], np.percentile(tot_mags[-1],99.5))))
        tot_mags = np.array(tot_mags).T


        return corner.corner(tot_mags, labels=names, truths=truths, range=rng, **kwargs)


    def save_hdf(self, filename, path='', overwrite=False, append=False):
        """Saves object data to HDF file (only works if MCMC is run)

        Samples are saved to /samples location under given path,
        :class:`ObservationTree` is saved to /obs location under given path.

        :param filename:
            Name of file to save to.  Should be .h5 file.

        :param path: (optional)
            Path within HDF file structure to save to.

        :param overwrite: (optional)
            If ``True``, delete any existing file by the same name
            before writing.

        :param append: (optional)
            If ``True``, then if a file exists, then just the path
            within the file will be updated.
        """
        if os.path.exists(filename):
            with pd.HDFStore(filename) as store:
                if path in store:
                    if overwrite:
                        os.remove(filename)
                    elif not append:
                        raise IOError('{} in {} exists.  Set either overwrite or append option.'.format(path,filename))

        if self.samples is not None:
            self.samples.to_hdf(filename, path+'/samples')
        else:
            pd.DataFrame().to_hdf(filename, path+'/samples')

        self.obs.save_hdf(filename, path+'/obs', append=True)

        with pd.HDFStore(filename) as store:
            # store = pd.HDFStore(filename)
            attrs = store.get_storer('{}/samples'.format(path)).attrs

            attrs.ic_type = type(self.ic)
            attrs.ic_bands = list(self.ic.bands)
            attrs.use_emcee = self.use_emcee
            if hasattr(self, '_mnest_basename'):
                attrs._mnest_basename = self._mnest_basename

            attrs._bounds = self._bounds
            attrs._priors = self._priors

            attrs.name = self.name
            store.close()

    @classmethod
    def load_hdf(cls, filename, path='', name=None):
        """
        A class method to load a saved StarModel from an HDF5 file.

        File must have been created by a call to :func:`StarModel.save_hdf`.

        :param filename:
            H5 file to load.

        :param path: (optional)
            Path within HDF file.

        :return:
            :class:`StarModel` object.
        """
        if not os.path.exists(filename):
            raise IOError('{} does not exist.'.format(filename))
        store = pd.HDFStore(filename)
        try:
            samples = store[path+'/samples']
            attrs = store.get_storer(path+'/samples').attrs
        except:
            store.close()
            raise

        try:
            ic = attrs.ic_type(attrs.ic_bands)
        except AttributeError:
            ic = attrs.ic_type

        use_emcee = attrs.use_emcee
        mnest = True
        try:
            basename = attrs._mnest_basename
        except AttributeError:
            mnest = False
        bounds = attrs._bounds
        priors = attrs._priors

        if name is None:
            try:
                name = attrs.name
            except:
                name = ''

        store.close()

        obs = ObservationTree.load_hdf(filename, path+'/obs', ic=ic)

        mod = cls(ic, obs=obs,
                  use_emcee=use_emcee, name=name)
        mod._samples = samples
        if mnest:
            mod._mnest_basename = basename
        mod._directory = os.path.dirname(filename)
        return mod

class BinaryStarModel(StarModel):
    _default_name = 'binary'
    def __init__(self, *args, **kwargs):
        kwargs['N'] = 2
        super(BinaryStarModel, self).__init__(*args, **kwargs)

    @classmethod
    def from_ini(cls, *args, **kwargs):
        kwargs['N'] = 2
        return super(BinaryStarModel, cls).from_ini(*args, **kwargs)

class TripleStarModel(StarModel):
    _default_name = 'triple'
    def __init__(self, *args, **kwargs):
        kwargs['N'] = 3
        super(TripleStarModel, self).__init__(*args, **kwargs)

    @classmethod
    def from_ini(cls, *args, **kwargs):
        kwargs['N'] = 3
        return super(TripleStarModel, cls).from_ini(*args, **kwargs)

class StarModelGroup(object):
    """A collection of StarModel objects with different model node specifications

    Pass a single StarModel, and model nodes will be cleared and replaced with
    different variants.
    """
    def __init__(self, base_model, max_multiples=1, max_stars=2):

        self.base_model = deepcopy(base_model)
        self.base_model.obs.clear_models()
        self.max_multiples = max_multiples
        self.max_stars = max_stars

        self.models = []
        for N, index in self.model_options:
            mod = deepcopy(self.base_model)
            mod.obs.define_models(self.ic, N=N, index=index)
            self.models.append(mod)

    @property
    def ic(self):
        return self.base_model.ic

    @property
    def N_stars(self):
        return len(self.base_model.obs.leaves)

    @property
    def N_options(self):
        return N_options(self.N_stars, max_multiples=self.max_multiples,
                         max_stars=self.max_stars)

    @property
    def index_options(self):
        return index_options(self.N_stars)

    @property
    def model_options(self):
        return [(N, index) for N in self.N_options for index in self.index_options]

########## Utility functions ###############

def N_options(N_stars, max_multiples=1, max_stars=2):
    return [N for N in itertools.product(np.arange(max_stars) + 1, repeat=N_stars)
            if (np.array(N)>1).sum() <= max_multiples]

def index_options(N_stars):
    if N_stars==1:
        return [0]

    options = []
    for ind in itertools.product(range(N_stars), repeat=N_stars):
        diffs = np.array(ind[1:]) - np.array(ind[:-1])
        if ind[0]==0 and diffs.max()<=1:
            options.append(ind)
    return options
