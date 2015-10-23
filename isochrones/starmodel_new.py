import numpy as np
import re

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



class StarModel(object):
    def __init__(self, ic):
        self._ic = ic
        self.n_params = 5
    
    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic        
    
    def evaluate(self, p, prop):
        """Returns value of 'prop' predicted by parameters 'p'
        
        p = (mass, age, feh, distance, AV)
        """
        mass, age, feh, distance, AV = p
        if prop in ['mass','age','feh','distance','AV']:
            return eval(prop)
        elif prop in self.ic.bands:
            return self.evaluate_mag(p, prop)
        else:
            return getattr(self.ic, prop)(mass, age, feh)

    def evaluate_mag(self, p, band):
        return self.ic.mag[band](*p)
        
class MultipleStarModel(StarModel):

    tags = ['A','B','C','D','E'] #order of tags w/in single system

    def __init__(self, ic, labels):
        """
        models is list of StarModel objects
        labels a list of labels; anything with the same label will be
            assumed to be physically associated, and will be assumed
            to be in descending order of mass.
            
        """
        self._ic = ic
        
        self.labels = labels
        
        self.systems = []
        self.full_labels = []
        self.Nstars = {}
        for l in self.labels:
            if l not in self.systems:
                self.systems.append(l)
            if l in self.Nstars:
                self.Nstars[l] += 1
            else:
                self.Nstars[l] = 1
            self.full_labels.append('{}_{}'.format(l, self.tags[self.Nstars[l]-1]))
        self.systems.sort()
        self.Nsystems = len(self.systems)
        
        n = 0
        for s in self.systems:
            n += 4 + self.Nstars[s]
        self.n_params = n
        
        self._photometry = {}
        
    def _parse_params(self, p):
        """
        To parse parameter vector p
        
        For each system labeled by 'l', Nparams = 4 + self.Nstars[l] (one mass per star)
        Loop through systems in label-order, grabbing the appropriate number of params
        
        returns list of "normal" param vectors for each StarModel
        """
        assert len(p)==self.n_params
        
        pdict = {}
        i=0
        for l in self.systems:
            n = self.Nstars[l]
            pdict[l] = p[i:i+4+n]
            i += 4 + n
        
        params = []
        mass_ind = {l:0 for l in self.systems}
        for l in self.labels:
            pars = [pdict[l][mass_ind[l]]]
            pars += pdict[l][-4:]
            mass_ind[l] += 1
            params.append(pars)
            
        return params
            
    def _describe_params(self):
        """
        Prints schema of params
        """
        
        s = '['
        for l in self.systems:
            n = self.Nstars[l]
            if n==1:
                s += 'mass_{}, '.format(l)
            else:
                for i in range(n):
                    s += 'mass_{}_{}, '.format(self.tags[i],l)
            s += 'age_{0}, feh_{0}, distance_{0}, AV_{0}, '.format(l)
        s = s[:-2]
        s += ']'
        print(s)
               
    def add_photometry(self, name, band, mag, relative=False):
        """
        Add photometric observation to model
        
        string : Name of instrument/survey
        band : photometric band
        mag : list of magnitude values.  If a particular
              star is not resolved in this observation,
              corresponding entry should be ``None``, and the blended
              magnitude should go with the next brightest.
        relative : Should be true to just provide relative (rather
                   than absolute) photometry.  In this case, the 
                   "reference" magnitude should be 0.
        
        Each observation gets stored in a dictionary.
        
        Let's do some examples, starting simple. 

        First, a physically associated 2-star model unresolved 
        in 2mass but resolved in Keck/NIRC2:
        
            mod = MultipleStarModel(dar, [0,0])
            mod.add_photometry('2mass', 'J', [(10.1, 0.02), None])
            mod.add_photometry('2mass', 'H', [(9.8, 0.02), None])
            mod.add_photometry('2mass', 'K', [(9.4, 0.02), None])
            mod.add_photometry('NIRC2', 'J', [0, (2.5, 0.03)], relative=True)

        """
        
        if name not in self._photometry:
            self._photometry[name] = {}
        
        if 'relative' not in self._photometry[name]:
            self._photometry[name]['relative'] = {}
        
        self._photometry[name]['relative'][band] = relative
        self._photometry[name][band] = mag
        
    def photometry_lnlike(self, p):
        """
        log-likelihood of observed photometry, given parameters p
        """
        
        parlist = self._parse_params(p)

        tot = 0
        for obs in self._photometry.keys():
            print(obs)
            for b in self._photometry[obs].keys():
                if b=='relative':
                    continue
                rel = self._photometry[obs]['relative'][b]
                if rel:
                    i_ref = self._photometry[obs][b].index(0)
                    ref_mag = super(MultipleStarModel, 
                                     self).evaluate_mag(parlist[i_ref], b)
                obs_mag = {s:(np.inf, 0) for s in self.systems}
                model_mag = {s:np.inf for s in self.systems}
                for i,l in enumerate(self.labels):
                    if self._photometry[obs][b][i] is not None:
                        if np.size(self._photometry[obs][b][i])==1:
                            continue    
                        mag = self._photometry[obs][b][i]
                        obs_mag[l] = addmags(obs_mag[l], mag)

                    this_mag = super(MultipleStarModel, 
                                     self).evaluate_mag(parlist[i], b)
                    model_mag[l] = addmags(model_mag[l], this_mag)
                for l in self.systems:
                    m, dm = obs_mag[l]
                    if rel:
                        model_mag[l] -= ref_mag
                    tot += -0.5*(m - model_mag[l])**2 / dm**2

                print(obs_mag, model_mag)

        return tot

    def evaluate(self, p, prop, which=None):
        """
        p is parsed according to ._parse_params

        ``which`` must be specified unless ``prop`` is photometric band
        
        delta_? is a magnitude difference.  In order for this to be
        defined, there needs to be a "reference" magnitude, which by 
        convention is the brightest unresolved system (in any band) 
        being considered.
        
        This means that there needs to be some structured way to 
        define the photometric observations, allowing for stars
        to be blended or resolved in any given band.
        
        
        
        """
        
        
        if prop in self.ic.bands:
            return self.evaluate_mag(p, prop, which=which)
        
        if which is None:
            raise ValueError('Must specify which system.')
            
        m = re.search('([a-zA-Z]+)(_([A-Z]))?', prop)
        
        if m.group(2) is None:
            tag = self.tags[0]
        else:
            tag = m.group(3)
        
        i_tag = self.tags.index(tag)
        if i_tag > self.Nstars[which] - 1:
            raise ValueError('System {} has only {} stars.'.format(which, self.Nstars[which]))
        
        prop = m.group(1)

        full_label = '{}_{}'.format(which, tag)
        
        parlist = self._parse_params(p)
        i = self.full_labels.index(full_label)
        
        return super(MultipleStarModel, self).evaluate(parlist[i], prop)
                    
    def evaluate_mag(self, p, band, which=None):
        """
        p is parsed according to ._parse_params
        
        band : name of desired photometric band
        
        which : Label of system for which property is desired.
                If ``None``, will return the flux-sum of all.
                If array-like, then the flux-sum of everything
                ``True``.
        
        What about subset of a system?
        
        """
        parlist = self._parse_params(p)
        
        if which is not None:
            assert which in self.systems
        
        mags = []
        for i,l in enumerate(self.labels):
            if which is not None:
                if l != which:
                    continue
            mag = super(MultipleStarModel, self).evaluate_mag(parlist[i], band)
            mags.append(mag)
        return addmags(*mags)
            
