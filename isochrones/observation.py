from __future__ import print_function, division
import os, re, sys
import numpy as np
import pandas as pd
import logging
from configobj import ConfigObj

from asciitree import LeftAligned, Traversal
from asciitree.drawing import BoxStyle, BOX_DOUBLE, BOX_BLANK

from itertools import chain, imap, izip, count
from collections import OrderedDict

from .isochrone import get_ichrone
from .utils import addmags, distance

class NodeTraversal(Traversal):
    """
    Custom subclass to traverse tree for ascii printing
    """
    def __init__(self, pars=None, **kwargs):
        self.pars = pars
        super(NodeTraversal,self).__init__(**kwargs)

    def get_children(self, node):
        return node.children
    
    def get_root(self, node):
        return node
        return node.get_root()
    
    def get_text(self, node):
        text = node.label
        if self.pars is not None:
            if hasattr(node, 'model_mag'):
                text += '; model={:.2f} ({})'.format(node.model_mag(self.pars),
                                                     node.lnlike(self.pars))
            if type(node)==ModelNode:
                root = node.get_root()
                if hasattr(root, 'spectroscopy'):
                    if node.label in root.spectroscopy:
                        for k,v in root.spectroscopy[node.label].items():
                            text += ', {}={}'.format(k,v)

                            modval = node.evaluate(self.pars[node.label], k)
                            lnl = -0.5*(modval - v[0])**2/v[1]**2
                            text += '; model={} ({})'.format(modval, lnl)
                    if node.label in root.limits:
                        for k,v in root.limits[node.label].items():
                            text += ', {} limits={}'.format(k,v)
                text += ': {}'.format(self.pars[node.label])

        else:
            if type(node)==ModelNode:
                root = node.get_root()
                if hasattr(root, 'spectroscopy'):
                    if node.label in root.spectroscopy:
                        for k,v in root.spectroscopy[node.label].items():
                            text += ', {}={}'.format(k,v)
                    if node.label in root.limits:
                        for k,v in root.limits[node.label].items():
                            text += ', {} limits={}'.format(k,v)
                #root = node.get_root()
                #if hasattr(root,'spectroscopy'):
                #    if node.label in root.spectroscopy:
                #        for k,v in root.spectroscopy[node.label].items():
                #            model = node.evaluate(self.pars[node.label], k)
                #            text += '\n  {}={} (model={})'.format(k,v,model)
        return text

class MyLeftAligned(LeftAligned):
    """For custom ascii tree printing
    """
    pars = None
    def __init__(self, pars=None, **kwargs):
        self.pars = pars
        self.traverse = NodeTraversal(pars)
        super(MyLeftAligned,self).__init__(**kwargs)
    
class Node(object):
    def __init__(self, label):

        self.label = label
        self.parent = None
        self.children = []
        self._leaves = None

    def __iter__(self):
        """
        Iterate through tree, leaves first

        following http://stackoverflow.com/questions/6914803/python-iterator-through-tree-with-list-of-children
        """
        for node in chain(*imap(iter, self.children)):
            yield node
        yield self

    def __getitem__(self, ind):
        for n,i in izip(self, count()):
            if i==ind:
                return n

    @property
    def is_root(self):
        return self.parent is None

    def get_root(self):
        if self.is_root:
            return self
        else:
            return self.parent.get_root()
        
    def get_ancestors(self):
        if self.parent.is_root:
            return []
        else:
            return [self.parent] + self.parent.get_ancestors()

    def print_ascii(self, fout=None, pars=None):
        box_tr = MyLeftAligned(pars,draw=BoxStyle(gfx=BOX_DOUBLE, horiz_len=1))
        if fout is None:
            print(box_tr(self))
        else: 
            fout.write(box_tr(self))

    @property
    def is_leaf(self):
        return len(self.children)==0 and not self.is_root

    def _clear_leaves(self):
        self._leaves = None
    
    def _clear_all_leaves(self):
        if not self.is_root:
            self.parent._clear_all_leaves()
        self._clear_leaves()
        
    def add_child(self, node):
        node.parent = self
        self.children.append(node)
        self._clear_all_leaves()

    def remove_children(self):
        self.children = []    
        self._clear_all_leaves()    

    def remove_child(self, label):
        """
        Removes node by label
        """
        ind = None
        for i,c in enumerate(self.children):
            if c.label==label:
                ind = i

        if ind is None:
            logging.warning('No child labeled {}.'.format(label))
            return
        self.children.pop(ind)
        self._clear_all_leaves()
    
    def attach_to_parent(self, node):
        # detach from current parent, if necessary
        if self.parent is not None:
            self.parent.remove_child(self.label)
            
        node.children += [self]
        self.parent = node
        self._clear_all_leaves()
    
    @property
    def leaves(self):
        if self._leaves is None:
            self._leaves = self._get_leaves()
        return self._leaves

    def _get_leaves(self):
        if self.is_leaf:
            return [self]
        else:
            leaves = []
            for c in self.children:
                leaves += c._get_leaves()
            return leaves

    def select_leaves(self, name):
        """Returns all leaves under all nodes matching name 

        """

        if self.is_leaf:
            return [self] if re.search(name, self.label) else []
        else:
            leaves = []
            if re.search(name, self.label):
                for c in self.children:
                    leaves += c._get_leaves() #all leaves
            else:
                for c in self.children:
                    leaves += c.select_leaves(name) #only matching ones
            return leaves
        
    @property
    def leaf_labels(self):
        return [l.label for l in self.leaves]
        
    def get_leaf(self, label):
        for l in self.leaves:
            if label==l.label:
                return l

    def get_obs_nodes(self):
        return [l for l in self if isinstance(l, ObsNode)]

    @property
    def obs_leaf_nodes(self):
        return self.get_obs_leaves()
    

    def get_obs_leaves(self):
        """Returns the last obs nodes that are leaves
        """
        obs_leaves = []
        for n in self:
            if n.is_leaf:
                if isinstance(n, ModelNode):
                    l = n.parent
                else:
                    l = n
            if l not in obs_leaves:
                obs_leaves.append(l)
        return obs_leaves

    def get_model_nodes(self):
        return [l for l in self._get_leaves() if isinstance(l, ModelNode)]

    @property
    def N_model_nodes(self):
        return len(self.get_model_nodes())

    def print_tree(self):
        print(self.label)
        

    def __str__(self):
        return self.label
                
    def __repr__(self):
        if self.is_leaf:
            s = "<{} '{}', parent='{}'>".format(self.__class__,
                                                        self.label,
                                                        self.parent)
        else:
            child_labels = [str(c) for c in self.children]
            s = "<{} '{}', parent='{}', children={}>".format(self.__class__,
                                                        self.label,
                                                        self.parent,
                                                        child_labels)
        return s
    
class ObsNode(Node):
    def __init__(self, observation, source, ref_node=None):

        self.observation = observation
        self.source = source
        self.reference = ref_node
        
        self.children = []
        self.parent = None
        self._leaves = None
        
        #indices of underlying models, defining physical systems        
        self._inds = None 
        self._n_params = None
        self._Nstars = None

        #for model_mag caching
        self._cache_key = None
        self._cache_val = None    

    @property
    def instrument(self):
        return self.observation.name
    
    @property
    def band(self):
        return self.observation.band

    @property
    def value(self):
        return (self.source.mag, self.source.e_mag)
    
    @property
    def resolution(self):
        return self.observation.resolution
    
    @property
    def relative(self):
        return self.source.relative
    
    @property
    def separation(self):
        return self.source.separation
    
    @property
    def pa(self):
        return self.source.pa
    

    @property
    def value_str(self):
        return '({:.2f}, {:.2f})'.format(*self.value)
    

    def distance(self, other):
        """Coordinate distance from another ObsNode
        """
        return distance((self.separation, self.pa), (other.separation, other.pa))
        
        r0, pa0 = (self.separation, self.pa)
        #logging.debug('r0={}, pa0={} (from {})'.format(r0, pa0, self))
        ra0 = r0*np.sin(pa0*np.pi/180)
        dec0 = r0*np.cos(pa0*np.pi/180)
        
        r1, pa1 = (other.separation, other.pa)
        #logging.debug('r1={}, pa1={} (from {})'.format(r0, pa0, other))
        ra1 = r1*np.sin(pa1*np.pi/180)
        dec1 = r1*np.cos(pa1*np.pi/180)

        dra = (ra1 - ra0)
        ddec = (dec1 - dec0)
        return np.sqrt(dra**2 + ddec**2)
        
    def _in_same_observation(self, other):
        return self.instrument==other.instrument and self.band==other.band

    @property
    def n_params(self):
        if self._n_params is None:
            self._n_params = 5 * len(self.leaves)
        return self._n_params
        
    def _get_inds(self):
        inds = [n.index for n in self.leaves]
        inds = list(set(inds))
        inds.sort()
        return inds
    
    def _clear_leaves(self):
        self._leaves = None
        self._inds = None
        self._n_params = None
        self._Nstars = None
        
    @property
    def Nstars(self):
        """
        dictionary of number of stars per system
        """
        if self._Nstars is None:
            N = {}
            for n in self.get_model_nodes():
                if n.index not in N:
                    N[n.index] = 1
                else:
                    N[n.index] += 1
            self._Nstars = N
        return self._Nstars
    

    @property
    def systems(self):
        lst = self.Nstars.keys()
        lst.sort()
        return lst

    @property
    def inds(self):
        if self._inds is None:
            self._inds = self._get_inds()
        return self._inds
    
    @property
    def label(self):
        if self.source.relative:
            band_str = 'delta-{}'.format(self.band)
        else:
            band_str = self.band
        return '{} {}={} @({:.2f}, {:.0f} [{:.2f}])'.format(self.instrument, 
                                                           band_str,
                                self.value_str, self.separation, self.pa,
                                                           self.resolution)

    @property
    def obsname(self):
        return '{}-{}'.format(self.instrument, self.band)    

    def get_system(self, ind):
        system = []
        for l in self.get_root().leaves:
            try:
                if l.index==ind:
                    system.append(l)
            except AttributeError:
                pass
        return system

    def add_model(self, ic, N=1, index=0):
        """
        Should only be able to do this to a leaf node.

        Either N and index both integers OR index is 
        list of length=N
        """
        if type(index) in [list,tuple]:
            if len(index) != N:
                raise ValueError('If a list, index must be of length N.')
        else:
            index = [index]*N

        for idx in index:
            existing = self.get_system(idx)
            tag = len(existing)
            self.add_child(ModelNode(ic, index=idx, tag=tag))
            
    def model_mag(self, pardict, use_cache=True):
        """
        pardict is a dictionary of parameters for all leaves
        gets converted back to traditional parameter vector
        """
        if pardict == self._cache_key and use_cache:
            #print('{}: using cached'.format(self))
            return self._cache_val

        #print('{}: calculating'.format(self))
        self._cache_key = pardict


        # Generate appropriate parameter vector from dictionary
        p = []
        for l in self.leaf_labels:
            p.extend(pardict[l])

        assert len(p) == self.n_params

        tot = np.inf
        #print('Building {} mag for {}:'.format(self.band, self))
        for i,m in enumerate(self.leaves):
            mag = m.evaluate(p[i*5:(i+1)*5], self.band)
            # logging.debug('{}: mag={}'.format(self,mag))
            #print('{}: {}({}) = {}'.format(m,self.band,p[i*5:(i+1)*5],mag))
            tot = addmags(tot, mag)

        self._cache_val = tot
        return tot

    def lnlike(self, pardict, use_cache=True):
        """
        returns log-likelihood of this observation

        pardict is a dictionary of parameters for all leaves
        gets converted back to traditional parameter vector
        """

        mag, dmag = self.value
        if np.isnan(dmag):
            return 0
        if self.relative:
            # If this *is* the reference, just return
            if self.reference is None:
                return 0
            mod = (self.model_mag(pardict, use_cache=use_cache) - 
                   self.reference.model_mag(pardict, use_cache=use_cache))
            mag -= self.reference.value[0]
        else:
            mod = self.model_mag(pardict, use_cache=use_cache)

        lnl = -0.5*(mag - mod)**2 / dmag**2

        # logging.debug('{} {}: mag={}, mod={}, lnlike={}'.format(self.instrument,
        #                                                         self.band,
        #                                                         mag,mod,lnl))
        return lnl

class DummyObsNode(ObsNode):
    def __init__(self, *args, **kwargs):
        self.observation = None
        self.source = None
        self.reference = None
        
        self.children = []
        self.parent = None
        self._leaves = None
        
        #indices of underlying models, defining physical systems        
        self._inds = None 
        self._n_params = None
        self._Nstars = None

        #for model_mag caching
        self._cache_key = None
        self._cache_val = None    

    @property
    def label(self):
        return '[dummy]'

    @property
    def value(self):
        return None, None

    def lnlike(self, *args, **kwargs):
        return 0

        
class ModelNode(Node):
    """
    These are always leaves; leaves are always these.

    Index keeps track of which physical system node is in.
    """
    def __init__(self, ic, index=0, tag=0):
        self._ic = ic
        self.index = index
        self.tag = tag
        
        self.children = []
        self.parent = None
        self._leaves = None

    @property
    def label(self):
        return '{}_{}'.format(self.index, self.tag)
        
    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic        

    def get_obs_ancestors(self):
        nodes = self.get_ancestors()
        return [n for n in nodes if isinstance(n, ObsNode)]

    @property
    def contributing_observations(self):
        """The instrument-band for all the observations feeding into this model node
        """
        return [n.obsname for n in self.get_obs_ancestors()]

    def evaluate(self, p, prop):
        if prop in self.ic.bands:
            return self.evaluate_mag(p, prop)
        elif prop=='mass':
            return p[0]
        elif prop=='age':
            return p[1]
        elif prop=='feh':
            return p[2]
        elif prop in ['Teff','logg','radius']:
            return getattr(self.ic, prop)(*p[:3])
        else:
            raise ValueError('property {} cannot be evaluated by Isochrone.'.format(prop))

    def evaluate_mag(self, p, band):
        return self.ic.mag[band](*p)

    def lnlike(self, *args, **kwargs):
        return 0        

class Source(object):
    def __init__(self, mag, e_mag, separation=0., pa=0.,
                relative=False, is_reference=False):
        self.mag = float(mag)
        self.e_mag = float(e_mag)
        self.separation = float(separation)
        self.pa = float(pa)
        self.relative = bool(relative)
        self.is_reference = bool(is_reference)

    def __str__(self):
        return '({}, {}) @({}, {})'.format(self.mag, self.e_mag,
                                            self.separation, self.pa)

    def __repr__(self):
        return self.__str__()

class Star(object):
    """Theoretical counterpart of Source. 
    """
    def __init__(self, pars, separation, pa):
        self.pars = pars
        self.separation = separation
        self.pa = pa        
        
    def distance(self, other):
        return distance((self.separation, self.pa),
                        (other.separation, other.pa))


class Observation(object):
    """
    Contains relevant information about imaging observation

    name: identifying string (typically the instrument)
    band: photometric bandpass
    resolution: *approximate* angular resolution of instrument.
         used for source matching between observations
    sources: list of Source objects

    """
    def __init__(self, name, band, resolution, sources=None,
                relative=False):
        self.name = name
        self.band = band
        self.resolution = resolution
        if sources is not None:
            if not np.all(type(s)==Source for s in sources):
                raise ValueError('Source list must be all Source objects.')

        self.sources = []
        if sources is None:
            sources = []
        for s in sources:
            self.add_source(s)

        self.relative = relative
        self._set_reference()

    def observe(self, stars, unc, ic=None):
        """Creates and adds appropriate synthetic Source objects for list of stars (max 2 for now)
        """
        if ic is None:
            ic = get_ichrone('mist')

        if len(stars) > 2:
            raise NotImplementedError('No support yet for > 2 synthetic stars')
            
        mags = [ic(*s.pars)['{}_mag'.format(self.band)].values[0] for s in stars]

        d = stars[0].distance(stars[1])
        
        if d < self.resolution:
            mag = addmags(*mags) + unc*np.random.randn()
            sources = [Source(mag, unc, stars[0].separation, stars[0].pa, 
                           relative=self.relative)]
        else:
            mags = np.array([m + unc*np.random.randn() for m in mags])
            if self.relative:
                mags -= mags.min()
            sources = [Source(m, unc, s.separation, s.pa, relative=self.relative)
                        for m,s in zip(mags, stars)]
        
        for s in sources:
            self.add_source(s)

        self._set_reference()

    def add_source(self, source):
        """
        Adds source to observation, keeping sorted order (in separation)
        """
        if not type(source)==Source:
            raise TypeError('Can only add Source object.')

        if len(self.sources)==0:
            self.sources.append(source)
        else:
            ind = 0
            for s in self.sources:
                # Keep sorted order of separation
                if source.separation < s.separation: 
                    break
                ind += 1

            self.sources.insert(ind, source)

        #self._set_reference()

    @property
    def brightest(self):
        mag0 = np.inf
        s0 = None
        for s in self.sources:
            if s.mag < mag0:
                mag0 = s.mag
                s0 = s
        return s0

    def _set_reference(self):
        """If relative, make sure reference node is set to brightest.
        """
        if len(self.sources) > 0:
            self.brightest.is_reference = True

    def __str__(self):
        return '{}-{}'.format(self.name, self.band)
    
    def __repr__(self):
        return str(self)
        
class ObservationTree(Node):
    """Builds a tree of Nodes from a list of Observation objects
    
    Organizes Observations from smallest to largest resolution,
    and at each stage attaches each source to the most probable
    match from the previous Observation.  Admittedly somewhat hack-y,
    but should *usually* do the right thing.  Check out `obs.print_ascii()`
    to visualize what this has done.
    """
    spec_props = ['Teff', 'logg', 'feh']

    def __init__(self, observations=None, name=None):
        
        if observations is None:
            observations = []
        
        if name is None:
            self.label = 'root'
        else:
            self.label = name
        self.parent = None

        self._observations = []
        self._build_tree()

        [self.add_observation(obs) for obs in observations]

        self._N = None
        self._index = None

        # Spectroscopic properties
        self.spectroscopy = {}

        # Limits (such as minimum on logg)
        self.limits = {}
        
        # Parallax measurements
        self.parallax = {}

        # This will be calculated and set at first access
        self._Nstars = None

        #likelihood cache
        self._cache_key = None
        self._cache_val = None

    @property
    def name(self):
        return self.label
    
    def _clear_cache(self):
        self._cache_key = None
        self._cache_val = None

    @classmethod
    def from_df(cls, df, **kwargs):
        """
        DataFrame must have the right columns.

        these are: name, band, resolution, mag, e_mag, separation, pa
        """
        tree = cls(**kwargs)

        for (n,b), g in df.groupby(['name','band']):
            #g.sort('separation', inplace=True) #ensures that the first is reference
            sources = [Source(**s[['mag','e_mag','separation','pa','relative']]) 
                        for _,s in g.iterrows()]
            obs = Observation(n, b, g.resolution.mean(),
                              sources=sources, relative=g.relative.any())
            tree.add_observation(obs)

        # For all relative mags, set reference to be brightest 



        return tree

    @classmethod
    def from_ini(cls, filename):
        config = ConfigObj(filename)

    def to_df(self):
        """
        Returns DataFrame with photometry from observations organized.

        This DataFrame should be able to be read back in to
        reconstruct the observation.
        """
        df = pd.DataFrame()
        name = []
        band = []
        resolution = []
        mag = []
        e_mag = []
        separation = []
        pa = []
        relative = []
        for o in self._observations:
            for s in o.sources:
                name.append(o.name)
                band.append(o.band)
                resolution.append(o.resolution)
                mag.append(s.mag)
                e_mag.append(s.e_mag)
                separation.append(s.separation)
                pa.append(s.pa)
                relative.append(s.relative)

        return pd.DataFrame({'name':name,'band':band,'resolution':resolution,
                             'mag':mag,'e_mag':e_mag,'separation':separation,
                             'pa':pa,'relative':relative})

    def save_hdf(self, filename, path='', overwrite=False, append=False):
        """
        Writes all info necessary to recreate object to HDF file
        
        Saves table of photometry in DataFrame

        Saves model specification, spectroscopy, parallax to attrs
        """

        if os.path.exists(filename):
            store = pd.HDFStore(filename)
            if path in store:
                store.close()
                if overwrite:
                    os.remove(filename)
                elif not append:
                    raise IOError('{} in {} exists.  Set either overwrite or append option.'.format(path,filename))
            else:
                store.close()

        df = self.to_df()
        df.to_hdf(filename, path+'/df')
        with pd.HDFStore(filename) as store:
            # store = pd.HDFStore(filename)
            attrs = store.get_storer(path+'/df').attrs
            attrs.spectroscopy = self.spectroscopy
            attrs.parallax = self.parallax
            attrs.N = self._N
            attrs.index = self._index
            store.close()

    @classmethod
    def load_hdf(cls, filename, path='', ic=None):
        """
        Loads stored ObservationTree from file.

        You can provide the isochrone to use; or it will default to MIST

        TODO: saving and loading must be fixed!  save ic type, bands, etc.
        """
        store = pd.HDFStore(filename)
        try:
            samples = store[path+'/df']
            attrs = store.get_storer(path+'/df').attrs        
        except:
            store.close()
            raise
        df = store[path+'/df']
        new = cls.from_df(df)
        
        if ic is None:
            ic = get_ichrone('mist')

        new.define_models(ic, N=attrs.N, index=attrs.index)
        new.spectroscopy = attrs.spectroscopy
        new.parallax = attrs.parallax
        store.close()
        return new


    def add_observation(self, obs):
        """Adds an observation to observation list, keeping proper order        
        """
        if len(self._observations)==0:
            self._observations.append(obs)
        else:
            res = obs.resolution
            ind = 0
            for o in self._observations:
                if res > o.resolution:
                    break
                ind += 1
            self._observations.insert(ind, obs)
        
        self._build_tree()
        self._clear_cache()
        
    def add_spectroscopy(self, label='0_0', **props):
        """
        Adds spectroscopic measurement to particular star(s) (corresponding to individual model node)

        Default 0_0 should be primary star

        legal inputs are 'Teff', 'logg', 'feh', and in form (val, err)
        """
        if label not in self.leaf_labels:
            raise ValueError('No model node named {} (must be in {}). Maybe define models first?'.format(label, self.leaf_labels))
        for k,v in props.items():
            if k not in self.spec_props:
                raise ValueError('Illegal property {} (only {} allowed).'.format(k, self.spec_props))
            if len(v) != 2:
                raise ValueError('Must provide (value, uncertainty) for {}.'.format(k))

        if label not in self.spectroscopy:
            self.spectroscopy[label] = {}

        for k,v in props.items():
            self.spectroscopy[label][k] = v

        self._clear_cache()

    def add_limit(self, label='0_0', **props):
        """Define limits to spectroscopic property of particular stars.

        Usually will be used for 'logg', but 'Teff' and 'feh' will also work.

        In form (min, max): e.g., t.add_limit(logg=(3.0,None))  

        None will be converted to (-)np.inf
        """

        if label not in self.leaf_labels:
            raise ValueError('No model node named {} (must be in {}). Maybe define models first?'.format(label, self.leaf_labels))
        for k,v in props.items():
            if k not in self.spec_props:
                raise ValueError('Illegal property {} (only {} allowed).'.format(k, self.spec_props))
            if len(v) != 2:
                raise ValueError('Must provide (min, max) for {}. (`None` is allowed value)'.format(k))

        if label not in self.limits:
            self.limits[label] = {}

        for k,v in props.items():
            vmin, vmax = v
            if vmin is None:
                vmin = -np.inf
            if vmax is None:
                vmax = np.inf
            self.limits[label][k] = (vmin, vmax)

        self._clear_cache()


    def add_parallax(self, plax, system=0):
        if len(plax)!=2:
            raise ValueError('Must enter (value,uncertainty).')
        if system not in self.systems:
            raise ValueError('{} not in systems ({}).'.format(system,self.systems))

        self.parallax[system] = plax
        self._clear_cache()

    def define_models(self, ic, leaves=None, N=1, index=0):
        """
        N, index are either integers or lists of integers.

        N : number of model stars per observed star
        index : index of physical association

        leaves: either a list of leaves, or a pattern by which 
        the leaves are selected (via `select_leaves`)

        If these are lists, then they are defined individually for 
        each leaf.

        If `index` is a list, then each entry must be either
        an integer or a list of length `N` (where `N` is the corresponding
            entry in the `N` list.)  

        This bugs up if you call it multiple times.  If you want
        to re-do a call to this function, please re-define the tree.
        """

        self.clear_models()

        if leaves is None:
            leaves = self._get_leaves()
        elif type(leaves)==type(''):
            leaves = self.select_leaves(leaves)

        # Sort leaves by distance, to ensure system 0 will be assigned
        # to the main reference star.


        if np.isscalar(N):
            N = (np.ones(len(leaves))*N)
            #if np.size(index) > 1:
            #    index = [index]
        N = np.array(N).astype(int)

        if np.isscalar(index):
            index = (np.ones_like(N)*index)
        index = np.array(index).astype(int)

        # Add the appropriate number of model nodes to each
        #  star in the highest-resoluion image
        for s,n,i in zip(leaves, N, index):
            # Remove any previous model nodes (should do some checks here?)
            s.remove_children()
            s.add_model(ic, n, i)

        # For each system, make sure tag _0 is the brightest.
        self._fix_labels()

        self._N = N
        self._index = index

        self._clear_all_leaves()

    def _fix_labels(self):
        """For each system, make sure tag _0 is the brightest, and make sure 
        system 0 contains the brightest star in the highest-resolution image
        """
        for s in self.systems:
            mag0 = np.inf
            n0 = None
            for n in self.get_system(s):
                mag, _ = n.parent.value
                if mag < mag0:
                     mag0 = mag
                     n0 = n

            # If brightest is not tag _0, then switch them.
            if n0.tag != 0:
                n_other = self.get_leaf('{}_{}'.format(s,0))        
                n_other.tag = n0.tag
                n0.tag = 0



    def get_system(self, ind):
        system = []
        for l in self.leaves:
            try:
                if l.index==ind:
                    system.append(l)
            except AttributeError:
                pass
        return system

    @property    
    def observations(self):
        return self._observations

    def select_observations(self, name):
        """Returns nodes whose instrument-band matches 'name'
        """
        return [n for n in self.get_obs_nodes() if n.obsname==name]

    def clear_models(self):
        for n in self:
            if isinstance(n, ModelNode):
                n.parent.remove_child(n.label)

        self._clear_all_leaves()

    def trim(self):
        """
        Trims leaves from tree that are not observed at highest-resolution level

        This is a bit hacky-- what it does is 
        """
        # Only allow leaves to stay on list (highest-resolution) level
        return

        for l in self._levels[-2::-1]:
            for n in l:
                if n.is_leaf:
                    n.parent.remove_child(n.label)

        self._clear_all_leaves() #clears cached list of leaves

    def p2pardict(self, p):
        """
        Given leaf labels, turns parameter vector into pardict
        """
        d = {}
        N = self.Nstars
        i = 0
        for s in self.systems:
            age, feh, dist, AV = p[i+N[s]:i+N[s]+4]
            for j in xrange(N[s]):
                l = '{}_{}'.format(s,j)
                mass = p[i+j]
                d[l] = [mass, age, feh, dist, AV]
            i += N[s] + 4
        return d

    @property
    def param_description(self):
        N = self.Nstars
        pars = []
        for s in self.systems:
            for j in xrange(N[s]):
                pars.append('mass_{}_{}'.format(s,j))
            for p in ['age', 'feh', 'distance', 'AV']:
                pars.append('{}_{}'.format(p,s))
        return pars

    @property
    def Nstars(self):
        if self._Nstars is None:
            N = {}
            for n in self.get_model_nodes():
                if n.index not in N:
                    N[n.index] = 1
                else:
                    N[n.index] += 1
            self._Nstars = N

        return self._Nstars

    @property
    def systems(self):
        # fix this! make sure it is unique!!!
        lst = list(chain(*[c.systems for c in self.children]))
        return sorted(set(lst))
    
    def print_ascii(self, fout=None, p=None):
        pardict = None
        if p is not None:
            pardict = self.p2pardict(p)
        super(ObservationTree, self).print_ascii(fout, pardict)

    def lnlike(self, p, use_cache=True):
        """
        takes parameter vector, constructs pardict, returns sum of lnlikes of non-leaf nodes
        """
        if np.all(p==self._cache_key) and use_cache:
            return self._cache_val
        self._cache_key = p

        pardict = self.p2pardict(p)

        # lnlike from photometry
        lnl = 0
        for n in self:
            if n is not self:
                lnl += n.lnlike(pardict, use_cache=use_cache)
            if not np.isfinite(lnl):
                self._cache_val = -np.inf
                return -np.inf

        # lnlike from spectroscopy
        for l in self.spectroscopy:
            for prop,(val,err) in self.spectroscopy[l].items():
                mod = self.get_leaf(l).evaluate(pardict[l], prop)
                lnl += -0.5*(val - mod)**2/err**2
            if not np.isfinite(lnl):
                self._cache_val = -np.inf
                return -np.inf

        # enforce limits
        for l in self.limits:
            for prop,(vmin,vmax) in self.limits[l].items():
                mod = self.get_leaf(l).evaluate(pardict[l], prop)
                if mod < vmin or mod > vmax or not np.isfinite(mod):
                    self._cache_val = -np.inf
                    return -np.inf

        # lnlike from parallax
        for s,(val,err) in self.parallax.items():
            dist = pardict['{}_0'.format(s)][3]
            mod = 1./dist * 1000.
            lnl += -0.5*(val-mod)**2/err**2

        if not np.isfinite(lnl):
            self._cache_val = -np.inf
            return -np.inf

        self._cache_val = lnl
        return lnl

    def _find_closest(self, n0):
        """returns the node in the tree that is closest to n0, but not
          in the same observation
        """
        dmin = np.inf
        nclose = None

        ds = []
        nodes = []
        
        ds.append(np.inf)
        nodes.append(self)

        for n in self:
            if n is n0:
                continue
            try:
                if n._in_same_observation(n0):
                    continue
                ds.append(n.distance(n0))
                nodes.append(n)
            except AttributeError:
                pass

        inds = np.argsort(ds)
        ds = [ds[i] for i in inds]
        nodes = [nodes[i] for i in inds]
        
        for d,n in zip(ds, nodes):
            try:
                if d < n.resolution or n.resolution==-1:
                    return n
            except AttributeError:
                pass

        # If nothing else works
        return self

    def _build_tree(self):
        #reset leaf cache, children
        self._clear_all_leaves()
        self.children = []

        for i,o in enumerate(self._observations):
            s0 = o.brightest
            ref_node = ObsNode(o, s0)
            for s in o.sources:
                if s.relative and not s.is_reference:
                    node = ObsNode(o, s, ref_node=ref_node)
                elif s.relative and s.is_reference:
                    node = ref_node
                else:
                    node = ObsNode(o, s)
                
                # For first level, no need to choose parent
                if i==0:
                    parent = self
                else:
                    # Find parent (closest node in tree)
                    parent = self._find_closest(node)
                        
                parent.add_child(node)

        # If after all this, there are no `ObsNode` nodes,
        # then add a dummy.
        if len(self.get_obs_nodes())==0:
            self.add_child(DummyObsNode())

    @classmethod
    def synthetic(cls, stars, surveys):
        pass

