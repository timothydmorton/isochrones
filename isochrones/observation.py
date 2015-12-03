from __future__ import print_function, division
import numpy as np
import logging

from asciitree import LeftAligned, Traversal
from asciitree.drawing import BoxStyle, BOX_DOUBLE, BOX_BLANK

from itertools import chain, imap, izip, count
from collections import OrderedDict

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
        if self.pars is not None and hasattr(node, 'model_mag'):
            text += '; model={:.2f} ({})'.format(node.model_mag(self.pars),
                                                 node.lnlike(self.pars))
        if type(node)==ModelNode:
            text += ': ({})'.format(self.pars['{}_{}'.format(node.index,node.tag)])
        return text

class MyLeftAligned(LeftAligned):
    """For custom ascii tree printing
    """
    pars = None
    def __init__(self, pars=None, **kwargs):
        self.pars = pars
        self.traverse = NodeTraversal(pars)
        super(MyLeftAligned,self).__init__(**kwargs)
    
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
        
    def print_ascii(self, pars=None):
        box_tr = MyLeftAligned(pars,draw=BoxStyle(gfx=BOX_DOUBLE, horiz_len=1))
        print(box_tr(self))
        
    @property
    def is_leaf(self):
        return len(self.children)==0

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
        
    @property
    def leaf_labels(self):
        return [l.label for l in self.leaves]
        
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
    def __init__(self, instrument, band, value,
                 separation=0., pa=0.,
                 relative=False,
                 reference=None):

        self.instrument = instrument
        self.band = band
        self.value = value
        self.relative = relative
        self.reference = reference
        
        self.separation = separation
        self.pa = pa
        
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
        
    def distance(self, other):
        """Coordinate distance from another ObsNode
        """
        r0, pa0 = (self.separation, self.pa)
        ra0 = r0*np.sin(pa0*np.pi/180)
        dec0 = r0*np.cos(pa0*np.pi/180)
        
        r1, pa1 = (other.separation, other.pa)
        ra1 = r1*np.sin(pa1*np.pi/180)
        dec1 = r1*np.cos(pa1*np.pi/180)

        dra = (ra1 - ra0)
        ddec = (dec1 - dec0)
        return np.sqrt(dra**2 + ddec**2)
        
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
            for n in self.leaves:
                if n.index not in N:
                    N[n.index] = 1
                else:
                    N[n.index] += 1
            self._Nstars = N
        return self._Nstars
        
    @property
    def systems(self):
        lst = self._Nstars.keys()
        lst.sort()
        return lst

    @property
    def inds(self):
        if self._inds is None:
            self._inds = self._get_inds()
        return self._inds
    
    @property
    def label(self):
        return '{} {}={} @({:.2f}, {:.0f})'.format(self.instrument, self.band,
                                self.value, self.separation, self.pa)

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

        """
        existing = self.get_system(index)
        initial_tag = len(existing) # + chr(65) is 'A' if I want to go back to letters

        for i in range(N):            
            #tag = chr(initial_tag+i)
            tag = initial_tag + i
            self.add_child(ModelNode(ic, index=index, tag=tag))
            
    def model_mag(self, pardict):
        """
        pardict is a dictionary of parameters for all leaves
        gets converted back to traditional parameter vector
        """
        if pardict == self._cache_key:
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
        print('Building {} mag for {}:'.format(self.band, self))
        for i,m in enumerate(self.leaves):
            mag = m.evaluate(p[i*5:(i+1)*5], self.band)
            print('{}: {}({}) = {}'.format(m,self.band,p[i*5:(i+1)*5],mag))
            tot = addmags(tot, mag)

        self._cache_val = tot
        return tot

    def lnlike(self, pardict):
        """
        returns log-likelihood of this observation

        pardict is a dictionary of parameters for all leaves
        gets converted back to traditional parameter vector
        """

        mag, dmag = self.value
        if self.relative:
            # If this *is* the reference, just return
            if self.reference is None:
                return 0
            mod = self.model_mag(pardict) - self.reference.model_mag(pardict)
        else:
            mod = self.model_mag(pardict)

        return -0.5*(mag - mod)**2 / dmag**2

        
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

    def evaluate(self, p, band):
        return self.ic.mag[band](*p)

    def lnlike(self, *args):
        return 0        

class Source(object):
    def __init__(self, mag, e_mag, separation=0., pa=0.,
                relative=False, reference=False):
        self.mag = mag
        self.e_mag = e_mag
        self.separation = separation
        self.pa = pa
        self.relative = relative
        self.reference = reference


class Observation(object):
    """
    Contains relevant information about imaging observation

    name: identifying string (typically the instrument)
    band: photometric bandpass
    resolution: *approximate* angular resolution of instrument.
          Used only to order Observation objects within the 
          observation tree.
    sources: list of Source objects

    """
    def __init__(self, name, band, resolution, sources=None,
                relative=False):
        self.name = name
        self.band = band
        self.resolution = resolution
        self.sources = []
        if sources is None:
            sources = []
        for s in sources:
            self.add_source(s)
        
    def add_source(self, source):
        """
        Adds source to observation, keeping sorted order
        """
        if not type(source)==Source:
            raise TypeError('Can only add Source object.')

        if len(self.sources)==0:
            self.sources.append(source)
        else:
            ind = 0
            for s in self.sources:
                if source.mag < s.mag:
                    break
                ind += 1

            self.sources.insert(ind, source)
        
    def __str__(self):
        return '{}-{}'.format(self.name, self.band)
    
    def __repr__(self):
        return str(self)
        
class ObservationTree(Node):
    """Builds a tree of Nodes from a list of Observation objects
    
    Organizes Observations from smallest to largest resolution,
    and at each stage attaches each source to the most probable
    match from the previous Observation.
    """
    def __init__(self, observations=None):
        
        if observations is None:
            observations = []
        
        self.label = 'root'
        self.parent = None

        self._levels = []
        self._observations = []
        self._build_tree()

        [self.add_observation(obs) for obs in observations]
        
        #likelihood cache
        self._cache_key = None
        self._cache_val = None

    @classmethod
    def from_df(cls, df):
        """
        DataFrame must have the right columns.

        these are: name, band, resolution, mag, e_mag, separation, pa
        """
        tree = cls()

        for (n,b), g in df.groupby(['name','band']):
            #g.sort('mag', inplace=True)
            sources = [Source(**s[['mag','e_mag','separation','pa','relative']]) 
                        for _,s in g.iterrows()]
            obs = Observation(n, b, g.resolution.mean(),
                              sources=sources, relative=g.relative.any())
            tree.add_observation(obs)

        return tree

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
        
    def define_models(self, ic, N=1, index=0):
        """
        N, index are either integers or lists of integers.

        N : number of model stars per observed star
        index : index of physical association

        If these are lists, then they are defined individually for 
        each star in the final level (highest-resoluion)
        """

        if np.size(N)==1:
            N = (np.ones(len(self._levels[-1]))*N).astype(int)
        if np.size(index)==1:
            index = (np.ones_like(N)*index).astype(int)

        # Add the appropriate number of model nodes to each
        #  star in the highest-resoluion image
        for s,n,i in zip(self._levels[-1], N, index):
            # Remove any previous model nodes (should do some checks here?)
            s.remove_children()
            s.add_model(ic, n, i)

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
                d[l] = [mass, age, feh, dist ,AV]
            i += N[s] + 4
        return d


    @property
    def Nstars(self):
        return self.children[0].Nstars    

    @property
    def systems(self):
        return self.children[0].systems
    

    def lnlike(self, p):
        """
        takes parameter vector, constructs pardict, returns sum of lnlikes of non-leaf nodes
        """
        if p==self._cache_key:
            return self._cache_val
        self._cache_key = p

        pardict = self.p2pardict(p)

        lnl = 0
        for n in self:
            if n is not self:
                lnl += n.lnlike(pardict)
        self._cache_val = lnl
        return lnl


    def _build_tree(self):
        """Constructs tree from [ordered] list of observations
        """
        #reset leaf cache, children
        self._clear_all_leaves()
        self.children = []
        self._levels = []
        
        for i,o in enumerate(self._observations):
            self._levels.append([])

            ref_node = None
            for s in o.sources:
                if s.relative and ref_node is None:
                    node = ObsNode(o.name, o.band,
                                       (s.mag, s.e_mag), 
                                        relative=True,
                                        reference=None)
                    ref_node = node
                else:
                    node = ObsNode(o.name, o.band, 
                                   (s.mag, s.e_mag),
                                   separation=s.separation, pa=s.pa,
                                   relative=s.relative,
                                   reference=ref_node)

                # For first level, no need to choose parent
                if i==0:
                    parent = self
                else:
                    # Loop through nodes of level above, choose
                    #  parent to be the closest one.
                    d_min = np.inf
                    for n in self._levels[i-1]:
                        d = node.distance(n)
                        if d < d_min:
                            d_min = d
                            parent = n
                        
                parent.add_child(node)
                self._levels[i].append(node)
        
            
