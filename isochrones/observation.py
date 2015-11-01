import numpy as np
import logging

from asciitree import LeftAligned, Traversal
from asciitree.drawing import BoxStyle, BOX_DOUBLE, BOX_BLANK



class NodeTraversal(Traversal):
    """
    Custom subclass to traverse tree for ascii printing
    """
    def get_children(self, node):
        return node.children
    
    def get_root(self, node):
        return node.get_root()
    
    def get_text(self, node):
        return node.label
    
class MyLeftAligned(LeftAligned):
    """For custom ascii tree printing
    """
    traverse = NodeTraversal()

    
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

    @property
    def is_root(self):
        return self.parent is None

    def get_root(self):
        if self.is_root:
            return self
        else:
            return self.parent.get_root()
        
    def print_ascii(self):
        box_tr = MyLeftAligned(draw=BoxStyle(gfx=BOX_DOUBLE, horiz_len=1))
        print box_tr(self)
        
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
        return [l.label for l in J.leaves]
        
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
    def inds(self):
        if self._inds is None:
            self._inds = self._get_inds()
        return self._inds
    
    @property
    def label(self):
        return '{} {}={} @({:.2f}, {:.0f})'.format(self.instrument, self.band,
                                self.value, self.separation, self.pa)

    def get_system(self, ind):
        if self.is_leaf:
            return []
        else:
            return [l for l in self.leaves if l.index==ind]
    
    def add_model(self, ic, N=1, index=0):
        """
        Should only be able to do this to a leaf node.
        """
        existing = self.get_system(index)
        initial_tag = 65 + len(existing) #chr(65) is 'A'
        
        for i in range(N):            
            tag = chr(initial_tag+i)
            self.add_child(ModelNode(ic, index=index, tag=tag))
            
    def model_mag(self, p):
        tot = np.inf
        for i,m in enumerate(self.leaves):
            tot = addmags(tot, m.evaluate(p[i*5:(i+1)*5], self.band))
        return tot
            
    def lnlike(self, p):
        assert len(p) == self.n_params
        
        mag, dmag = self.value
        if self.relative:
            # If this *is* the reference, just return
            if self.reference is None:
                return 0
            mod = self.model_mag(p) - self.reference.model_mag(p)
        else:
            mod = self.model_mag(p)

        return -0.5*(mag - mod)**2 / dmag**2
        
class ModelNode(Node):
    """
    These are always leaves; leaves are always these.

    Index keeps track of which physical system node is in.
    """
    def __init__(self, ic, index=0, tag='A'):
        self._ic = ic
        self.index = index
        self.tag = tag
        
        self.children = []
        self.parent = None

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
        if sources is None:
            sources = []
        self.sources = sources
        
    def add_source(self, source):
        if not type(source)==Source:
            raise TypeError('Can only add Source object.')
        self.sources.append(source)
        
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
        
    def _build_tree(self):
        """Constructs tree from [ordered] list of observations
        """
        #reset leaf cache, children
        self._clear_all_leaves()
        self.children = []
        self._levels = []
        
        for i,o in enumerate(self._observations):
            self._levels.append([])
            for s in o.sources:
                ref_node = None
                if s.relative:
                    ref_node = ObsNode(o.name, o.band,
                                             (o.sources[0].mag, 
                                              o.sources[0].e_mag),
                                             relative=True,
                                             ref_node=None)
                    
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
        
            
