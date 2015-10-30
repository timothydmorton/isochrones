import numpy as np
import logging

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
                 nodetype='absolute',
                 reference=None):

        self.instrument = instrument
        self.band = band
        self.value = value
        self.nodetype = nodetype
        self.reference = reference
        
        self.children = []
        self.parent = None
        self._leaves = None
        
        #indices of underlying models, defining physical systems        
        self._inds = None 
        self._n_params = None
        self._Nstars = None
        
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
        return '{} {}={}'.format(self.instrument, self.band,
                                self.value)

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
        if self.nodetype=='absolute':
            mod = self.model_mag(p)
        elif self.nodetype=='relative':
            mod = self.model_mag(p) - self.reference.model_mag(p)

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
        