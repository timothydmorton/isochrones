isochrones
==========

Provides simple interface for interacting with stellar model grids.  The guts of this code is a 3-d linear interpolation in mass, age, Fe/H space.  That is, any model 

Basic usage::

    >>> from isochrones.dartmouth import Dartmouth_Isochrone
    >>> dar = Dartmouth_Isochrone()
    >>> dar.radius(1.0,9.6,0.0) #mass [solar], log(age), Fe/H
    0.9886235
    
Note that the first time a stellar model module is loaded, it will download the necessary data (~few hundred MB) and save it to a folder in your home directory called ~/.isochrones.
