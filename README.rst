isochrones
==========
http://dx.doi.org/10.5281/zenodo.8475

Provides simple interface for interacting with stellar model grids.  The guts of this code is a 3-d linear interpolation in mass, age, Fe/H space.  That is, the model predicts the various properties as functions of these inputs.

Basic usage::

    >>> from isochrones.dartmouth import Dartmouth_Isochrone
    >>> dar = Dartmouth_Isochrone()
    >>> dar.radius(1.0,9.6,0.0) #mass [solar], log(age), Fe/H
    0.9886235
    
or, more generally (results returned in ``pandas`` ``DataFrame`` object)::

    >>> dar(1.0,9.6,0.0)
          B_mag   D51_mag     H_mag     I_mag     J_mag     K_mag  Kepler_mag  \
    0  5.430236  4.933568  3.334693  4.082671  3.645123  3.300209    4.536936   
    R_mag         Teff     U_mag    V_mag  age     g_mag     i_mag  \
    0  4.408233  5845.864438  5.624884  4.76735  9.6  5.060191  4.487868   
    logL      logg  mass     r_mag    radius     z_mag  
    0  0.010176  4.448366     1  4.589966  0.988624  4.482989  

    
Note that the first time a stellar model module is loaded, it will download the necessary data (~few hundred MB) and save it to a folder in your home directory called ~/.isochrones.
