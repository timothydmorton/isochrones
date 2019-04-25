.. _api:

API
===

.. module:: isochrones

This page details the methods and classes provided by the
``isochrones`` package.  The :class:`ModelGrid` object handles
the book-keeping aspects of loading a given grid into a ``pandas.DataFrame``
object.  However, :class:`ModelGrid` works behind the scenes; you should
usually only work directly through the :class:`Isochrone` object, which handles the grid interpolation. Finally, the :class:`StarModel` object is used to fit the stellar models constrained by observed data.

ModelGrid
---------

.. autoclass:: isochrones.grid.ModelGrid
    :members:

Isochrone
--------

Any usage of ``isochrones`` involves instantiating an
:class:`Isochrone`, usually through a specific subclass such as
:class:`mist.MIST_Isochrone`.

.. autoclass:: isochrones.Isochrone
   :members:

.. autoclass:: isochrones.FastIsochrone
   :members:    

MIST
---------

Stellar model grids from the `MESA Isochrones and Stellar Tracks 
<http://waps.cfa.harvard.edu/MIST/>`_.  These grids cover a larger range in mass, 
age, and [Fe/H], compared to the Dartmouth grids, and are thus preferred.
Because this is a larger grid, interpolation here is implemented via the
:class:`FastIsochrone` implementation, rather than the standard Delaunay 
triangulation-based :class:`Isochrone` implementation.  

.. autoclass:: isochrones.mist.MIST_Isochrone
   :members:

Dartmouth
-----------

Stellar model grids from the `Dartmouth Stellar Evolution Database
<http://stellar.dartmouth.edu/models/>`_.  Because these grids contain
fewer points, both interpolation options are available: :class:`Dartmouth_Isochrone` and :class:`Dartmouth_FastIsochrone`.

.. autoclass:: isochrones.dartmouth.Dartmouth_Isochrone
   :members:

.. autoclass:: isochrones.dartmouth.Dartmouth_FastIsochrones
   :members:

StarModel
---------

Estimating a star's physical properties based on a set of observed
properties involves instantiating a :class:`StarModel`.

.. autoclass:: isochrones.StarModel
    :members:

