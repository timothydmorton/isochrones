.. _api:

API
===

.. module:: isochrones

This page details the methods and classes provided by the
``isochrones`` module.  The main workhorse for the stellar grid
interpolation is the :class:`Isochrone` object, and the
:class:`StarModel` object is used to fit the stellar models to
observational data.


Isochrone
--------

Any usage of ``isochrones`` will involve instantiating an
:class:`Isochrone`, though usually through a subclass such as
:class:`dartmouth.Dartmouth_Isochrone` object.

.. autoclass:: isochrones.Isochrone
   :members:

Dartmouth
---------

Stellar model grids from the `Dartmouth Stellar Evolution Database
<http://stellar.dartmouth.edu/models/>`_.  Most tested of the three
available grids, and therefore recommended.

.. autoclass:: isochrones.dartmouth.Dartmouth_Isochrone
   :members:


StarModel
---------

Estimating a star's physical properties based on a set of observed
properties will involve instantiating a :class:`StarModel`.

.. autoclass:: isochrones.StarModel
    :members:


