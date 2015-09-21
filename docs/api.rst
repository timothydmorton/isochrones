.. _api:

API
===

.. module:: isochrones

This page details the methods and classes provided by the
``isochrones`` package.  The main workhorse for the stellar grid
interpolation is the :class:`Isochrone` object, and the
:class:`StarModel` object is used to fit the stellar models to
observational data.


Isochrone
--------

Any usage of ``isochrones`` involves instantiating an
:class:`Isochrone`, though usually through a specific subclass such as
:class:`dartmouth.Dartmouth_Isochrone`.

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
properties involves instantiating a :class:`StarModel`.

.. autoclass:: isochrones.StarModel
    :members:

BinaryStarModel
---------

This allows fitting for the combined light of two stars; same distance,
age, and Fe/H.

.. autoclass:: isochrones.starmodel.BinaryStarModel
    :members:


TripleStarModel
---------

Just like :class:`BinaryStarModel`, but allows fitting for the
combined light of *three* stars; same distance,
age, and Fe/H.

.. autoclass:: isochrones.starmodel.TripleStarModel
    :members:




