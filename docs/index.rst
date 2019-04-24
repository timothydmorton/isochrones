isochrones
===========

**Isochrones** is a python package that provides a simple interface to grids
of stellar evolution models, enabling the following common use cases:

  * Interpolating stellar model values at desired locations.
  * Generating properties of synthetic stellar populations.
  * Determining stellar properties of either single- or multiple-star systems,
    based on arbitrary observables.

The central goal of **isochrones** is to standardize model-grid-based stellar
parameter inference, and to enable such inference under different sets of
stellar models.  For now, only MIST models are included, but we hope to incorporate
YAPSI and PARSEC models as well.


.. toctree::
   :maxdepth: 2

   install.ipynb
   quickstart.ipynb
   interpolate.ipynb
   modelgrids.ipynb
   bc.ipynb
   grid_interpolator.ipynb
   starmodel.ipynb
   multiple.ipynb
