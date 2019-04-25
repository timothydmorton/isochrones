isochrones
=======

The ``isochrones`` Python package aims to provide a 
simple common interface to different stellar model grids, and to 
simplify the task of inferring model-based physical stellar properties
given arbitrary observations of a star (or multiple stars).

The package is built around three basic objects: :class:`ModelGrid`,
which takes care of the bookkeeping aspects of storing and parsing
a given grid of stellar models; :class:`Isochrone`, which
takes care of the grid interpolation; and :class:`StarModel`,
which is the top-level interface for fitting stellar properties.

While ``isochrones`` comes packaged with two different model grids
(`MIST <http://waps.cfa.harvard.edu/MIST/>`_ and 
`Dartmouth <http://stellar.dartmouth.edu/models/>`_), 
it can be easily extended to other
model grids.  Of these two grid choices
(accessible through the :class:`MIST_Isochrone` and :class:`Dartmouth_Isochrone`
objects), MIST may be preferred because it covers a broader
range of age, mass, and metallicity than the Dartmouth Models.

For posterior sampling, ``isochrones`` defaults to MultiNest/PyMultiNest for sampling (see `here <http://astrobetter.com/wiki/MultiNest+Installation+Notes>`_ for installation instructions), but will fall back on `emcee
<http://dan.iel.fm/emcee/current/>`_ if (Py)MultiNest
is not installed.

Note that the first time you import any of the pre-packaged model
grids, it will download the required data for you.  If you like, you can
also download the data files directly and save them to ``~/.isochrones``
(or to a location defined by an ``$ISOCHRONES`` environment variable.)

.. note::

  The downloaded & unpacked data files, as well as some ancillary data 
  created for convenience by the package, will take up about 10 Gb 
  of disk space.  So if you're planning to use ``isochrones`` on a system 
  on which you have a home directory quota, you may wish to explicitly
  define an ``$ISOCHRONES`` environment variable somewhere where you have
  more storage space. 

I welcome community feedback to help improve this tool.  The code is
hosted at `GitHub <http://github.com/timothydmorton/isochrones>`_;
please feel free to contribute.  

.. .. note::

..   New in v0.9, fitting is now done by default using MultiNest, if 
..   available on your system via PyMultiNest.  If you wish to take 
..   advantage of this (highly recommended) feature, you can follow
..   `these <http://astrobetter.com/wiki/MultiNest+Installation+Notes>`_
..   instructions for installing MultiNest and PyMultinest.  If you do 
..   not have MultiNest available, the fits should still work using ``emcee``.

.. warning::
  
  If you have been a user of ``isochrones`` prior to v1.0, you will need
  to download the new grid data.  There has also been significant change
  to the code base.  Most backward compatibility should be preserved, but 
  `raise an issue <http://github.com/timothydmorton/isochrones/issures>`_
  if you have problems with the transition.


Installation
------------

To install, you can get the most recently released version from PyPI::

    pip install isochrones

Or you can clone from github::

    git clone https://github.com/timothydmorton/isochrones.git
    cd isochrones
    python setup.py install

The last command may require ``--user`` if you don't have root privileges.

After installation, run the test suite to check if everything works::

    nosetests isochrones

Be patient the first time you do this, as it will have to download ~1.5 Gb
of stellar grid data if you have not already done so.  
If there is a problem with the automated downloading,
you can also directly download the necessary files from 
`here <https://zenodo.org/record/161241>`_ and put them in ``~/.isochrones``
(or ``$ISOCHRONES``).

Basic Usage
---------

To find, for example, what a stellar model grid predicts for stellar
radius at a given mass, log(age), and metallicity::

    >>> from isochrones.mist import MIST_Isochrone
    >>> mist = MIST_Isochrone()
    >>> mist.radius(1.0, 9.7, 0.0) #M/Msun, log10(age), Fe/H
        1.0429784536817184

Importantly (for purposes of synthesizing populations of stars, e.g.),
you can pass array-like values, rather than single values::

    >>> mist.radius([0.8, 1.0, 1.2], 9.7, 0.0)
        array([ 0.75965718,  1.04297845,  1.96445299])

You can also interpolate broadband magnitudes, at a given 
distance and A_V extinction, as follows::

    >>> mass, age, feh, distance, AV = (0.95, 9.61, -0.2, 200, 0.2)
    >>> mist.mag['g'](mass, age, feh, distance, AV)
        11.788065261437591

You can see what bands are available to an :class:`Isochrone` object
by checking the ``bands`` attribute:

    >>> mist.bands
        ['B', 'G', 'H', 'J', 'K', 'Kepler', 'V', 'W1', 'W2', 'W3', 'g', 'i', 'r', 'z']

If you wish to use a different set of photometric bands, you may initialize the
:class:`Isochrone` with a ``bands`` keyword argument.  However, the 
:class:`ModelGrid` object used by the :class:`Isochrone` must know how to 
interpret that band name, and where to get that data, via the :func:`get_band`
method.

Fitting Stellar Properties
------------------------

If you want to estimate physical parameters for a star for which you
have measured spectroscopic properties, you would do something like
the following:

.. code-block:: python

    from isochrones import StarModel
    from isochrones.mist import MIST_Isochrone

    #spectroscopic properties (value, uncertainty)
    Teff = (5770, 80)
    logg = (4.44, 0.08)
    feh = (0.00, 0.10)
    
    mist = MIST_Isochrone()

    model  = StarModel(mist, Teff=Teff, logg=logg, feh=feh)
    model.fit()

The model now has a ``samples`` property that contains all of the
samples generated by the MultiNest/MCMC chain in a :class:`pandas.DataFrame`
object---or more specifically, it contains both the samples generated
directly from the chain and the corresponding values of all the model
properties (e.g. radius, synthetic photometry, etc.) evaluated at each
chain link.  You can also visualize the results using:

.. code-block:: python

   model.corner_physical()

Note that a :class:`isochrones.StarModel` can be initialized with any arguments
that correspond to properties predicted by the model grids---that is,
in addition to spectroscopic properties, apparent magnitudes (and
errors) may also be included among the keyword arguments, as well as parallax
(in miliarcseconds) and asteroseismic properties (``nu_max`` or ``delta_nu``).

After running the MultiNest/MCMC chain, you can save the results::

    model.save_hdf('starmodel.h5')

Which you can then read back in later as::

    model = StarModel.load_hdf('starmodel.h5')

In addition, if you would like to entertain the possibility of a star
having light from more than one component, you can also fit a binary
or triple star model by providing the additional keyword argument ``N=2``
or ``N=3`` to the :class:`StarModel` initialization.  You can also set up
a :class:`StarModel` that allows for light from multiple stars to be blended
in some bandpasses but resolved in others.  See the 
`demo notebook <https://github.com/timothydmorton/isochrones/blob/master/notebooks/demo.ipynb>`_ for more details on how to do this.

The easiest way to initialize and fit a :class:`StarModel` is to create a 
``star.ini`` file in a directory called ``mystar`` (for example), and then 
run the ``starfit`` command-line script that gets installed with ``isochrones``.
Again, see the `demo notebook <https://github.com/timothydmorton/isochrones/blob/master/notebooks/demo.ipynb>`_ for more details.

API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   api
