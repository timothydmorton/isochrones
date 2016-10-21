isochrones
==========
.. image:: https://zenodo.org/badge/6253/timothydmorton/isochrones.svg   
    :target: http://dx.doi.org/10.5281/zenodo.16304

Provides simple interface for interacting with stellar model grids.
Version 1.0 is now up on PyPI, with substantial changes, most notably
inclusion of the `MIST model grids <http://waps.cfa.harvard.edu/MIST/>`_. 
Along with broader coverage in all parameters, these grids also provide
synthetic photometry in the Gaia G band.

Installation
-----------

Install with ``pip install isochrones`` or by cloning the repository
and running ``python setup.py install``.

To deal correctly with extinction (important if you are trying to fit
any wide bandpasses such as Gaia G band), you will also need to install
the following packages::

    pip install git+git://github.com/mfouesneau/pyextinction
    pip install git+git://github.com/mfouesneau/pyphot
    pip install git+git://github.com/mfouesneau/pystellibs

Without installing these, there is still a default extinction law included,
from `Schlafly (2016) <http://e.schlaf.ly/apored/extcurve.html>`_. 

The package also depends on the suite of scientific and numerical libraries
contained within `Anaconda <http://www.continuum.io/downloads>`_.    

For usage details, `read the documentation <http://isochrones.rtfd.org>`_ or
check out the `demo notebook <http://nbviewer.ipython.org/github/timothydmorton/isochrones/blob/master/notebooks/demo.ipynb>`_.

Stellar fits will want to use ``MultiNest``/``pymultinest`` if those packages are installed (recommended); otherwise the fits will default to MCMC fitting using ``emcee``.

Basic Usage
------------

For simplest out-of-the-box usage after installation, make a file called ``star.ini`` that
looks something like this::
    
    Teff = 5770, 100
    feh = 0.0, 0.15
    logg = 4.5, 0.1
    V = 10.0, 0.05
    
Any combination of spectroscopic or [supported] photometric properties, as well
as parallax, can go as properties into this file.
    
Once you've made this file, type ``starfit`` at the command line.  Or if you want to be organized,
you can put ``star.ini`` in a folder called ``mystar`` [or whatever]
and run ``starfit mystar`` from the command line.  The ``starfit`` script
will create an HDF5 file containing the saved ``StarModel``, which you 
can load from python using ``StarModel.load_hdf``, as well as triangle
plots illustrating the fit.

Attribution
------------
If you use ``isochrones`` in your research, please cite `this ASCL reference <http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2015ascl.soft03010M&data_type=BIBTEX&db_key=AST&nocookieset=1>`_.
