isochrones
==========
.. image:: https://travis-ci.com/timothydmorton/isochrones.svg?branch=master
    :target: https://travis-ci.com/timothydmorton/isochrones

Provides simple interface for interacting with stellar model grids.

https://isochrones.readthedocs.io

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
