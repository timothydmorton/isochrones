from setuptools import setup, find_packages
import os,sys

import re, shutil
from tempfile import mkstemp
import subprocess as sp

def readme():
    with open('README.rst') as f:
        return f.read()

# Inject version from 'git describe' into __init__.py
try:
    vsn = sp.check_output(['git','describe']).strip()

    fh, abs_path = mkstemp()
    initfile = os.path.join('isochrones','__init__.py')
    with open(abs_path, 'w') as new_file:
        with open(initfile) as old_file:
            for line in old_file:
                if re.match('__version__', line):
                    new_file.write("__version__ = '{}'\n".format(vsn))
                else:
                    new_file.write(line)
    os.close(fh)
    os.remove(initfile)
    shutil.move(abs_path, initfile)
except:
    raise

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__ISOCHRONES_SETUP__ = True
import isochrones
version = isochrones.__version__

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Push a new tag to GitHub.
if "tag" in sys.argv:
    os.system("git tag -a {0} -m 'version {0}'".format(version))
    os.system("git push --tags")
    sys.exit()

setup(name = "isochrones",
    version = version,
    description = "Defines objects for interpolating stellar model grids.",
    long_description = readme(),
    author = "Timothy D. Morton",
    author_email = "tim.morton@gmail.com",
    url = "https://github.com/timothydmorton/isochrones",
    packages = find_packages(),
    package_data = {'':['data/*']},
    scripts = ['scripts/starfit',
               'scripts/batch_starfit',
               'scripts/starmodel-select',
               'scripts/starfit-summarize',
               'scripts/isochrones-dartmouth_write_hdf',
               'scripts/isochrones-dartmouth_write_tri'],
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    install_requires=['pandas>=0.14','astropy>=0.3','emcee>=2.0',
                      'numpy>=1.9', 'tables>=3.0',
                      'asciitree', 'corner'],
    zip_safe=False
) 
