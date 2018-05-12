from setuptools import setup, find_packages
import os, sys


def readme():
    with open('README.rst') as f:
        return f.read()

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
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
    package_data = {'isochrones':['data/*', 'tests/star*/*.ini']},
    scripts = ['scripts/starfit',
               'scripts/batch_starfit',
               'scripts/starmodel-select',
               'scripts/starfit-summarize',
               'scripts/isochrones-dartmouth_write_tri'],
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    install_requires=['pandas>=0.14','astropy>=0.3','emcee>=2.0',
                      'numpy>=1.9', 'tables>=3.0', 'scipy>=0.19',
                      'asciitree', 'corner', 'astroquery',
                      'configobj'],
    zip_safe=False
)
