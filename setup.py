from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name = "isochrones",
    version = "0.1",
    description = "Defines objects for interpolating stellar model grids.",
    long_description = readme(),
    author = "Timothy D. Morton",
    author_email = "tim.morton@gmail.com",
    url = "https://github.com/timothydmorton/isochrones",
    packages = find_packages(),
    package_data = {'':['data/*']},
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    install_requires=['plotutils','pandas>=0.14','astropy>=0.3','emcee>=2.0'],
    zip_safe=False
) 
