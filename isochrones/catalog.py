import re
import os

import numpy as np

try:
    import holoviews as hv
except ImportError:
    hv = None

from . import get_ichrone
from .starmodel import SingleStarModel, BinaryStarModel, TripleStarModel
from .utils import band_pairs


class StarCatalog(object):
    """Catalog of star measurements


    Parameters
    ----------
    df : `pandas.DataFrame`
        Table containing stellar measurements.  Names of uncertainty columns are
        tagged with `_unc`.  If `bands` is not provided, then names of photometric
        bandpasses will be determined by looking for columns tagged with `_mag`.

    bands ; list(str)
        List of photometric bandpasses in table.  If not provided, will be inferred.

    props : list(str)
        Names of other properties in table (e.g., `Teff`, `logg`, `parallax`, etc.).

    """

    def __init__(self, df, bands=None, props=None, no_uncs=False):
        self._df = df

        if bands is None:
            bands = []
            for c in df.columns:
                m = re.search("(.+)_mag$", c)
                if m:
                    bands.append(m.group(1))
        self.bands = tuple(bands)
        self.band_cols = tuple("{}_mag".format(b) for b in self.bands)

        self.props = tuple() if props is None else tuple(props)

        if not no_uncs:
            for c in self.band_cols + self.props:
                if c not in self.df.columns:
                    raise ValueError("{} not in DataFrame!".format(c))
                if not "{}_unc".format(c) in self.df.columns:
                    raise ValueError("{0} uncertainty ({0}_unc) not in DataFrame!".format(c))

        self._ds = None
        self._hr = None

    def __setstate__(self, odict):
        self.__dict__ = odict
        self._hr = None

    def __len__(self):
        return len(self.df)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, newdf):
        self._df = newdf
        self._ds = None
        self._hr = None

    def get_measurement(self, prop, values=False):
        return self.df[prop].values, self.df[prop + "_unc"].values

    def iter_bands(self, **kwargs):
        for b, col in zip(self.bands, self.band_cols):
            yield b, self.get_measurement(col, **kwargs)

    def iter_props(self, **kwargs):
        for p in self.props:
            yield p, self.get_measurement(p, **kwargs)

    @property
    def ds(self):
        if self._ds is None:
            df = self.df.copy()
            for b1, b2 in band_pairs(self.bands):
                mag1 = self.df["{}_mag".format(b1)]
                mag2 = self.df["{}_mag".format(b2)]

                df[b2] = mag2
                df["{0}-{1}".format(b1, b2)] = mag1 - mag2

            self._ds = hv.Dataset(df)

        return self._ds

    @property
    def hr(self):
        if self._hr is None:
            layout = []
            opts = dict(invert_yaxis=True, tools=["hover"])
            for b1, b2 in band_pairs(self.bands):
                kdims = ["{}-{}".format(b1, b2), "{}_mag".format(b1)]
                layout.append(hv.Points(self.ds, kdims=kdims, vdims=self.ds.kdims).options(**opts))
            self._hr = hv.Layout(layout)
        return self._hr

    def iter_models(self, ic=None, N=1):
        if ic is None:
            ic = get_ichrone("mist", bands=self.bands)

        mod_type = {1: SingleStarModel, 2: BinaryStarModel, 3: TripleStarModel}

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            mags = {b: (row["{}_mag".format(b)], row["{}_mag_unc".format(b)]) for b in self.bands}
            props = {p: (row[p], row["{}_unc".format(p)]) for p in self.props}
            yield mod_type[N](ic, **mags, **props, name=row.name)
            i += 1

    def write_ini(self, ic=None, root=".", N=1):
        if ic is None:
            ic = get_ichrone("mist", bands=self.bands)

        n_pre = int(np.log10(len(self)) // 2)  # log_100
        dirs = []
        for mod in self.iter_models(ic, N=N):
            path = os.path.join(root, str(mod.name)[:n_pre])
            mod.write_ini(root=path)
            dirs.append(os.path.abspath(os.path.join(path, mod.name)))

        return dirs
