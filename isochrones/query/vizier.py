from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd

from .query import EmptyQueryError
from .catalog import Catalog


class VizierCatalog(Catalog):
    columns = ("**", "_r", "_RAJ2000", "_DEJ2000")

    def _run_query(self):
        if self._empty:
            raise EmptyQueryError("{} is empty!".format(self))
        try:
            self._table = Vizier(columns=list(self.columns)).query_region(
                self.query_coords, radius=self.query.radius, catalog=self.vizier_name
            )[0]
        except IndexError:
            self._empty = True
            raise EmptyQueryError("{} returns empty!".format(self))
        self._coords = SkyCoord(self._table["_RAJ2000"], self._table["_DEJ2000"], unit="deg")
        self._table["PA"] = self.coords.position_angle(self.query_coords).deg
        # self._table['separation'] = self.coords.separation(self.query_coords).arcsec


class TwoMASS(VizierCatalog):
    name = "twomass"
    vizier_name = "2mass"
    epoch = 2000.0
    bands = {"Jmag": "J", "Hmag": "H", "Kmag": "K"}
    id_column = "_2MASS"


class Tycho2(VizierCatalog):
    name = "Tycho2"
    vizier_name = "tycho2"
    epoch = 2000
    bands = {"BTmag": "BT", "VTmag": "VT"}
    conversions = ["B", "V"]

    def get_id(self, brightest=False):
        if brightest:
            row = self.brightest
        else:
            row = self.closest

        return "{:.0f}-{:.0f}-{:.0f}".format(row["TYC1"], row["TYC2"], row["TYC3"])

    def V(self, brightest=False):
        """
        http://www.aerith.net/astro/color_conversion.html
        """
        mags = self.get_photometry(brightest=brightest, convert=False)
        VT, dVT = mags["VT"]
        BT, dBT = mags["BT"]
        if -0.25 < BT - VT < 2.0:
            (a, b, c, d) = (0.00097, 0.1334, 0.05486, 0.01998)
            V = VT + a - b * (BT - VT) + c * (BT - VT) ** 2 - d * (BT - VT) ** 3

            dVdVT = 1 + b - 2 * c * (BT - VT) + 3 * d * (BT - VT) ** 2
            dVdBT = -b + 2 * c * (BT - VT) - 3 * d * (BT - VT) ** 2
            dV = np.sqrt((dVdVT ** 2 * dVT ** 2) + (dVdBT ** 2 * dBT ** 2))

        else:
            raise ValueError("BT-VT outside of range to convert")

        return V, dV

    def BmV(self, brightest=False):
        mags = self.get_photometry(brightest=brightest, convert=False)
        VT, dVT = mags["VT"]
        BT, dBT = mags["BT"]
        if 0.5 < (BT - VT) < 2.0:
            (e, f, g) = (0.007813, 0.1489, 0.03384)
            BmV = (BT - VT) - e * (BT - VT) - f * (BT - VT) ** 2 + g * (BT - VT) ** 3

            dBmVdVT = -1 + e + 2 * f * (BT - VT) - 3 * g * (BT - VT) ** 2
            dBmVdBT = -dBmVdVT

        elif -0.25 < (BT - VT) < 0.5:
            (h, i, j) = (0.006, 0.1069, 0.1459)
            BmV = (BT - VT) - h - i * (BT - VT) + j * (BT - VT) ** 2

            dBmVdVT = -1 - i - 2 * j * (BT - VT)
            dBmVdBT = -dBmVdVT

        else:
            raise ValueError("BT-VT outside of range to convert")

        dBmV = np.sqrt((dBmVdVT ** 2 * dVT ** 2) + (dBmVdBT ** 2 * dBT ** 2))
        return BmV, dBmV

    def B(self, brightest=False):
        BmV, dBmV = self.BmV(brightest=brightest)
        V, dV = self.V(brightest=brightest)

        B = BmV + V
        dB = np.sqrt(dBmV ** 2 + dV ** 2)
        return B, dB


class WISE(VizierCatalog):
    name = "WISE"
    vizier_name = "allwise"
    epoch = 2000
    bands = {"W1mag": "W1", "W2mag": "W2", "W3mag": "W3"}  # W4 left out.
    id_column = "AllWISE"


class Gaia(VizierCatalog):
    name = "Gaia"
    vizier_name = "I/345/gaia2"
    epoch = 2015.5
    bands = {"Gmag": "G", "BPmag": "BP", "RPmag": "RP"}
    id_column = "Source"

    @property
    def is_good(self):
        """Quality cut from Appendix B of https://arxiv.org/pdf/1804.09378.pdf
        """
        t = self._table
        good = t["RPlx"] > 10
        good &= t["RFG"] > 50
        good &= t["RFRP"] > 20
        good &= t["RFBP"] > 20

        # Dropping the following, because the stars I need always fail...?
        # good &= t['E_BP-RP_'] < 1.3 + 0.06 * (t['BPmag'] - t['RPmag'])**2
        # good &= t['E_BP-RP_'] > 1.0 + 0.015 * (t['BPmag'] - t['RPmag'])**2

        good &= t["Nper"] > 8

        factor = np.where(np.exp(-0.4 * (t["Gmag"] - 19.5)) < 1, 1.0, np.exp(-0.4 * (t["Gmag"] - 19.5)))
        good &= t["chi2AL"] / (t["NgAL"] - 5) < 1.44 * factor

        good = pd.Series(good, dtype=bool)
        good[np.isnan(good)] = False
        return good
