import re

from ..bc import BolometricCorrectionGrid


class MISTBolometricCorrectionGrid(BolometricCorrectionGrid):
    name = "mist"

    phot_bands = dict(
        UBVRIplus=[
            "Bessell_U",
            "Bessell_B",
            "Bessell_V",
            "Bessell_R",
            "Bessell_I",
            "2MASS_J",
            "2MASS_H",
            "2MASS_Ks",
            "Kepler_Kp",
            "Kepler_D51",
            "Hipparcos_Hp",
            "Tycho_B",
            "Tycho_V",
            "Gaia_G_DR2Rev",
            "Gaia_BP_DR2Rev",
            "Gaia_RP_DR2Rev",
            "Gaia_G_MAW",
            "Gaia_BP_MAWf",
            "Gaia_BP_MAWb",
            "Gaia_RP_MAW",
            "TESS",
            "Gaia_G_EDR3",
            "Gaia_BP_EDR3",
            "Gaia_RP_EDR3",
        ],
        WISE=["WISE_W1", "WISE_W2", "WISE_W3", "WISE_W4"],
        CFHT=["CFHT_u", "CFHT_g", "CFHT_r", "CFHT_i_new", "CFHT_i_old", "CFHT_z"],
        DECam=["DECam_u", "DECam_g", "DECam_r", "DECam_i", "DECam_z", "DECam_Y"],
        GALEX=["GALEX_FUV", "GALEX_NUV"],
        JWST=[
            "F070W",
            "F090W",
            "F115W",
            "F140M",
            "F150W2",
            "F150W",
            "F162M",
            "F164N",
            "F182M",
            "F187N",
            "F200W",
            "F210M",
            "F212N",
            "F250M",
            "F277W",
            "F300M",
            "F322W2",
            "F323N",
            "F335M",
            "F356W",
            "F360M",
            "F405N",
            "F410M",
            "F430M",
            "F444W",
            "F460M",
            "F466N",
            "F470N",
            "F480M",
        ],
        LSST=["LSST_u", "LSST_g", "LSST_r", "LSST_i", "LSST_z", "LSST_y"],
        PanSTARRS=["PS_g", "PS_r", "PS_i", "PS_z", "PS_y", "PS_w", "PS_open"],
        SkyMapper=["SkyMapper_u", "SkyMapper_v", "SkyMapper_g", "SkyMapper_r", "SkyMapper_i", "SkyMapper_z"],
        SPITZER=["IRAC_3.6", "IRAC_4.5", "IRAC_5.8", "IRAC_8.0"],
        UKIDSS=["UKIDSS_Z", "UKIDSS_Y", "UKIDSS_J", "UKIDSS_H", "UKIDSS_K"],
        SDSSugriz=["SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"],
        HST_ACSWF=["ACS_WFC_F435W", "ACS_WFC_F475W", "ACS_WFC_F502N", 
            "ACS_WFC_F550M", "ACS_WFC_F555W", "ACS_WFC_F606W", "ACS_WFC_F625W",
            "ACS_WFC_F658N", "ACS_WFC_F660N", "ACS_WFC_F775W", "ACS_WFC_F814W", 
            "ACS_WFC_F850LP", "ACS_WFC_F892N"],
        HST_ACSHR=["ACS_HRC_F220W", "ACS_HRC_F250W", "ACS_HRC_F330W", "ACS_HRC_F344N",
            "ACS_HRC_F435W", "ACS_HRC_F475W", "ACS_HRC_F502N", "ACS_HRC_F550M", "ACS_HRC_F555W", 
            "ACS_HRC_F606W", "ACS_HRC_F625W", "ACS_HRC_F658N", "ACS_HRC_F660N", "ACS_HRC_F775W", 
            "ACS_HRC_F814W", "ACS_HRC_F850LP", "ACS_HRC_F892N"],
        HST_WFC3=[
            "WFC3_UVIS_F200LP",
            "WFC3_UVIS_F218W",
            "WFC3_UVIS_F225W",
            "WFC3_UVIS_F275W",
            "WFC3_UVIS_F280N",
            "WFC3_UVIS_F300X",
            "WFC3_UVIS_F336W",
            "WFC3_UVIS_F343N",
            "WFC3_UVIS_F350LP",
            "WFC3_UVIS_F373N",
            "WFC3_UVIS_F390M",
            "WFC3_UVIS_F390W",
            "WFC3_UVIS_F395N",
            "WFC3_UVIS_F410M",
            "WFC3_UVIS_F438W",
            "WFC3_UVIS_F467M",
            "WFC3_UVIS_F469N",
            "WFC3_UVIS_F475W",
            "WFC3_UVIS_F475X",
            "WFC3_UVIS_F487N",
            "WFC3_UVIS_F502N",
            "WFC3_UVIS_F547M",
            "WFC3_UVIS_F555W",
            "WFC3_UVIS_F600LP",
            "WFC3_UVIS_F606W",
            "WFC3_UVIS_F621M",
            "WFC3_UVIS_F625W",
            "WFC3_UVIS_F631N",
            "WFC3_UVIS_F645N",
            "WFC3_UVIS_F656N",
            "WFC3_UVIS_F657N",
            "WFC3_UVIS_F658N",
            "WFC3_UVIS_F665N",
            "WFC3_UVIS_F673N",
            "WFC3_UVIS_F680N",
            "WFC3_UVIS_F689M",
            "WFC3_UVIS_F763M",
            "WFC3_UVIS_F775W",
            "WFC3_UVIS_F814W",
            "WFC3_UVIS_F845M",
            "WFC3_UVIS_F850LP",
            "WFC3_UVIS_F953N",
            "WFC3_IR_F098M",
            "WFC3_IR_F105W",
            "WFC3_IR_F110W",
            "WFC3_IR_F125W",
            "WFC3vIR_F126N",
            "WFC3_IR_F127M",
            "WFC3_IR_F128N",
            "WFC3_IR_F130N",
            "WFC3_IR_F132N",
            "WFC3_IR_F139M",
            "WFC3_IR_F140W",
            "WFC3_IR_F153M",
            "WFC3_IR_F160W",
            "WFC3_IR_F164N",
            "WFC3_IR_F167N"
        ],
        HST_WFPC2=[
            "WFPC2_F218W",
            "WFPC2_F255W",
            "WFPC2_F300W",
            "WFPC2_F336W",
            "WFPC2_F439W",
            "WFPC2_F450W",
            "WFPC2_F555W",
            "WFPC2_F606W",
            "WFPC2_F622W",
            "WFPC2_F675W",
            "WFPC2_F791W",
            "WFPC2_F814W",
            "WFPC2_F850LP"
        ]

    )

    default_bands = ("J", "H", "K", "G", "BP", "RP", "W1", "W2", "W3", "TESS", "Kepler")

    def get_df(self, *args, **kwargs):
        df = super().get_df(*args, **kwargs)
        return df.xs(3.1, level="Rv")

    @classmethod
    def get_band(cls, b, **kwargs):
        """Defines what a "shortcut" band name refers to.  Returns phot_system, band

        """
        phot = None

        # Default to SDSS for these
        if b in ["u", "g", "r", "i", "z"]:
            phot = "SDSSugriz"
            band = "SDSS_{}".format(b)
        elif b in ["U", "B", "V", "R", "I"]:
            phot = "UBVRIplus"
            band = "Bessell_{}".format(b)
        elif b in ["J", "H", "Ks"]:
            phot = "UBVRIplus"
            band = "2MASS_{}".format(b)
        elif b == "K":
            phot = "UBVRIplus"
            band = "2MASS_Ks"
        elif b in ["kep", "Kepler", "Kp"]:
            phot = "UBVRIplus"
            band = "Kepler_Kp"
        elif b == "TESS":
            phot = "UBVRIplus"
            band = "TESS"
        elif b in ["W1", "W2", "W3", "W4"]:
            phot = "WISE"
            band = "WISE_{}".format(b)
        elif b in ("G", "BP", "RP"):
            phot = "UBVRIplus"
            band = "Gaia_{}_DR2Rev".format(b)
            if "version" in kwargs:
                if kwargs["version"] in ("1.1", "1.2"):
                    band += "_DR2Rev"
        elif b == "Bp":
            phot = "UBVRIplus"
            band = "Gaia_BP_DR2Rev"
            if "version" in kwargs:
                if kwargs["version"] in ("1.1", "1.2"):
                    band += "_DR2Rev"
        elif b == "Rp":
            phot = "UBVRIplus"
            band = "Gaia_RP_DR2Rev"
            if "version" in kwargs:
                if kwargs["version"] in ("1.1", "1.2"):
                    band += "_DR2Rev"
        else:
            m = re.match("([a-zA-Z]+)_([a-zA-Z_]+)", b)
            if m:
                if m.group(1) in cls.phot_bands.keys():
                    phot = m.group(1)
                    if phot == "PanSTARRS":
                        band = "PS_{}".format(m.group(2))
                    else:
                        band = m.group(0)
                elif m.group(1) in ["UK", "UKIRT"]:
                    phot = "UKIDSS"
                    band = "UKIDSS_{}".format(m.group(2))

        if phot is None:
            for system, bands in cls.phot_bands.items():
                if b in bands:
                    phot = system
                    band = b
                    break
            if phot is None:
                raise ValueError("MIST grids cannot resolve band {}!".format(b))
        return phot, band
