import re

from ..bc import BolometricCorrectionGrid


class MISTBolometricCorrectionGrid(BolometricCorrectionGrid):
    name = 'mist'

    phot_bands = dict(UBVRIplus=['Bessell_U', 'Bessell_B', 'Bessell_V',
                                 'Bessell_R', 'Bessell_I', '2MASS_J', '2MASS_H', '2MASS_Ks',
                                 'Kepler_Kp', 'Kepler_D51', 'Hipparcos_Hp',
                                 'Tycho_B', 'Tycho_V', 'Gaia_G_DR2Rev', 'Gaia_BP_DR2Rev',
                                 'Gaia_RP_DR2Rev', 'TESS'],
                      WISE=['WISE_W1', 'WISE_W2', 'WISE_W3', 'WISE_W4'],
                      CFHT=['CFHT_u', 'CFHT_g', 'CFHT_r',
                            'CFHT_i_new', 'CFHT_i_old', 'CFHT_z'],
                      DECam=['DECam_u', 'DECam_g', 'DECam_r',
                             'DECam_i', 'DECam_z', 'DECam_Y'],
                      GALEX=['GALEX_FUV', 'GALEX_NUV'],
                      JWST=['F070W', 'F090W', 'F115W', 'F140M',
                            'F150W2', 'F150W', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W',
                            'F210M', 'F212N', 'F250M', 'F277W', 'F300M', 'F322W2', 'F323N',
                            'F335M', 'F356W', 'F360M', 'F405N', 'F410M', 'F430M', 'F444W',
                            'F460M', 'F466N', 'F470N', 'F480M'],
                      LSST=['LSST_u', 'LSST_g', 'LSST_r',
                            'LSST_i', 'LSST_z', 'LSST_y'],
                      PanSTARRS=['PS_g', 'PS_r', 'PS_i', 'PS_z',
                                 'PS_y', 'PS_w', 'PS_open'],
                      SkyMapper=['SkyMapper_u', 'SkyMapper_v', 'SkyMapper_g',
                                 'SkyMapper_r', 'SkyMapper_i', 'SkyMapper_z'],
                      SPITZER=['IRAC_3.6', 'IRAC_4.5', 'IRAC_5.8', 'IRAC_8.0'],
                      UKIDSS=['UKIDSS_Z', 'UKIDSS_Y', 'UKIDSS_J',
                              'UKIDSS_H', 'UKIDSS_K'],
                      SDSSugriz=['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z'])

    default_bands = ('J', 'H', 'K', 'G', 'BP', 'RP', 'W1', 'W2', 'W3', 'TESS', 'Kepler')

    def get_df(self, *args, **kwargs):
        df = super().get_df(*args, **kwargs)
        return df.xs(3.1, level='Rv')

    @classmethod
    def get_band(cls, b, **kwargs):
        """Defines what a "shortcut" band name refers to.  Returns phot_system, band

        """
        phot = None

        # Default to SDSS for these
        if b in ['u', 'g', 'r', 'i', 'z']:
            phot = 'SDSSugriz'
            band = 'SDSS_{}'.format(b)
        elif b in ['U', 'B', 'V', 'R', 'I']:
            phot = 'UBVRIplus'
            band = 'Bessell_{}'.format(b)
        elif b in ['J', 'H', 'Ks']:
            phot = 'UBVRIplus'
            band = '2MASS_{}'.format(b)
        elif b == 'K':
            phot = 'UBVRIplus'
            band = '2MASS_Ks'
        elif b in ['kep', 'Kepler', 'Kp']:
            phot = 'UBVRIplus'
            band = 'Kepler_Kp'
        elif b == 'TESS':
            phot = 'UBVRIplus'
            band = 'TESS'
        elif b in ['W1', 'W2', 'W3', 'W4']:
            phot = 'WISE'
            band = 'WISE_{}'.format(b)
        elif b in ('G', 'BP', 'RP'):
            phot = 'UBVRIplus'
            band = 'Gaia_{}_DR2Rev'.format(b)
            if 'version' in kwargs:
                if kwargs['version'] in ('1.1', '1.2'):
                    band += '_DR2Rev'
        elif b == 'Bp':
            phot = 'UBVRIplus'
            band = 'Gaia_BP_DR2Rev'
            if 'version' in kwargs:
                if kwargs['version'] in ('1.1', '1.2'):
                    band += '_DR2Rev'
        elif b == 'Rp':
            phot = 'UBVRIplus'
            band = 'Gaia_RP_DR2Rev'
            if 'version' in kwargs:
                if kwargs['version'] in ('1.1', '1.2'):
                    band += '_DR2Rev'
        else:
            m = re.match('([a-zA-Z]+)_([a-zA-Z_]+)', b)
            if m:
                if m.group(1) in cls.phot_bands.keys():
                    phot = m.group(1)
                    if phot == 'PanSTARRS':
                        band = 'PS_{}'.format(m.group(2))
                    else:
                        band = m.group(0)
                elif m.group(1) in ['UK', 'UKIRT']:
                    phot = 'UKIDSS'
                    band = 'UKIDSS_{}'.format(m.group(2))

        if phot is None:
            for system, bands in cls.phot_bands.items():
                if b in bands:
                    phot = system
                    band = b
                    break
            if phot is None:
                raise ValueError('MIST grids cannot resolve band {}!'.format(b))
        return phot, band
